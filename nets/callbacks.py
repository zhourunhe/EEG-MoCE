from .batchnorm import SchedulableBatchNorm
from torch.types import Number
import torch
import os
import tempfile

class Callback:

    def on_fit_start(self, trainer, net):
        pass

    def on_train_epoch_start(self, trainer, net):
        pass

    def on_train_batch_start(self, trainer, net, batch, batch_idx):
        pass

    def on_train_epoch_end(self, trainer, net):
        pass

    def on_fit_end(self, trainer, net):
        pass

class ConstantMomentumBatchNormScheduler(Callback):
    def __init__(self, eta, eta_test) -> None:
        self.eta_ = eta
        self.eta_test_ = eta_test
        self.bn_modules_ = []


    def on_fit_start(self, trainer, net):
        if isinstance(net, torch.nn.Module):
            model = net
        else:
            raise NotImplementedError()
        # extract momentum batch norm parameters
        if model is not None:
            self.bn_modules_ = [m for m in model.modules() 
                if isinstance(m, SchedulableBatchNorm)]
        else:
            self.bn_modules_ = []

        for m in self.bn_modules_:
            m.set_eta(eta=self.eta_, eta_test = self.eta_test_)

    def __repr__(self) -> str:
        return f'ConstantMomentumBatchNormScheduler - eta={self.eta_:.3f}, eta_test={self.eta_test_:.3f}'


class MomentumBatchNormScheduler(ConstantMomentumBatchNormScheduler):
    def __init__(self, epochs : Number, bs : Number = 32, bs0 : Number = 64, tau0 : Number = 0.9) -> None:
        assert(bs <= bs0)
        super().__init__(1. - tau0, 1. - tau0 ** (bs/bs0))
        self.epochs = epochs
        self.rho = (bs/bs0) ** (1/self.epochs)
        self.tau0 = tau0
        self.bs = bs
        self.bs0 = bs0

    def __repr__(self) -> str:
        return f'MomentumBatchNormScheduler - eta={self.eta_:.3f}, eta_tst={self.eta_test_:.3f}'

    def on_train_epoch_start(self, trainer, net):
        # Handle edge case when epochs <= 1 to avoid division by zero
        if self.epochs <= 1:
            self.eta_ = 1.0
            w = 1.0
        else:
            self.eta_ = 1. - (self.rho ** (self.epochs * max(self.epochs - trainer.current_epoch,0)/(self.epochs-1)) - self.rho ** self.epochs)
            w = max(self.epochs - trainer.current_epoch,0)/(self.epochs-1)
        
        for m in self.bn_modules_:
            m.set_eta(eta = self.eta_)
        
        tau_test = self.tau0 ** (self.bs/self.bs0 * (1-w) + w * 1)
        self.eta_test_ = 1 - tau_test
        for m in self.bn_modules_:
            m.set_eta(eta_test = self.eta_test_)


class EarlyStopping(Callback):

    def __init__(self, metric='val_loss', higher_is_better=False, patience=15, verbose=False): 
                                                             
        self.tempdir = tempfile.TemporaryDirectory()     
        self.patience = patience
        self.metric = metric
        self.sign = -1 if higher_is_better else 1
        self.counter = 0
        self.best_score = self.sign * torch.Tensor([float('Inf')])
        self.best_epoch = -1
        self.verbose = verbose

    def on_train_epoch_end(self, trainer, net):                            

        current_score = self.sign * torch.Tensor([float('Inf')])
        for record in trainer.records[::-1]:                                             
            if record['epoch'] == trainer.current_epoch and self.metric in record:
                current_score = record[self.metric]
                break

        if current_score < self.best_score:      
            self.counter = 0
            self.best_score = current_score
            self.best_epoch = trainer.current_epoch
            if self.verbose:
                print(f'ES: new best score {self.best_score} for metric {self.metric} ...')
            self._save_checkpoint(net)
        else:
            self.counter += 1                 
        
        if self.counter >= self.patience:
            trainer.stop_fit()

    def _save_checkpoint(self, net):

        if self.verbose:
            print(f'ES: saving model ...')
        torch.save(net.state_dict(), os.path.join(self.tempdir.name, 'es_state_dict.pt'))  

    def on_fit_end(self, trainer, net):          
        # if early stopping was triggered
        path = os.path.join(self.tempdir.name, 'es_state_dict.pt')        
        if self.counter >= 0 and os.path.exists(path):
            if self.verbose:
                print(f'ES: loading best model ...')
            net.load_state_dict(torch.load(path))


class CurvatureWarmupScheduler(Callback):
    """Learning rate warmup scheduler for curvature parameters"""
    
    def __init__(self, warmup_epochs=None, warmup_steps=None, warmup_scale=0.01, 
                 base_lr_scale=0.1, mode='epoch', verbose=False):
        assert (warmup_epochs is not None) or (warmup_steps is not None), \
            "Must specify either warmup_epochs or warmup_steps"
        assert mode in ['epoch', 'step'], "mode must be 'epoch' or 'step'"
        
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.warmup_scale = warmup_scale
        self.base_lr_scale = base_lr_scale
        self.mode = mode
        self.verbose = verbose
        self.curvature_param_group_idx = None
        self.base_lr = None
        
    def on_fit_start(self, trainer, net):
        """Initialize: find curvature parameter group"""
        if trainer.optimizer is None:
            return
        
        # Get base_lr_scale from model configuration (if available)
        if hasattr(net, 'curvature_lr_scale'):
            self.base_lr_scale = net.curvature_lr_scale
        
        # Collect curvature parameters from all manifolds
        curvature_params = []
        manifold_names = ['manifold', 'eeg_manifold', 'audio_manifold', 'vision_manifold', 'fusion_manifold']
        for mname in manifold_names:
            if hasattr(net, mname):
                manifold = getattr(net, mname)
                if hasattr(manifold, 'k') and manifold.k.requires_grad:
                    curvature_params.append(manifold.k)
            
        # Find curvature parameter group (parameter group containing manifold.k)
        for idx, group in enumerate(trainer.optimizer.param_groups):
            for param in group['params']:
                # Check if it's a curvature parameter
                if any(param is k for k in curvature_params):
                    self.curvature_param_group_idx = idx
                    # base_lr should be the normal learning rate (lr * base_lr_scale)
                    self.base_lr = trainer.lr * self.base_lr_scale
                    # Initialize learning rate during warmup
                    initial_lr = self.base_lr * self.warmup_scale
                    group['lr'] = initial_lr
                    if self.verbose:
                        print(f'CurvatureWarmup: Found curvature parameter group (idx={idx}), containing {len(curvature_params)} curvature parameters')
                        print(f'  base_lr={self.base_lr:.6f}, initial_lr={initial_lr:.6f} (warmup_scale={self.warmup_scale})')
                    break
            if self.curvature_param_group_idx is not None:
                break
        
        if self.curvature_param_group_idx is None and self.verbose:
            print('CurvatureWarmup: Warning - Curvature parameter group not found, warmup will not take effect')
    
    def _update_curvature_lr(self, trainer, progress):
        """Update learning rate for curvature parameters"""
        if self.curvature_param_group_idx is None or trainer.optimizer is None:
            return
            
        if progress < 1.0:
            current_scale = self.warmup_scale + (1.0 - self.warmup_scale) * progress
        else:
            current_scale = 1.0
        
        group = trainer.optimizer.param_groups[self.curvature_param_group_idx]
        new_lr = self.base_lr * current_scale
        group['lr'] = new_lr
        
        if self.verbose and (trainer.current_step % 100 == 0 or progress >= 1.0):
            print(f'CurvatureWarmup: step={trainer.current_step}, epoch={trainer.current_epoch}, '
                  f'progress={progress:.3f}, lr_scale={current_scale:.4f}, lr={new_lr:.6f}')
    
    def on_train_epoch_start(self, trainer, net):
        """Update learning rate at the start of each epoch"""
        if self.mode == 'epoch' and self.warmup_epochs is not None:
            progress = min(trainer.current_epoch / self.warmup_epochs, 1.0)
            self._update_curvature_lr(trainer, progress)
    
    def on_train_batch_start(self, trainer, net, batch, batch_idx):
        """Update learning rate at the start of each step"""
        if self.mode == 'step' and self.warmup_steps is not None:
            progress = min(trainer.current_step / self.warmup_steps, 1.0)
            self._update_curvature_lr(trainer, progress)
    
    def __repr__(self) -> str:
        if self.mode == 'epoch':
            return f'CurvatureWarmupScheduler(mode={self.mode}, warmup_epochs={self.warmup_epochs}, ' \
                   f'warmup_scale={self.warmup_scale}, base_lr_scale={self.base_lr_scale})'
        else:
            return f'CurvatureWarmupScheduler(mode={self.mode}, warmup_steps={self.warmup_steps}, ' \
                   f'warmup_scale={self.warmup_scale}, base_lr_scale={self.base_lr_scale})'

