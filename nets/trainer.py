import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
import yaml
import time
import pprint

from .callbacks import Callback

import traceback
import nets.functionals as fn


class Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()

def pprint(x):
    _utils_pp.pprint(x)
class Trainer:

    def __init__(self, max_epochs, callbacks, min_epochs=None, loss=None, device=None, dtype=None, lr=0.01, weight_decay=1e-3, verify_manifold=False, monitor_curvature=False):
        self.lr = lr
        self.weight_decay = weight_decay
        self.min_epochs = min_epochs
        self.epochs = max_epochs
        self.loss_fn = loss
        self.current_epoch = 0
        self.current_step = 0
        self.records = []
        for callback in callbacks:
            assert(isinstance(callback, Callback))
        self.callbacks = callbacks

        self.device_ = device
        self.dtype_ = dtype
        self.verify_manifold = verify_manifold  # Whether to verify manifold constraints
        self.monitor_curvature = monitor_curvature  # Whether to monitor curvature

        self.stop_fit_ = False
        self.optimizer = None

    def fit(self, model : torch.nn.Module, train_dataloader : torch.utils.data.DataLoader, val_dataloader : torch.utils.data.DataLoader):

        model = model.to(dtype=self.dtype_, device=self.device_)

        self.optimizer = model.configure_optimizers(lr=self.lr, weight_decay=self.weight_decay)

        [callback.on_fit_start(self, model) for callback in self.callbacks]

        timer = Timer()
        for epoch in range(self.epochs):

            self.current_epoch = epoch
            [callback.on_train_epoch_start(self, model) for callback in self.callbacks]

            self.train_epoch(model, train_dataloader)

            trn_res = self.test(model, train_dataloader)
            trn_res = {f'trn_{k}': v for k, v in trn_res.items()}

            val_res = self.test(model, val_dataloader)
            val_res = {f'val_{k}': v for k, v in val_res.items()}

            self.log_dict(trn_res)
            self.log_dict(val_res)

            # Output every epoch
            log_dict = trn_res | val_res
            
            # Add curvature monitoring
            if self.monitor_curvature and hasattr(model, 'manifold') and hasattr(model.manifold, 'k'):
                curvature_value = model.manifold.k.item()
                log_dict['curvature'] = curvature_value
                if hasattr(model, 'learnable_curvature') and model.learnable_curvature:
                    # Note: gradient may have been zeroed, just try to get it
                    if model.manifold.k.grad is not None:
                        log_dict['k_grad'] = model.manifold.k.grad.item()
            
            # Calculate ETA
            progress = (epoch + 1) / self.epochs if self.epochs > 0 else 1.0
            elapsed = timer.measure()
            eta = timer.measure(progress) if progress > 0 else '0s'
            
            print(f'epoch={epoch:3d}/{self.epochs-1} gd-step={self.current_step:5d} ETA:{elapsed}/{eta}', end=' ')
            [print(f"{k + '=':10}{v:6.4f}", end=' ') for k,v in log_dict.items()]
            print('')                                                                 # New line


            [callback.on_train_epoch_end(self, model) for callback in self.callbacks]

            if self.stop_fit_:
                break

        [callback.on_fit_end(self, model) for callback in self.callbacks]

    def stop_fit(self):
        if self.min_epochs and self.current_epoch > self.min_epochs:
            self.stop_fit_ = True
        elif self.min_epochs is None:
            self.stop_fit_ = True
        

    def train_epoch(self, model : torch.nn.Module, train_dataloader : torch.utils.data.DataLoader):


        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            [callback.on_train_batch_start(self, model, batch, batch_idx) for callback in self.callbacks]
            features, y = batch
            features['inputs'] = features['inputs'].to(dtype=self.dtype_, device=self.device_)
            y = y.to(device=self.device_)
            
            # If manifold verification is enabled, verify during forward pass
            if self.verify_manifold:
                pred, x_flatten = model(**features, verify_manifold=True)
                # Verify if features are on the manifold
                if batch_idx == 0 and self.current_epoch % 10 == 0:  # Print detailed info every 10 epochs
                    manifold_info = model.check_features_on_manifold(x_flatten, return_details=True)
                    if not manifold_info['all_on_manifold']:
                        print(f"Warning: Features not on manifold during training - max_error={manifold_info['max_error']:.6f}")
            else:
                pred, x_flatten = model(**features)
            cn_loss = self.loss_fn(pred, y)
            loss = cn_loss
            loss.backward()
            # Clip gradient separately for curvature parameters
            if hasattr(model, 'manifold') and hasattr(model.manifold, 'k') and model.manifold.k.requires_grad:
                if model.manifold.k.grad is not None:
                    torch.nn.utils.clip_grad_norm_([model.manifold.k], max_norm=1.0)
            self.optimizer.step()
            # Ensure curvature parameter k > 0 (numerical stability requirement, minimum value is eps=1e-5)
            if hasattr(model, 'manifold') and hasattr(model.manifold, 'k') and model.manifold.k.requires_grad:
                with torch.no_grad():
                    model.manifold.k.clamp_(min=model.manifold.min_k)
            self.optimizer.zero_grad()
            self.current_step += 1

    
    def test(self, model : torch.nn.Module, dataloader : torch.utils.data.DataLoader):

        model.eval()
        loss = 0

        y_true = []
        y_hat = []
        manifold_stats = []

        with torch.no_grad():
            for batch_ix, (features, y) in enumerate(dataloader):
                features['inputs'] = features['inputs'].to(dtype=self.dtype_, device=self.device_)
                y = y.to(device=self.device_)
                  
                features['inputs'] = features['inputs'].to(dtype=self.dtype_, device=self.device_)
                #domain = features['domains'].to(dtype=self.dtype_,device=self.device_)
                #feature=features['inputs'].unsqueeze(1)
                
                if self.verify_manifold:
                    pred, features_hyper = model(**features, verify_manifold=True)
                    # Collect manifold statistics
                    manifold_info = model.check_features_on_manifold(features_hyper, return_details=True)
                    manifold_stats.append(manifold_info)
                else:
                    pred, _ = model(**features)
                    
                #pred = model(features['inputs'])
                loss += self.loss_fn(pred, y).item()
                y_true.append(y)
                y_hat.append(pred.argmax(1))

        loss /= batch_ix + 1

        y_true_np = torch.cat(y_true).detach().cpu().numpy()
        y_hat_np = torch.cat(y_hat).detach().cpu().numpy()
        
        score = balanced_accuracy_score(y_true_np, y_hat_np).item()
        f1_macro = f1_score(y_true_np, y_hat_np, average='macro').item()
        
        result = dict(loss=loss, score=score, f1_macro=f1_macro)
        
        # If manifold verification is enabled, add manifold statistics
        if self.verify_manifold and manifold_stats:
            avg_max_error = sum(s['max_error'] for s in manifold_stats) / len(manifold_stats)
            avg_mean_error = sum(s['mean_error'] for s in manifold_stats) / len(manifold_stats)
            all_on_manifold = all(s['all_on_manifold'] for s in manifold_stats)
            result['manifold_ok'] = all_on_manifold
            result['manifold_max_error'] = avg_max_error
            result['manifold_mean_error'] = avg_mean_error

        return result


    def log_dict(self, dictionary):
        self.records.append(dictionary | dict(epoch=self.current_epoch))
