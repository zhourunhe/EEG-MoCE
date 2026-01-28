"""
Multimodal Hyperbolic Emotion Recognition (EEG-MoCE)

Uses HMultimodalNet with EEG, Audio, Vision encoders and hyperbolic cross-attention fusion.
Data format: EEG (B, num_electrodes, chunk_size), Audio (B, 128, 1024), Vision (B, time_length, 35).
"""

import argparse
import torch
import sklearn
import pandas as pd
from nets.utils.data import StratifiedMultimodalDataLoader, MultimodalDomainDataset
from nets.trainer import Trainer, Timer, pprint
from nets.callbacks import EarlyStopping, MomentumBatchNormScheduler, CurvatureWarmupScheduler
from nets.HMultimodalNet import HMultimodalNet
import nets.functionals as fn
from copy import deepcopy
import nets.batchnorm as bn
import numpy as np
import os
import glob
from datetime import datetime


# Configuration

parser = argparse.ArgumentParser(description='Multimodal Hyperbolic Emotion Recognition')

# Data paths
parser.add_argument('--eeg_path', type=str, 
                    default='./data/EAV_EEG',
                    help='Path to EEG data (subject_XX_eeg.dat files)')
parser.add_argument('--audio_path', type=str, 
                    default='./data/EAV_Audio',
                    help='Path to Audio data (subject_XX_au.dat files)')
parser.add_argument('--vision_path', type=str, 
                    default='./data/EAV_Vision_AU',
                    help='Path to Vision AU data (subjectXX/Video_AU folders)')

# Cross-validation
parser.add_argument('--cv_method', type=str, default='kfold', choices=['loso', 'kfold'],
                    help='Cross-validation method')

parser.add_argument('--curvature_init', type=float, default=0.5,
                    help='Initial curvature value')
parser.add_argument('--curvature_lr_scale', type=float, default=1.0,
                    help='Learning rate scale for curvature')
parser.add_argument('--curvature_prior_init', type=float, default=0.3,
                    help='Initial value for curvature prior weight λ')

# Training params
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')

# Debug params
parser.add_argument('--verify_manifold', type=lambda x: x.lower() == 'true', default=False,
                    help='Verify points are on manifold during forward pass (True/False, default: False)')

args = parser.parse_args()

# Network configuration
cfg = dict(
    epochs=args.epochs,
    batch_size_train=args.batch_size,
    domains_per_batch=5,
    validation_size=0.2,
    evaluation='inter-subject',
    dtype=torch.float64,
    training=True,
    lr=args.lr,
    weight_decay=args.weight_decay,
    cv_method=args.cv_method,
    mdl_kwargs=dict(
        bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
        domain_adaptation=True,
        learnable_curvature=True,
        curvature_lr_scale=args.curvature_lr_scale,
        curvature_init=args.curvature_init,
        curvature_mode='mean',
        dropout=args.dropout,
        curvature_prior_init=args.curvature_prior_init,
    )
)

# GPU setup
gpu_id = args.gpu_id

if torch.cuda.is_available():
    if gpu_id < torch.cuda.device_count():
        device = torch.device(f'cuda:{gpu_id}')
        print(f'Using GPU: {gpu_id} ({torch.cuda.get_device_name(gpu_id)})')
    else:
        device = torch.device('cuda:0')
        print(f'Specified GPU {gpu_id} does not exist, using default GPU 0')
else:
    device = torch.device('cpu')
    print('CUDA is not available, using CPU')


# Load Multimodal Data

n_classes = 5  # EAV dataset has 5 emotion classes

print("\n" + "=" * 80)
print("Loading Multimodal Dataset")
print("=" * 80)

# Load multimodal data with trial_id alignment
eeg_data, audio_data, vision_data, labels, subjects, sessions, trial_ids = fn.load_multimodal_dataset(
    eeg_path=args.eeg_path,
    audio_path=args.audio_path,
    vision_path=args.vision_path,
    min_vision_trial_length=603
)

print(f"\nData shapes:")
print(f"  EEG: {eeg_data.shape}")
print(f"  Audio: {audio_data.shape}")
print(f"  Vision: {vision_data.shape}")
print(f"  Labels: {labels.shape}")

# Group subject-session as domain (EAV Dataset has only one session)
domain_labels = [f"{sub}_{sess}" for sub, sess in zip(subjects, sessions)]
domain = sklearn.preprocessing.LabelEncoder().fit_transform(domain_labels)
assert eeg_data.shape[0] == audio_data.shape[0] == vision_data.shape[0] == labels.shape[0] == domain.shape[0]
domain = torch.from_numpy(domain)

def sfuda_offline(eeg_ds, audio_ds, vision_ds, domains_ds, model: HMultimodalNet):
    """Source-Free Unsupervised Domain Adaptation"""
    model.eval()
    model.domainadapt_finetune(
        eeg_ds.to(dtype=cfg['dtype'], device=device),
        audio_ds.to(dtype=cfg['dtype'], device=device),
        vision_ds.to(dtype=cfg['dtype'], device=device),
        domains_ds,
        None
    )


# Custom Trainer for Multimodal Model

class MultimodalTrainer(Trainer):
    """Trainer for Multimodal data"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, model, train_dataloader, val_dataloader):
        """Override fit to add multimodal curvature logging"""
        from nets.trainer import Timer
        
        self.model = model
        optimizers = model.configure_optimizers(lr=self.lr, weight_decay=self.weight_decay)
        if isinstance(optimizers, list):
            self.optimizers = optimizers
        else:
            self.optimizers = [optimizers]
        self.stop_fit_ = False
        
        [callback.on_fit_start(self, model) for callback in self.callbacks]
        
        timer = Timer()
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            [callback.on_train_epoch_start(self, model) for callback in self.callbacks]
            
            self.train_epoch(model, train_dataloader)
            
            trn_res = self.test(model, train_dataloader, return_contributions=False)
            trn_res = {f'trn_{k}': v for k, v in trn_res.items()}
            
            val_res = self.test(model, val_dataloader, return_contributions=False)
            val_res = {f'val_{k}': v for k, v in val_res.items()}
            
            self.log_dict(trn_res)
            self.log_dict(val_res)
            
            progress = (epoch + 1) / self.epochs if self.epochs > 0 else 1.0
            elapsed = timer.measure()
            eta = timer.measure(progress) if progress > 0 else '0s'
            
            print(f'ep={epoch:3d}/{self.epochs-1} ETA:{elapsed}/{eta}', end=' ')
            print(f"trn={trn_res.get('trn_score', 0):.3f}", end=' ')
            print(f"val={val_res.get('val_score', 0):.3f}", end=' ')
            print(f"loss={val_res.get('val_loss', 0):.3f}", end=' ')
            
            if self.monitor_curvature:
                curvatures = model.get_curvatures()
                print(f"k_e={curvatures['eeg']:.3f}", end=' ')
                print(f"k_a={curvatures['audio']:.3f}", end=' ')
                print(f"k_v={curvatures['vision']:.3f}", end=' ')
                print(f"k_f={curvatures['fusion']:.3f}", end='')
            print('')
            
            [callback.on_train_epoch_end(self, model) for callback in self.callbacks]
            
            if self.stop_fit_:
                break
        
        [callback.on_fit_end(self, model) for callback in self.callbacks]
    
    def train_epoch(self, model, train_dataloader):
        """Override train_epoch to handle multimodal inputs"""
        model.train()
        
        for batch_idx, batch in enumerate(train_dataloader):
            [callback.on_train_batch_start(self, model, batch, batch_idx) for callback in self.callbacks]
            
            features, y = batch
            eeg_inputs = features['eeg_inputs'].to(dtype=self.dtype_, device=self.device_)
            audio_inputs = features['audio_inputs'].to(dtype=self.dtype_, device=self.device_)
            vision_inputs = features['vision_inputs'].to(dtype=self.dtype_, device=self.device_)
            domains = features['domains'].to(device=self.device_)
            y = y.to(device=self.device_)
            
            if self.verify_manifold:
                pred, features_out = model(eeg_inputs, audio_inputs, vision_inputs, domains, verify_manifold=True)
            else:
                pred, features_out = model(eeg_inputs, audio_inputs, vision_inputs, domains)
            
            loss = self.loss_fn(pred, y)
            loss.backward()
            
            for manifold_name in ['eeg_manifold', 'audio_manifold', 'vision_manifold']:
                if hasattr(model, manifold_name):
                    manifold = getattr(model, manifold_name)
                    if hasattr(manifold, 'k') and manifold.k.requires_grad and manifold.k.grad is not None:
                        torch.nn.utils.clip_grad_norm_([manifold.k], max_norm=1.0)
            
            for opt in self.optimizers:
                opt.step()
            
            for manifold_name in ['eeg_manifold', 'audio_manifold', 'vision_manifold']:
                if hasattr(model, manifold_name):
                    manifold = getattr(model, manifold_name)
                    if hasattr(manifold, 'k') and manifold.k.requires_grad:
                        with torch.no_grad():
                            manifold.k.clamp_(min=manifold.min_k)
            
            for opt in self.optimizers:
                opt.zero_grad()
            self.current_step += 1
    
    def test(self, model, dataloader, return_contributions=True):
        """Override test to handle multimodal inputs"""
        model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        n_batches = 0
        all_contributions = []
        
        with torch.no_grad():
            for batch in dataloader:
                features, labels = batch
                
                eeg_inputs = features['eeg_inputs'].to(dtype=self.dtype_, device=self.device_)
                audio_inputs = features['audio_inputs'].to(dtype=self.dtype_, device=self.device_)
                vision_inputs = features['vision_inputs'].to(dtype=self.dtype_, device=self.device_)
                domains = features['domains'].to(device=self.device_)
                labels = labels.to(device=self.device_)
                
                logits, features_out = model(
                    eeg_inputs, audio_inputs, vision_inputs, domains,
                    return_contributions=return_contributions
                )
                
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                n_batches += 1
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if return_contributions and 'contributions' in features_out:
                    all_contributions.append(features_out['contributions'])
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        from sklearn.metrics import balanced_accuracy_score, f1_score
        score = balanced_accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        
        result = {'score': score, 'f1_macro': f1_macro, 'loss': avg_loss}
        
        if all_contributions:
            avg_eeg = np.mean([c['eeg_contribution'] for c in all_contributions])
            avg_audio = np.mean([c['audio_contribution'] for c in all_contributions])
            avg_vision = np.mean([c['vision_contribution'] for c in all_contributions])
            result['eeg_contribution'] = avg_eeg
            result['audio_contribution'] = avg_audio
            result['vision_contribution'] = avg_vision
        
        return result


# Training Loop

random_seed = 42
import random
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
records = []
all_curvatures = []
all_curvature_priors = []
saved_models = []

# Convert to tensors
X_eeg = torch.from_numpy(eeg_data)
X_audio = torch.from_numpy(audio_data)
X_vision = torch.from_numpy(vision_data)
y = sklearn.preprocessing.LabelEncoder().fit_transform(labels)
y = torch.from_numpy(y)

# Cross-validation
subject_count = len(np.unique(subjects))

cv_method = cfg.get('cv_method', 'loso').lower()

if cv_method == 'loso':
    cv_outer = sklearn.model_selection.LeaveOneGroupOut()
    print(f'Using LOSO (Leave-One-Subject-Out): {subject_count} folds')
elif cv_method == 'kfold':
    if subject_count < 10:
        cv_outer = sklearn.model_selection.LeaveOneGroupOut()
        print(f'Using LOSO (subject count < 10): {subject_count} folds')
    else:
        cv_outer = sklearn.model_selection.GroupKFold(n_splits=10)
        print(f'Using 10-fold CV: 10 folds')
else:
    cv_outer = sklearn.model_selection.LeaveOneGroupOut()
    print(f'Invalid cv_method "{cv_method}", using LOSO: {subject_count} folds')

cv_outer_group = subjects

# Train/validation split
cv_inner_group = [f"{d.item()}_{l}" for d, l in zip(domain, y)]
cv_inner_group = sklearn.preprocessing.LabelEncoder().fit_transform(cv_inner_group)

# Model kwargs
mdl_kwargs = deepcopy(cfg['mdl_kwargs'])
mdl_kwargs['num_classes'] = n_classes
mdl_kwargs['eeg_num_electrodes'] = X_eeg.shape[1]
mdl_kwargs['eeg_chunk_size'] = X_eeg.shape[2]
mdl_kwargs['audio_freq_bins'] = X_audio.shape[1]
mdl_kwargs['audio_time_frames'] = X_audio.shape[2]
mdl_kwargs['vision_num_au'] = X_vision.shape[2]
mdl_kwargs['vision_time_length'] = X_vision.shape[1]
mdl_kwargs['domains'] = domain.unique()

all_splits = list(cv_outer.split(X_eeg, y, cv_outer_group))
total_folds = len(all_splits)
fold_timer = Timer()

# Training Mode
print("\n" + "=" * 80)
print("Starting Training")
print("=" * 80)
print(f"Verify Manifold: {args.verify_manifold}")

for ix_fold, (fit, test) in enumerate(all_splits):
        # Calculate progress
        progress = (ix_fold + 1) / total_folds if total_folds > 0 else 1.0
        elapsed = fold_timer.measure()
        eta = fold_timer.measure(progress) if progress > 0 else '0s'
        print(f'\nFOLD:{ix_fold}/{total_folds-1} ETA:{elapsed}/{eta}')

        # Split fitting data into train and validation
        cv_inner = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=cfg['validation_size'], random_state=random_seed)
        train, val = next(cv_inner.split(X_eeg[fit], y[fit], cv_inner_group[fit]))

        # Adjust domains per batch
        du = domain[fit][train].unique()
        if cfg['domains_per_batch'] > len(du):
            domains_per_batch = len(du)
        else:
            domains_per_batch = cfg['domains_per_batch']

        print("  Applying channel-wise normalization...")
        X_eeg_train_norm, X_eeg_val_norm, X_eeg_test_norm, _ = fn.normalize_physio_signals(
            X_eeg[fit][train], X_eeg[fit][val], X_eeg[test], signal_name='EEG'
        )
        ds_train = MultimodalDomainDataset(
            X_eeg_train_norm, X_audio[fit][train], X_vision[fit][train],
            y[fit][train], domain[fit][train]
        )
        ds_val = MultimodalDomainDataset(
            X_eeg_val_norm, X_audio[fit][val], X_vision[fit][val],
            y[fit][val], domain[fit][val]
        )

        loader_train = StratifiedMultimodalDataLoader(
            ds_train, cfg['batch_size_train'],
            domains_per_batch=domains_per_batch,
            shuffle=True, drop_last=False
        )
        loader_val = torch.utils.data.DataLoader(ds_val, batch_size=len(ds_val))
        model = HMultimodalNet(
            device=device, dtype=cfg['dtype'], **mdl_kwargs
        ).to(device=device, dtype=cfg['dtype'])
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"Initial curvatures: {model.get_curvatures()}")

        # Callbacks
        es = EarlyStopping(metric='val_loss', higher_is_better=False, patience=20, verbose=False)
        bn_sched = MomentumBatchNormScheduler(
            epochs=cfg['epochs'] - 10,
            bs0=cfg['batch_size_train'],
            bs=cfg['batch_size_train'] / cfg['domains_per_batch'],
            tau0=0.85
        )

        callbacks = [es, bn_sched]
        if cfg['mdl_kwargs'].get('learnable_curvature', False):
            curvature_warmup = CurvatureWarmupScheduler(
                warmup_epochs=5,
                warmup_scale=0.01,
                base_lr_scale=cfg['mdl_kwargs'].get('curvature_lr_scale', 0.1),
                mode='epoch',
                verbose=False
            )
            callbacks.append(curvature_warmup)

        # Create trainer
        trainer = MultimodalTrainer(
            max_epochs=cfg['epochs'],
            min_epochs=70,
            callbacks=callbacks,
            loss=torch.nn.CrossEntropyLoss(),
            device=device,
            dtype=torch.float64,
            lr=cfg['lr'],
            verify_manifold=args.verify_manifold,
            monitor_curvature=True
        )

        # Fit model
        trainer.fit(model, train_dataloader=loader_train, val_dataloader=loader_val)
        print(f'ES best epoch={es.best_epoch}')

        if trainer.monitor_curvature:
            final_curvatures = model.get_curvatures()
            print(f'Fold {ix_fold} - Final curvatures: {final_curvatures}')
            all_curvatures.append(final_curvatures)
            
            curvature_prior_weights = model.get_curvature_prior_weights()
            if curvature_prior_weights:
                print(f'Fold {ix_fold} - Curvature prior weights (λ): {curvature_prior_weights}')
                all_curvature_priors.append(curvature_prior_weights)

        sfuda_offline_net = deepcopy(model)
        test_subjects_fold = np.unique(subjects[test])
        
        saved_models.append({
            'fold': ix_fold,
            'model_state_dict': deepcopy(model.state_dict()),
            'test_subjects': test_subjects_fold.tolist(),
            'test_indices': test.tolist() if isinstance(test, np.ndarray) else test,
        })

        train_res = trainer.test(model, dataloader=loader_train, return_contributions=True)
        train_contributions = {
            'train_eeg_contribution': train_res.get('eeg_contribution'),
            'train_audio_contribution': train_res.get('audio_contribution'),
            'train_vision_contribution': train_res.get('vision_contribution')
        }
        print(f"Fold {ix_fold} - Train contributions: EEG={train_contributions['train_eeg_contribution']:.4f}, Audio={train_contributions['train_audio_contribution']:.4f}, Vision={train_contributions['train_vision_contribution']:.4f}")
        test_domains = domain[test].unique()
        for test_domain in test_domains:
            fold = ix_fold
            print(f"Fold:{fold}, test domain: {test_domain}")
            
            mask = domain[test] == test_domain
            ds_test = MultimodalDomainDataset(
                X_eeg_test_norm[mask], X_audio[test][mask], X_vision[test][mask],
                y[test][mask], domain[test][mask]
            )
            loader_test = torch.utils.data.DataLoader(ds_test, batch_size=len(ds_test))
            
            sfuda_offline(
                ds_test.eeg_features, ds_test.audio_features, ds_test.vision_features,
                ds_test.domains, sfuda_offline_net
            )
            
            res = trainer.test(sfuda_offline_net, dataloader=loader_test, return_contributions=True)
            print(f"HMultimodalNet, Test results: {res}")

            result_record = dict(
                dataset='multimodal',
                fold=fold,
                domain=test_domain.item(),
                cv_method=cv_method,
                score=res['score'],
                f1_macro=res['f1_macro'],
                loss=res['loss'],
                **train_contributions
            )

            if trainer.monitor_curvature:
                curvatures = sfuda_offline_net.get_curvatures()
                for k, v in curvatures.items():
                    result_record[f'curvature_{k}'] = v
                
                curvature_prior_weights = sfuda_offline_net.get_curvature_prior_weights()
                if curvature_prior_weights:
                    result_record['curvature_prior_lambda'] = curvature_prior_weights.get('lambda')

            records.append(result_record)


# Save Results

resdf = pd.DataFrame(records)
mean_score = float(resdf['score'].mean()) if 'score' in resdf.columns else None
std_score = float(resdf['score'].std()) if 'score' in resdf.columns else None
mean_f1_macro = float(resdf['f1_macro'].mean()) if 'f1_macro' in resdf.columns else None
std_f1_macro = float(resdf['f1_macro'].std()) if 'f1_macro' in resdf.columns else None
mean_loss = float(resdf['loss'].mean()) if 'loss' in resdf.columns else None
std_loss = float(resdf['loss'].std()) if 'loss' in resdf.columns else None

if all_curvatures:
    avg_curvatures = {
        k: np.mean([c[k] for c in all_curvatures])
        for k in all_curvatures[0].keys()
    }
else:
    avg_curvatures = {}

if all_curvature_priors:
    lambda_values = [c.get('lambda') for c in all_curvature_priors if c.get('lambda') is not None]
    if lambda_values:
        avg_curvature_prior_lambda = np.mean(lambda_values)
        std_curvature_prior_lambda = np.std(lambda_values)
    else:
        avg_curvature_prior_lambda = None
        std_curvature_prior_lambda = None
else:
    avg_curvature_prior_lambda = None
    std_curvature_prior_lambda = None

avg_contributions = {}
train_contribution_keys = ['train_eeg_contribution', 'train_audio_contribution', 'train_vision_contribution']
for key in train_contribution_keys:
    values = [r.get(key) for r in records if r.get(key) is not None]
    if values:
        avg_contributions[key] = np.mean(values)
        avg_contributions[f'{key}_std'] = np.std(values)

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename_base = f'results_multimodal_{cv_method}_{timestamp}'
txt_path = os.path.join(results_dir, f'{filename_base}.txt')

# Save models to subfolder
if saved_models:
    save_dir = os.path.join(results_dir, f'save_multimodal_{cv_method}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    config_save = {
        'cfg': cfg,
        'mdl_kwargs': mdl_kwargs,
        'args': vars(args),
        'timestamp': timestamp,
        'cv_method': cv_method,
        'n_classes': n_classes,
    }
    torch.save(config_save, os.path.join(save_dir, 'config.pth'))
    
    # Save each fold's model
    for model_info in saved_models:
        fold = model_info['fold']
        test_subjects = model_info['test_subjects']
        # Create filename with test subject info
        subjects_str = '_'.join([f'{s:02d}' if isinstance(s, int) else str(s) for s in test_subjects])
        model_filename = f'fold_{fold}_test_subjects_{subjects_str}.pth'
        
        torch.save({
            'fold': fold,
            'test_subjects': test_subjects,
            'test_indices': model_info['test_indices'],
            'model_state_dict': model_info['model_state_dict'],
        }, os.path.join(save_dir, model_filename))
    
    print(f"\nModels saved to: {save_dir}")

with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("MULTIMODAL HYPERBOLIC EMOTION RECOGNITION RESULTS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Cross-Validation Method: {cv_method}\n")
    f.write(f"Number of Classes: {n_classes}\n")
    f.write(f"Total Folds: {len(records)}\n")
    f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("Data Paths:\n")
    f.write("-" * 80 + "\n")
    f.write(f"EEG: {args.eeg_path}\n")
    f.write(f"Audio: {args.audio_path}\n")
    f.write(f"Vision: {args.vision_path}\n\n")
    
    f.write("Training Configuration:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Epochs: {cfg['epochs']}, LR: {cfg['lr']}, Weight Decay: {cfg['weight_decay']}\n")
    f.write(f"Batch Size: {cfg['batch_size_train']}, Domains per Batch: {cfg['domains_per_batch']}\n")
    f.write(f"GPU ID: {args.gpu_id}, Validation Size: {cfg['validation_size']}\n\n")
    
    f.write("Hyperbolic Configuration:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Learnable Curvature: {cfg['mdl_kwargs'].get('learnable_curvature', False)}\n")
    f.write(f"Curvature Init: {cfg['mdl_kwargs'].get('curvature_init', 1.0)}, LR Scale: {cfg['mdl_kwargs'].get('curvature_lr_scale', 1.0)}\n")
    f.write(f"Curvature Mode: mean, Curvature Aware: True\n")
    f.write(f"Curvature Prior Init (λ₀): {cfg['mdl_kwargs'].get('curvature_prior_init', 0.01)}\n")
    f.write(f"Verify Manifold: {args.verify_manifold}\n\n")

    f.write("=" * 80 + "\n")
    f.write("DETAILED RESULTS (Per Fold)\n")
    f.write("=" * 80 + "\n\n")
    for idx, record in enumerate(records):
        f.write(f"Fold {record.get('fold', idx)} - Domain {record.get('domain', 'N/A')}:\n")
        f.write("-" * 80 + "\n")
        if 'score' in record:
            f.write(f"  Score (Balanced Accuracy): {record['score']:.6f}\n")
        if 'f1_macro' in record:
            f.write(f"  F1 Macro: {record['f1_macro']:.6f}\n")
        if 'loss' in record:
            f.write(f"  Loss: {record['loss']:.6f}\n")
        for k, v in record.items():
            if k.startswith('curvature_') and not k.startswith('curvature_prior'):
                f.write(f"  {k}: {v:.6f}\n")
        if 'curvature_prior_lambda' in record and record['curvature_prior_lambda'] is not None:
            f.write(f"  Curvature Prior (λ): {record['curvature_prior_lambda']:.6f}\n")
        if 'train_eeg_contribution' in record and record['train_eeg_contribution'] is not None:
            f.write(f"  Train Modality Contributions:\n")
            f.write(f"    EEG: {record['train_eeg_contribution']:.4f} ({record['train_eeg_contribution']*100:.1f}%)\n")
            f.write(f"    Audio: {record['train_audio_contribution']:.4f} ({record['train_audio_contribution']*100:.1f}%)\n")
            f.write(f"    Vision: {record['train_vision_contribution']:.4f} ({record['train_vision_contribution']*100:.1f}%)\n")
        f.write("\n")

    f.write("=" * 80 + "\n")
    f.write("SUMMARY STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Score Statistics (Balanced Accuracy):\n")
    f.write("-" * 80 + "\n")
    if mean_score is not None:
        f.write(f"  Mean Score: {mean_score:.6f}\n")
    if std_score is not None:
        f.write(f"  Std Score: {std_score:.6f}\n")
    f.write("\n")

    f.write("F1 Macro Statistics:\n")
    f.write("-" * 80 + "\n")
    if mean_f1_macro is not None:
        f.write(f"  Mean F1 Macro: {mean_f1_macro:.6f}\n")
    if std_f1_macro is not None:
        f.write(f"  Std F1 Macro: {std_f1_macro:.6f}\n")
    f.write("\n")

    f.write("Loss Statistics:\n")
    f.write("-" * 80 + "\n")
    if mean_loss is not None:
        f.write(f"  Mean Loss: {mean_loss:.6f}\n")
    if std_loss is not None:
        f.write(f"  Std Loss: {std_loss:.6f}\n")
    f.write("\n")

    f.write("Curvature Statistics:\n")
    f.write("-" * 80 + "\n")
    f.write(f"  Curvature Init: {cfg['mdl_kwargs'].get('curvature_init', 1.0):.6f}\n")
    for k, v in avg_curvatures.items():
        f.write(f"  Mean {k} curvature: {v:.6f}\n")
    f.write("\n")

    f.write("Curvature Prior (λ) Statistics:\n")
    f.write("-" * 80 + "\n")
    if avg_curvature_prior_lambda is not None:
        f.write(f"  Initial λ: {cfg['mdl_kwargs'].get('curvature_prior_init', 0.01):.6f}\n")
        f.write(f"  Learned λ: {avg_curvature_prior_lambda:.6f} ± {std_curvature_prior_lambda:.6f}\n")
    else:
        f.write("  Curvature-aware attention disabled or no data available\n")
    f.write("\n")

    f.write("Modality Fusion Contribution Statistics:\n")
    f.write("-" * 80 + "\n")
    if avg_contributions:
        if 'train_eeg_contribution' in avg_contributions:
            f.write("  Train Set Contributions:\n")
            f.write(f"    EEG: {avg_contributions['train_eeg_contribution']:.4f} ± {avg_contributions.get('train_eeg_contribution_std', 0):.4f} ({avg_contributions['train_eeg_contribution']*100:.1f}%)\n")
            f.write(f"    Audio: {avg_contributions['train_audio_contribution']:.4f} ± {avg_contributions.get('train_audio_contribution_std', 0):.4f} ({avg_contributions['train_audio_contribution']*100:.1f}%)\n")
            f.write(f"    Vision: {avg_contributions['train_vision_contribution']:.4f} ± {avg_contributions.get('train_vision_contribution_std', 0):.4f} ({avg_contributions['train_vision_contribution']*100:.1f}%)\n")
    else:
        f.write("  No contribution data available\n")
    f.write("\n")

print(f"\nResults saved to: {txt_path}")

# Print summary
print("\n" + "=" * 80)
print("Results Summary Statistics:")
print("=" * 80)
print(f"Mean Score (Balanced Accuracy): {mean_score:.6f} ± {std_score:.6f}" if mean_score is not None and std_score is not None else "N/A")
print(f"Mean F1 Macro: {mean_f1_macro:.6f} ± {std_f1_macro:.6f}" if mean_f1_macro is not None and std_f1_macro is not None else "N/A")
print(f"Mean Loss: {mean_loss:.6f} ± {std_loss:.6f}" if mean_loss is not None and std_loss is not None else "N/A")
print(f"Average Curvatures: {avg_curvatures}")
if avg_curvature_prior_lambda is not None:
    print(f"Curvature Prior (λ) - Init: {cfg['mdl_kwargs'].get('curvature_prior_init', 0.01):.4f}, Learned: {avg_curvature_prior_lambda:.4f} ± {std_curvature_prior_lambda:.4f}")
if avg_contributions:
    if 'train_eeg_contribution' in avg_contributions:
        print(f"Train Contributions: EEG={avg_contributions.get('train_eeg_contribution', 0)*100:.1f}%, Audio={avg_contributions.get('train_audio_contribution', 0)*100:.1f}%, Vision={avg_contributions.get('train_vision_contribution', 0)*100:.1f}%")
print("=" * 80)

