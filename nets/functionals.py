from lib.lorentz.manifold import CustomLorentz
import torch
from torch.functional import Tensor
import numpy as np
from scipy.linalg import sqrtm, inv 
import os
import pickle

def euler_align(raw_array):
    """Euler alignment on (trials, channels, samples) array"""
    cov_matrices = [np.cov(trial, rowvar=True) for trial in raw_array]
    #cov_matrices = np.cov(raw_array,axis=0, rowvar=True)
    mean_cov_matrix = np.mean(cov_matrices, axis=0)
    
    # Compute transformation matrix
    trans_matrix = inv(sqrtm(mean_cov_matrix))
    
    # Apply transformation to all trials using broadcasting
    return trans_matrix @ raw_array


def compute_channel_stats(data):
    """Channel-wise mean/std from (N, C, T) data -> (C,), (C,)"""
    if isinstance(data, torch.Tensor):
        data = data.numpy() if not data.is_cuda else data.cpu().numpy()
    
    # Assume shape is (N, C, T) - compute mean/std over samples and time
    # Result: (C,) for each channel
    mean = data.mean(axis=(0, 2))  # Mean over samples and time
    std = data.std(axis=(0, 2))    # Std over samples and time
    std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
    
    return mean, std


def normalize_channels(data, mean, std):
    """Channel-wise normalization: (data - mean) / std"""
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean).to(dtype=data.dtype)
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std).to(dtype=data.dtype)
    
    # Reshape for broadcasting: (C,) -> (1, C, 1)
    mean = mean.view(1, -1, 1)
    std = std.view(1, -1, 1)
    
    return (data - mean) / std


def normalize_physio_signals(X_train, X_val, X_test, signal_name='signal'):
    """Normalize using train+val stats -> (train_norm, val_norm, test_norm, stats)"""
    # Combine train and val for computing statistics
    X_fit = torch.cat([X_train, X_val], dim=0)
    
    # Compute statistics
    mean, std = compute_channel_stats(X_fit)
    
    # Normalize all splits
    X_train_norm = normalize_channels(X_train, mean, std)
    X_val_norm = normalize_channels(X_val, mean, std)
    X_test_norm = normalize_channels(X_test, mean, std)
    
    print(f"  {signal_name} normalized: mean range [{mean.min():.3f}, {mean.max():.3f}], "
          f"std range [{std.min():.3f}, {std.max():.3f}]")
    
    return X_train_norm, X_val_norm, X_test_norm, {'mean': mean, 'std': std}


def load_multimodal_dataset(eeg_path, audio_path, vision_path, min_vision_trial_length=603):
    """Load EAV multimodal dataset with trial_id alignment"""
    import pandas as pd
    import re
    
    AU_REGRESSION_COLS = [
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r',
        'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
        'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r'
    ]
    
    AU_CLASSIFICATION_COLS = [
        'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c',
        'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c',
        'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c'
    ]
    
    AU_FEATURE_COLS = AU_REGRESSION_COLS + AU_CLASSIFICATION_COLS
    
    EMOTION_MAP = {
        'neutral': 0,
        'anger': 1,
        'calmness': 2,
        'sadness': 3,
        'happiness': 4
    }
    
    all_eeg = []
    all_audio = []
    all_vision = []
    all_labels = []
    all_subjects = []
    all_sessions = []
    all_trial_ids = []
    
    print("Loading multimodal dataset...")
    print(f"EEG path: {eeg_path}")
    print(f"Audio path: {audio_path}")
    print(f"Vision path: {vision_path}")
    
    for subject_id in range(1, 43):
        eeg_file = os.path.join(eeg_path, f'subject_{subject_id:02d}_eeg.dat')
        audio_file = os.path.join(audio_path, f'subject_{subject_id:02d}_au.dat')
        vision_folder = os.path.join(vision_path, f'subject{subject_id}', 'Video_AU')
        
        if not os.path.exists(eeg_file):
            print(f"Subject {subject_id}: EEG file not found, skipping")
            continue
        if not os.path.exists(audio_file):
            print(f"Subject {subject_id}: Audio file not found, skipping")
            continue
        if not os.path.exists(vision_folder):
            print(f"Subject {subject_id}: Vision folder not found, skipping")
            continue
        
        try:
            eeg_data_dict = np.load(eeg_file, allow_pickle=True).item()
            eeg_data = eeg_data_dict['data']
            eeg_labels = eeg_data_dict['label']
            eeg_trial_ids = eeg_data_dict.get('trial_id', np.arange(2, 202, 2))
            
            if len(eeg_labels.shape) > 1:
                eeg_labels = np.argmax(eeg_labels, axis=1)
            
            audio_data_dict = np.load(audio_file, allow_pickle=True).item()
            audio_data = audio_data_dict['data']
            audio_labels = audio_data_dict['label']
            audio_trial_ids = audio_data_dict.get('trial_id', np.arange(2, 202, 2))
            
            if len(audio_labels.shape) > 1:
                audio_labels = np.argmax(audio_labels, axis=1)
            
            vision_csv_files = [f for f in os.listdir(vision_folder) if f.endswith('_AU.csv')]
            vision_by_trial_id = {}
            
            for csv_file in vision_csv_files:
                match = re.match(r'^(\d+)_Trial_\d+_(Listening|Speaking)_(Neutral|Anger|Calmness|Sadness|Happiness)_AU\.csv', csv_file, re.IGNORECASE)
                if match:
                    vision_trial_id = int(match.group(1))
                    if vision_trial_id % 2 != 0:
                        continue
                    
                    file_path = os.path.join(vision_folder, csv_file)
                    try:
                        df = pd.read_csv(file_path)
                        
                        missing_cols = [col for col in AU_FEATURE_COLS if col not in df.columns]
                        if missing_cols:
                            continue
                        
                        au_data = df[AU_FEATURE_COLS].values
                        
                        if np.isnan(au_data).any():
                            au_data = np.nan_to_num(au_data, nan=0.0)
                        
                        n_timepoints = au_data.shape[0]
                        if n_timepoints > min_vision_trial_length:
                            au_data = au_data[n_timepoints - min_vision_trial_length:, :]
                        elif n_timepoints < min_vision_trial_length:
                            pad_length = min_vision_trial_length - n_timepoints
                            au_data = np.pad(au_data, ((pad_length, 0), (0, 0)), mode='constant')
                        
                        vision_by_trial_id[vision_trial_id] = au_data
                    except Exception as e:
                        print(f"Error loading vision file {csv_file}: {e}")
                        continue
            
            eeg_by_trial_id = {tid: (eeg_data[i], eeg_labels[i]) for i, tid in enumerate(eeg_trial_ids)}
            audio_by_trial_id = {tid: audio_data[i] for i, tid in enumerate(audio_trial_ids)}
            
            common_trial_ids = set(eeg_by_trial_id.keys()) & set(audio_by_trial_id.keys()) & set(vision_by_trial_id.keys())
            common_trial_ids = sorted(list(common_trial_ids))
            
            if len(common_trial_ids) == 0:
                print(f"Subject {subject_id}: No common trial_ids found")
                continue
            
            subject_eeg = []
            subject_audio = []
            subject_vision = []
            subject_labels = []
            subject_trial_ids = []
            
            print(f"\n--- Subject {subject_id:02d} Trial ID Verification ---")
            print(f"  EEG trial_ids ({len(eeg_trial_ids)}): {sorted(eeg_trial_ids)[:5]}...{sorted(eeg_trial_ids)[-5:]}")
            print(f"  Audio trial_ids ({len(audio_trial_ids)}): {sorted(audio_trial_ids)[:5]}...{sorted(audio_trial_ids)[-5:]}")
            print(f"  Vision trial_ids ({len(vision_by_trial_id)}): {sorted(vision_by_trial_id.keys())[:5]}...{sorted(vision_by_trial_id.keys())[-5:]}")
            print(f"  Common trial_ids ({len(common_trial_ids)}): {common_trial_ids[:5]}...{common_trial_ids[-5:]}")
            
            eeg_set = set(eeg_trial_ids)
            audio_set = set(audio_trial_ids)
            if eeg_set != audio_set:
                missing_in_audio = eeg_set - audio_set
                missing_in_eeg = audio_set - eeg_set
                if missing_in_audio:
                    print(f"  WARNING: trial_ids in EEG but not Audio: {sorted(missing_in_audio)}")
                if missing_in_eeg:
                    print(f"  WARNING: trial_ids in Audio but not EEG: {sorted(missing_in_eeg)}")
            
            odd_trials = [tid for tid in common_trial_ids if tid % 2 != 0]
            if odd_trials:
                raise ValueError(f"Subject {subject_id}: Found odd trial_ids (should be even only): {odd_trials}")
            
            out_of_range = [tid for tid in common_trial_ids if tid < 2 or tid > 200]
            if out_of_range:
                print(f"  WARNING: trial_ids outside expected range [2,200]: {out_of_range}")
            
            for tid in common_trial_ids:
                eeg_sample, label = eeg_by_trial_id[tid]
                audio_sample = audio_by_trial_id[tid]
                vision_sample = vision_by_trial_id[tid]
                
                subject_eeg.append(eeg_sample)
                subject_audio.append(audio_sample)
                subject_vision.append(vision_sample)
                subject_labels.append(label)
                subject_trial_ids.append(tid)
            
            n_trials = len(common_trial_ids)
            
            assert len(subject_eeg) == len(subject_audio) == len(subject_vision) == len(subject_labels) == len(subject_trial_ids), \
                f"Subject {subject_id}: Mismatched array lengths!"
            
            all_eeg.append(np.stack(subject_eeg, axis=0))
            all_audio.append(np.stack(subject_audio, axis=0))
            all_vision.append(np.stack(subject_vision, axis=0))
            all_labels.append(np.array(subject_labels))
            all_subjects.extend([subject_id] * n_trials)
            all_sessions.extend([1] * n_trials)
            all_trial_ids.extend(subject_trial_ids)
            
            print(f"  ✓ Subject_{subject_id:02d} VERIFIED: {n_trials} trials aligned (EEG: {eeg_sample.shape}, Audio: {audio_sample.shape}, Vision: {vision_sample.shape})")
            
        except Exception as e:
            print(f"Error loading subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_eeg:
        raise ValueError("No data loaded. Check the paths and file structure.")
    
    eeg_data = np.concatenate(all_eeg, axis=0)
    audio_data = np.concatenate(all_audio, axis=0)
    vision_data = np.concatenate(all_vision, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    subjects = np.array(all_subjects)
    sessions = np.array(all_sessions)
    trial_ids = np.array(all_trial_ids)
    
    print("\n" + "=" * 60)
    print("FINAL MULTIMODAL ALIGNMENT VERIFICATION")
    print("=" * 60)
    
    assert eeg_data.shape[0] == audio_data.shape[0] == vision_data.shape[0] == len(labels) == len(subjects) == len(trial_ids), \
        f"Global dimension mismatch! EEG:{eeg_data.shape[0]}, Audio:{audio_data.shape[0]}, Vision:{vision_data.shape[0]}, Labels:{len(labels)}, Subjects:{len(subjects)}, Trial_ids:{len(trial_ids)}"
    
    odd_count = np.sum(trial_ids % 2 != 0)
    if odd_count > 0:
        raise ValueError(f"Found {odd_count} odd trial_ids! All should be even (Speaking trials).")
    
    unique_subjects = np.unique(subjects)
    print(f"\nPer-subject trial_id verification:")
    for subj in unique_subjects:
        subj_mask = subjects == subj
        subj_trial_ids = trial_ids[subj_mask]
        subj_labels = labels[subj_mask]
        
        is_sorted = np.all(subj_trial_ids[:-1] <= subj_trial_ids[1:])
        n_unique = len(np.unique(subj_trial_ids))
        
        print(f"  Subject {subj:02d}: {len(subj_trial_ids)} trials, IDs [{subj_trial_ids.min()}-{subj_trial_ids.max()}], "
              f"unique:{n_unique}, sorted:{is_sorted}, labels:{dict(zip(*np.unique(subj_labels, return_counts=True)))}")
    
    print(f"\n✓ ALL VERIFICATIONS PASSED")
    print("=" * 60)
    
    print(f"\nMultimodal Dataset Summary:")
    print(f"  Total trials: {len(labels)}")
    print(f"  EEG shape: {eeg_data.shape}")
    print(f"  Audio shape: {audio_data.shape}")
    print(f"  Vision shape: {vision_data.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Trial IDs shape: {trial_ids.shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"  Number of subjects: {len(np.unique(subjects))}")
    
    return eeg_data, audio_data, vision_data, labels, subjects, sessions, trial_ids