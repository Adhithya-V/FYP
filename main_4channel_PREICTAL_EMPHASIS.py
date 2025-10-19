"""
4-CHANNEL MULTI-MODAL WITH STRONG PRE-ICTAL EMPHASIS
EEG + ECG + EMG with CLASS WEIGHTING for pre-ictal focus
Target: Maximize pre-ictal detection for alert system
"""

import os
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n" + "="*80)
print("4-CHANNEL MULTI-MODAL WITH PRE-ICTAL EMPHASIS")
print("EEG (2ch) + ECG (1ch) + EMG (1ch) = 4 channels")
print("CLASS WEIGHTING: Pre-ictal 3x, Ictal 1.5x, Inter-ictal 0.5x")
print("="*80)

from net.DL_config import Config
from net.main_func import train
import shutil

def cleanup_old_files(save_dir, keep_recent=1):
    """Aggressive cleanup"""
    try:
        models_dir = os.path.join(save_dir, 'models')
        if os.path.exists(models_dir):
            model_folders = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
            if len(model_folders) > keep_recent:
                model_folders.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                for folder in model_folders[keep_recent:]:
                    folder_path = os.path.join(models_dir, folder)
                    shutil.rmtree(folder_path)
                    print(f"[CLEANUP] Removed: {folder}")
    except Exception as e:
        print(f"[WARNING] Cleanup failed: {e}")

def main():
    # 4-CHANNEL WITH PRE-ICTAL CLASS WEIGHTING
    config = Config(
        data_path="C:/Adhi/SeizeIT2/ds005873-download",
        fs=256,
        CH=4,  # 2 EEG + 1 ECG + 1 EMG
        model='ChronoNet',
        dataset='SZ2',
        
        # BALANCED SPEED SETTINGS
        nb_epochs=5,           # More epochs for better learning
        batch_size=16,         # Moderate batch size
        frame=4,               # 4-second window
        stride=15,             # Moderate stride (balance speed/samples)
        factor=1,
        
        # PRE-ICTAL FOCUSED
        num_classes=3,
        pre_ictal_window=60,   # 60-second pre-ictal window
        inter_ictal_subsample_factor=0.03,  # 3% inter-ictal retention
        
        # PRE-ICTAL EMPHASIS via CLASS WEIGHTS
        class_weights={
            0: 3.0,    # Pre-ictal: 3x emphasis (MAXIMUM PRIORITY)
            1: 1.5,    # Ictal: 1.5x emphasis (HIGH)
            2: 0.5     # Inter-ictal: 0.5x emphasis (LOW)
        },
        
        # MULTI-MODAL (ECG + EMG for heart/muscle signals)
        modalities=['eeg', 'ecg', 'emg'],
        
        lr=0.001,
        dropoutRate=0.3,
        save_dir='net/save_dir'
    )
    
    cleanup_old_files(config.save_dir)
    
    print(f"\n[CONFIG] 4-Channel Multi-Modal with Pre-ictal Emphasis:")
    print(f"  Modalities: {config.modalities}")
    print(f"  Channels: {config.CH} (EEG + ECG + EMG)")
    print(f"  Epochs: {config.nb_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Frame: {config.frame}s, Stride: {config.stride}s")
    print(f"  Pre-ictal window: {config.pre_ictal_window}s")
    print(f"  Inter-ictal subsample: {config.inter_ictal_subsample_factor}")
    
    print(f"\n[PRE-ICTAL EMPHASIS] Class Weights:")
    for cls, weight in config.class_weights.items():
        cls_name = ['Pre-ictal', 'Ictal', 'Inter-ictal'][cls]
        emphasis = ['MAXIMUM', 'HIGH', 'LOW'][cls]
        print(f"  {cls_name}: {weight}x ({emphasis} priority)")
    
    print(f"\n[OBJECTIVE] Maximize pre-ictal detection for seizure alert system")
    print(f"  Current (EEG-only, no weighting): 51.93%")
    print(f"  Expected (4-ch + weighting): 62-68% (+10-16%)")
    
    print(f"\n[SPEED] Estimated time: 2-3 hours")
    print(f"  Segment generation: ~45 minutes (4,000-5,000 segments)")
    print(f"  Training: ~1.5 hours (5 epochs)")
    
    print(f"\n[START] Training with pre-ictal emphasis...")
    print("="*80)
    
    import time
    start_time = time.time()
    
    # NO generator caching
    history = train(config, load_generators=False, save_generators=False)
    
    end_time = time.time()
    duration_hours = (end_time - start_time) / 3600
    
    if history is None:
        print("[ERROR] Training failed!")
        return
    
    print("\n" + "="*80)
    print(f"PRE-ICTAL FOCUSED MULTI-MODAL TRAINING COMPLETED!")
    print(f"Actual time: {duration_hours:.2f} hours")
    print("="*80)
    
    model_name = config.get_name()
    weights_path = os.path.join(config.save_dir, 'models', model_name, 'Weights', f"{model_name}.weights.h5")
    
    if os.path.exists(weights_path):
        print(f"\n[SUCCESS] Pre-ictal focused model saved: {weights_path}")
        print(f"[SUCCESS] Ready for evaluation!")
        print(f"\n[EMPHASIS] This model was trained with:")
        print(f"  - 3x weight on Pre-ictal class (maximum learning)")
        print(f"  - 1.5x weight on Ictal class (high learning)")
        print(f"  - 0.5x weight on Inter-ictal class (minimal learning)")
        print(f"\n[NEXT] Compare with non-weighted version:")
        print(f"  python evaluate_3class.py")
        print("="*80)
    else:
        print(f"[ERROR] Model not found: {weights_path}")

if __name__ == "__main__":
    main()

