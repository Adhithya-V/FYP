"""
3-CLASS EVALUATION SCRIPT
Generates classification report, confusion matrix, and plots WITHOUT retraining
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n" + "="*80)
print("3-CLASS PREICTAL MODEL EVALUATION")
print("="*80)

from net.DL_config import Config
from net.key_generator import generate_data_keys_preictal
from net.generator_ds import SequentialGenerator
from net.routines import predict_net
from net.ChronoNet import net as chrononet

# Load the same config used for training
# UPDATED FOR MULTI-MODAL 4-CHANNEL MODEL
config = Config(
    data_path="C:/Adhi/SeizeIT2/ds005873-download",
    fs=256,
    CH=4,  # CHANGED: 4 channels (EEG + ECG + EMG)
    model='ChronoNet',
    dataset='SZ2',
    frame=4,  # CHANGED: Match training config
    stride=15,  # CHANGED: Match training config
    factor=1,
    num_classes=3,
    pre_ictal_window=60,  # CHANGED: Match training config
    inter_ictal_subsample_factor=0.03,  # CHANGED: Match training config
    modalities=['eeg', 'ecg', 'emg'],  # ADDED: Multi-modal
    save_dir='net/save_dir'
)

model_name = config.get_name()
weights_path = os.path.join(config.save_dir, 'models', model_name, 'Weights', f"{model_name}.weights.h5")

print(f"Model: {model_name}")
print(f"Weights path: {weights_path}")

if not os.path.exists(weights_path):
    print(f"\n[ERROR] Model weights not found at: {weights_path}")
    sys.exit(1)

print(f"[SUCCESS] Model weights found!")

# Check if training history exists
history_path = os.path.join(config.save_dir, 'models', model_name, 'History', f"{model_name}.csv")
if os.path.exists(history_path):
    print(f"[INFO] Loading training history from: {history_path}")
    history_df = pd.read_csv(history_path)
    
    # Plot Training vs Validation Loss and Accuracy
    print("\n[INFO] Generating training history plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    if 'loss' in history_df.columns and 'val_loss' in history_df.columns:
        ax1.plot(history_df['loss'], label='Training Loss', linewidth=2, marker='o')
        ax1.plot(history_df['val_loss'], label='Validation Loss', linewidth=2, marker='s')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('3-Class Training vs Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'accuracy' in history_df.columns and 'val_accuracy' in history_df.columns:
        ax2.plot(history_df['accuracy'], label='Training Accuracy', linewidth=2, marker='o')
        ax2.plot(history_df['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('3-Class Training vs Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(config.save_dir, 'training_history_3class.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"[SAVED] Training history plot: {plot_path}")
    plt.close()
else:
    print(f"[WARNING] Training history not found at: {history_path}")

# Load test data
print("\n[INFO] Loading test data...")
test_pats_list = pd.read_csv(os.path.join('net', 'datasets', 'SZ2_test.tsv'), 
                              sep='\t', header=None, skiprows=[0, 1, 2])
test_pats_list = test_pats_list[0].to_list()

test_recs_list = [[s, r.split('_')[-2]] for s in test_pats_list 
                  for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg')) 
                  if 'edf' in r]

print(f"[INFO] Test patients: {len(test_pats_list)}")
print(f"[INFO] Test recordings: {len(test_recs_list)}")

# Generate test segments
print("\n[INFO] Generating test segments...")
test_segments = generate_data_keys_preictal(config, test_recs_list, verbose=True)
print(f"[INFO] Total test segments: {len(test_segments)}")

# Create test generator
print("\n[INFO] Creating test generator...")
gen_test = SequentialGenerator(config, test_recs_list, test_segments, 
                               batch_size=32, shuffle=False, verbose=True)

# Load model and predict
print("\n[INFO] Loading model and generating predictions...")
model = chrononet(config)
y_pred_probs, y_true = predict_net(gen_test, weights_path, model)
y_pred = np.argmax(y_pred_probs, axis=1)

# Overall metrics
class_names = ['Pre-ictal', 'Ictal', 'Inter-ictal']
overall_acc = np.mean(y_pred == y_true)

print("\n" + "="*80)
print("OVERALL RESULTS")
print("="*80)
print(f"Overall Accuracy: {overall_acc:.4f} ({100*overall_acc:.2f}%)")
print(f"Total test samples: {len(y_true)}")

# Classification Report
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# Per-class detailed metrics
print("\n" + "="*80)
print("PER-CLASS DETAILED RESULTS")
print("="*80)
for i, class_name in enumerate(class_names):
    mask_true = (y_true == i)
    mask_pred = (y_pred == i)
    
    if np.sum(mask_true) > 0:
        tp = np.sum(mask_true & mask_pred)
        fp = np.sum(~mask_true & mask_pred)
        fn = np.sum(mask_true & ~mask_pred)
        tn = np.sum(~mask_true & ~mask_pred)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        print(f"\n{class_name}:")
        print(f"  Samples: {np.sum(mask_true)}")
        print(f"  Precision: {precision:.4f} ({100*precision:.2f}%)")
        print(f"  Recall (Sensitivity): {recall:.4f} ({100*recall:.2f}%)")
        print(f"  Specificity: {specificity:.4f} ({100*specificity:.2f}%)")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}")

# Confusion Matrix Heatmap
print("\n[INFO] Generating confusion matrix plot...")
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
            yticklabels=class_names, ax=ax, cbar_kws={'label': 'Count'},
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Predicted Class', fontsize=14, fontweight='bold')
ax.set_ylabel('Ground Truth Class', fontsize=14, fontweight='bold')
ax.set_title('3-Class Confusion Matrix (Test Set)', fontsize=16, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(config.save_dir, 'confusion_matrix_3class.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"[SAVED] Confusion matrix: {cm_path}")
plt.close()

# GT vs Prediction scatter plot
print("\n[INFO] Generating GT vs Prediction plot...")
fig, ax = plt.subplots(figsize=(18, 6))
sample_size = min(1000, len(y_true))
x_axis = np.arange(sample_size)

ax.scatter(x_axis, y_true[:sample_size], alpha=0.6, label='Ground Truth', s=30, c='blue', marker='o')
ax.scatter(x_axis, y_pred[:sample_size], alpha=0.6, label='Prediction', s=30, c='red', marker='x')
ax.set_xlabel('Sample Index (Temporal Sequence)', fontsize=12, fontweight='bold')
ax.set_ylabel('Class', fontsize=12, fontweight='bold')
ax.set_title(f'Ground Truth vs Prediction (First {sample_size} samples in temporal order)', fontsize=14, fontweight='bold')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Pre-ictal', 'Ictal', 'Inter-ictal'])
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
pred_path = os.path.join(config.save_dir, 'gt_vs_prediction_3class.png')
plt.savefig(pred_path, dpi=150, bbox_inches='tight')
print(f"[SAVED] GT vs Prediction plot: {pred_path}")
plt.close()

# Class distribution comparison
print("\n[INFO] Generating class distribution plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Ground Truth distribution
unique_true, counts_true = np.unique(y_true, return_counts=True)
ax1.bar([class_names[int(i)] for i in unique_true], counts_true, color=['blue', 'green', 'orange'])
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Ground Truth Class Distribution', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for i, (cls, count) in enumerate(zip(unique_true, counts_true)):
    ax1.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

# Predicted distribution
unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
ax2.bar([class_names[int(i)] for i in unique_pred], counts_pred, color=['blue', 'green', 'orange'])
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for i, (cls, count) in enumerate(zip(unique_pred, counts_pred)):
    ax2.text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
dist_path = os.path.join(config.save_dir, 'class_distribution_3class.png')
plt.savefig(dist_path, dpi=150, bbox_inches='tight')
print(f"[SAVED] Class distribution plot: {dist_path}")
plt.close()

print("\n" + "="*80)
print("ALL EVALUATION REPORTS AND PLOTS COMPLETED!")
print("="*80)
print("\nGenerated files:")
print(f"  1. {cm_path}")
print(f"  2. {pred_path}")
print(f"  3. {dist_path}")
if os.path.exists(os.path.join(config.save_dir, 'training_history_3class.png')):
    print(f"  4. {os.path.join(config.save_dir, 'training_history_3class.png')}")
print(f"\nModel ready for STM Nucleo: {weights_path}")
print("="*80)

