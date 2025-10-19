"""
Single Patient Demo Script for Review Presentation
==================================================

This script:
1. Loads a trained model
2. Tests on a SINGLE patient's data
3. Generates accuracy metrics and plots
4. Perfect for software demo during review

Usage:
    python demo_single_patient.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Add project paths
sys.path.append(os.getcwd())
from net.DL_config import Config
from net.ChronoNet import net as ChronoNet
from net.key_generator import generate_data_keys_preictal
from net.generator_ds import SegmentedGenerator
from classes.data import Data
from classes.annotation import Annotation

# ============================================================================
# CONFIGURATION
# ============================================================================

# Choose a test patient (single subject for demo)
# Recommended: sub-117 (79% acc), sub-109 (74% acc)
DEMO_PATIENT = "sub-117"  # Change this to any test patient

# Model path (multi-modal 4-channel model)
MODEL_PATH = "net/save_dir/models/ChronoNet_subsample_factor1/Weights/ChronoNet_subsample_factor1.weights.h5"

# Configuration matching the trained model
config = Config(
    data_path="C:/Adhi/SeizeIT2/ds005873-download",
    fs=256,
    CH=4,  # Multi-modal: EEG(2) + ECG(1) + EMG(1)
    model='ChronoNet',
    batch_size=16,
    frame=4,
    stride=15,
    factor=1,
    num_classes=3,
    pre_ictal_window=60,
    inter_ictal_subsample_factor=0.03,
    modalities=['eeg', 'ecg', 'emg']
)

# ============================================================================
# LOAD MODEL
# ============================================================================

print("=" * 70)
print("SEIZURE PREDICTION DEMO - SINGLE PATIENT")
print("=" * 70)
print(f"\nDemo Patient: {DEMO_PATIENT}")
print(f"Model: {os.path.basename(MODEL_PATH)}")
print(f"Modalities: EEG + ECG + EMG (4 channels)")
print(f"Classes: Pre-ictal (0) / Ictal (1) / Inter-ictal (2)")
print("\n" + "=" * 70 + "\n")

print("Loading model...")
model = ChronoNet(config)
model.load_weights(MODEL_PATH)
print("Model loaded successfully!\n")

# ============================================================================
# GENERATE DATA FOR SINGLE PATIENT
# ============================================================================

print(f"Generating data segments for patient {DEMO_PATIENT}...")

# Load test patients using the EXACT same method as evaluate_3class.py
test_pats_list = pd.read_csv(os.path.join('net', 'datasets', 'SZ2_test.tsv'), 
                              sep='\t', header=None, skiprows=[0, 1, 2])
test_pats_list = test_pats_list[0].to_list()

# Filter for only the demo patient
if DEMO_PATIENT not in test_pats_list:
    print(f"ERROR: Patient {DEMO_PATIENT} not found in test set!")
    print("Available test patients:")
    for p in sorted(test_pats_list):
        print(f"  - {p}")
    sys.exit(1)

# Get recordings for this patient (EXACT same logic as evaluate_3class.py)
test_recs = [[DEMO_PATIENT, r.split('_')[-2]] 
             for r in os.listdir(os.path.join(config.data_path, DEMO_PATIENT, 'ses-01', 'eeg')) 
             if 'edf' in r]

print(f"Found {len(test_recs)} recording(s) for {DEMO_PATIENT}")

# Generate segment keys using the exact same method
try:
    test_segments = generate_data_keys_preictal(config, test_recs, verbose=True)
except ZeroDivisionError:
    print(f"ERROR: Division by zero in segment generation for patient {DEMO_PATIENT}!")
    print("This happens when patient has seizures but no valid pre-ictal segments.")
    print("Try a different patient or adjust the pre_ictal_window in config.")
    sys.exit(1)

print(f"Generated {len(test_segments)} test segments\n")

if len(test_segments) == 0:
    print(f"ERROR: No valid segments found for patient {DEMO_PATIENT}!")
    print("This patient may not have seizure annotations or the data format is incompatible.")
    print("Try a different patient or check the data configuration.")
    sys.exit(1)

# ============================================================================
# CREATE DATA GENERATOR
# ============================================================================

print("Creating data generator...")
test_generator = SegmentedGenerator(
    config=config,
    recs=test_recs,
    segments=test_segments,
    shuffle=False,
    batch_size=config.batch_size,
    num_workers=1,
    verbose=False
)
print(f"Generator ready with {len(test_generator)} batches\n")

if len(test_generator) == 0:
    print(f"ERROR: Generator created but has 0 batches!")
    print("This means no valid segments could be processed.")
    sys.exit(1)

# ============================================================================
# RUN PREDICTION
# ============================================================================

print("Running model prediction...")
print("(This may take 1-2 minutes...)\n")

# Collect ground truth
y_true_list = []
for i in range(len(test_generator)):
    _, y_batch = test_generator[i]
    y_true_list.append(y_batch)
y_true_onehot = np.vstack(y_true_list)
y_true = np.argmax(y_true_onehot, axis=1)  # Convert to class indices

# Get predictions
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted class indices

print("\nPrediction complete!\n")

# ============================================================================
# CALCULATE METRICS
# ============================================================================

print("=" * 70)
print("RESULTS FOR PATIENT", DEMO_PATIENT)
print("=" * 70)

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {accuracy*100:.2f}%\n")

# Classification report
class_names = ['Pre-ictal', 'Ictal', 'Inter-ictal']
print("Classification Report:")
print("-" * 70)
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print("-" * 70)
print("              Predicted:")
print("              Pre-ictal  Ictal  Inter-ictal")
print(f"Actual Pre-ictal:    {cm[0][0]:4d}     {cm[0][1]:4d}      {cm[0][2]:4d}")
print(f"       Ictal:        {cm[1][0]:4d}     {cm[1][1]:4d}      {cm[1][2]:4d}")
print(f"       Inter-ictal:  {cm[2][0]:4d}     {cm[2][1]:4d}      {cm[2][2]:4d}")

# Per-class sensitivity
print("\nPer-Class Sensitivity (Recall):")
print("-" * 70)
for i, name in enumerate(class_names):
    if cm[i].sum() > 0:
        sens = cm[i][i] / cm[i].sum() * 100
        print(f"{name:15s}: {sens:6.2f}% ({cm[i][i]}/{cm[i].sum()})")
    else:
        print(f"{name:15s}: N/A (no samples)")

print("\n" + "=" * 70 + "\n")

# ============================================================================
# GENERATE PLOTS
# ============================================================================

print("Generating plots...\n")

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create figure with 2 subplots (1 row, 2 columns)
fig = plt.figure(figsize=(16, 6))

# --- Plot 1: Confusion Matrix ---
ax1 = plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - Patient {DEMO_PATIENT}', fontsize=14, fontweight='bold')
plt.ylabel('Actual Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)

# --- Plot 2: Ground Truth vs Prediction (First 200 samples) ---
ax2 = plt.subplot(1, 2, 2)
display_samples = min(200, len(y_true))
x_axis = np.arange(display_samples)
plt.plot(x_axis, y_true[:display_samples], 'g-', linewidth=2, label='Ground Truth', alpha=0.7)
plt.plot(x_axis, y_pred[:display_samples], 'r--', linewidth=1.5, label='Prediction', alpha=0.8)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Class (0=Pre, 1=Ictal, 2=Inter)', fontsize=12)
plt.title(f'Ground Truth vs Prediction (First {display_samples} samples)', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(-0.5, 2.5)
plt.yticks([0, 1, 2], ['Pre-ictal', 'Ictal', 'Inter-ictal'])

# Overall title
fig.suptitle(f'Seizure Prediction Demo - Patient {DEMO_PATIENT}\nMulti-Modal (EEG+ECG+EMG) | Accuracy: {accuracy*100:.2f}%', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
output_file = f"demo_patient_{DEMO_PATIENT}_results.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved: {output_file}")

# Show plot
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETE!")
print("=" * 70)
print(f"\nPatient: {DEMO_PATIENT}")
print(f"Total Samples: {len(y_true)}")
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"\nResults plot: {output_file}")
print("\n" + "=" * 70)
print("\nThis demo is ready for your review presentation!")
print("Change DEMO_PATIENT variable to test other patients.")
print("=" * 70 + "\n")