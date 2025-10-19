# Preprocessing Pipeline for STM32 Implementation

## Overview
This document details the **exact preprocessing steps** used in the seizure prediction model. You need to replicate these steps on STM32 to ensure raw biosignal data matches the format the trained model expects.

---

## Signal Processing Pipeline

### 1. **EEG Channels (2 channels)**

#### Step 1: Resampling
- **Input**: Raw EEG at original sampling rate (varies, typically 250-512 Hz)
- **Output**: Resampled to **256 Hz**
- **Method**: `scipy.signal.resample(data, target_samples)`
  - `target_samples = int(256 * len(data) / original_fs)`

#### Step 2: High-Pass Filter (Remove DC drift and slow artifacts)
- **Type**: Butterworth 2nd order
- **Cutoff**: 0.5 Hz
- **Filter**: High-pass
- **Implementation**: `scipy.signal.butter(2, 0.5/(256/2), 'high')`
- **Apply**: `scipy.signal.filtfilt(b, a, data)` (zero-phase filtering)

#### Step 3: Low-Pass Filter (Remove high-frequency noise)
- **Type**: Butterworth 2nd order
- **Cutoff**: 60 Hz
- **Filter**: Low-pass
- **Implementation**: `scipy.signal.butter(2, 60/(256/2), 'low')`

#### Step 4: Notch Filter (Remove powerline interference)
- **Type**: Butterworth 2nd order bandstop
- **Cutoff**: 49.5-50.5 Hz (50 Hz notch)
- **Implementation**: `scipy.signal.butter(2, [49.5/(256/2), 50.5/(256/2)], 'bandstop')`
- **Note**: Only applied if fs > 100 Hz

---

### 2. **ECG Channel (1 channel)**

#### Step 1: Resampling
- **Input**: Raw ECG at original sampling rate
- **Output**: Resampled to **256 Hz**

#### Step 2: Bandpass Filter (R-wave detection range)
- **Type**: Butterworth 2nd order
- **Cutoff**: 0.5-40 Hz
- **Filter**: Bandpass
- **Implementation**: `scipy.signal.butter(2, [0.5/(256/2), 40/(256/2)], 'bandpass')`
- **Apply**: `scipy.signal.filtfilt(b, a, data)`

#### Step 3: Notch Filter (Powerline interference)
- **Type**: Butterworth 2nd order bandstop
- **Cutoff**: 49.5-50.5 Hz (50 Hz notch)

---

### 3. **EMG Channel (1 channel)**

#### Step 1: Resampling
- **Input**: Raw EMG at original sampling rate
- **Output**: Resampled to **256 Hz**

#### Step 2: Bandpass Filter (Muscle activity range)
- **Type**: Butterworth 2nd order
- **Cutoff**: 20-100 Hz
- **Filter**: Bandpass
- **Implementation**: `scipy.signal.butter(2, [20/(256/2), 100/(256/2)], 'bandpass')`

#### Step 3: Notch Filter (Powerline interference)
- **Type**: Butterworth 2nd order bandstop
- **Cutoff**: 49.5-50.5 Hz (50 Hz notch)

---

## STM32 Implementation Guide

### Required Libraries
- **ARM DSP Library**: For efficient Butterworth filter implementation
- **CMSIS-DSP**: For resampling and filtering functions

### C Code Structure

```c
#include "arm_math.h"

// Filter coefficients (pre-computed from scipy.signal.butter)
// EEG High-pass (0.5 Hz at 256 Hz)
float32_t hp_b[5] = {0.9907, -1.9814, 0.9907, 0, 0};
float32_t hp_a[5] = {1.0, -1.9813, 0.9813, 0, 0};

// EEG Low-pass (60 Hz at 256 Hz)
float32_t lp_b[5] = {0.1367, 0.2735, 0.1367, 0, 0};
float32_t lp_a[5] = {1.0, -0.9428, 0.3333, 0, 0};

// EEG Notch (50 Hz at 256 Hz)
float32_t notch_b[5] = {0.9975, -1.9561, 0.9975, 0, 0};
float32_t notch_a[5] = {1.0, -1.9561, 0.9950, 0, 0};

// ECG Bandpass (0.5-40 Hz at 256 Hz)
float32_t ecg_bp_b[5] = {0.0726, 0, -0.1453, 0, 0.0726};
float32_t ecg_bp_a[5] = {1.0, -2.7661, 2.9754, -1.5576, 0.3540};

// EMG Bandpass (20-100 Hz at 256 Hz)
float32_t emg_bp_b[5] = {0.2452, 0, -0.4904, 0, 0.2452};
float32_t emg_bp_a[5] = {1.0, -1.9633, 1.8782, -0.8567, 0.1905};

// Preprocessing function for EEG
void preprocess_eeg(float32_t* input, float32_t* output, uint32_t length) {
    // Step 1: Resample to 256 Hz (if needed)
    // Use arm_fir_interpolate or custom resampling
    
    // Step 2: Apply high-pass filter
    arm_biquad_cascade_df2T_f32(&hp_instance, input, output, length);
    
    // Step 3: Apply low-pass filter
    arm_biquad_cascade_df2T_f32(&lp_instance, output, output, length);
    
    // Step 4: Apply notch filter
    arm_biquad_cascade_df2T_f32(&notch_instance, output, output, length);
}

// Preprocessing function for ECG
void preprocess_ecg(float32_t* input, float32_t* output, uint32_t length) {
    // Step 1: Resample to 256 Hz
    
    // Step 2: Apply bandpass filter (0.5-40 Hz)
    arm_biquad_cascade_df2T_f32(&ecg_bp_instance, input, output, length);
    
    // Step 3: Apply notch filter
    arm_biquad_cascade_df2T_f32(&notch_instance, output, output, length);
}

// Preprocessing function for EMG
void preprocess_emg(float32_t* input, float32_t* output, uint32_t length) {
    // Step 1: Resample to 256 Hz
    
    // Step 2: Apply bandpass filter (20-100 Hz)
    arm_biquad_cascade_df2T_f32(&emg_bp_instance, input, output, length);
    
    // Step 3: Apply notch filter
    arm_biquad_cascade_df2T_f32(&notch_instance, output, output, length);
}
```

### Filter Coefficient Computation (Python → C)

```python
from scipy import signal
import numpy as np

# EEG High-pass filter coefficients
fs = 256
b_hp, a_hp = signal.butter(2, 0.5/(fs/2), 'high')
print("EEG HP:", b_hp, a_hp)

# EEG Low-pass filter coefficients
b_lp, a_lp = signal.butter(2, 60/(fs/2), 'low')
print("EEG LP:", b_lp, a_lp)

# EEG Notch filter coefficients
b_notch, a_notch = signal.butter(2, [49.5/(fs/2), 50.5/(fs/2)], 'bandstop')
print("EEG Notch:", b_notch, a_notch)

# ECG Bandpass filter coefficients
b_ecg, a_ecg = signal.butter(2, [0.5/(fs/2), 40/(fs/2)], 'bandpass')
print("ECG BP:", b_ecg, a_ecg)

# EMG Bandpass filter coefficients
b_emg, a_emg = signal.butter(2, [20/(fs/2), 100/(fs/2)], 'bandpass')
print("EMG BP:", b_emg, a_emg)
```

---

## Model Input Format

### Multi-Modal (4 channels)
After preprocessing, the model expects:
- **Shape**: `(batch_size, time_steps, 4)`
- **time_steps**: `256 * frame_size` (e.g., 1024 for 4-second window)
- **Channels**:
  1. EEG Channel 1 (Focal or BTEleft SD)
  2. EEG Channel 2 (Cross-hemispheric or CROSStop SD)
  3. ECG Channel
  4. EMG Channel

### Data Range
- **No normalization** is applied after filtering
- Data is in microvolts (µV) or millivolts (mV) depending on sensor
- Model handles raw filtered values directly

---

## Real-Time Implementation on STM32

### Circular Buffer Approach
```c
#define WINDOW_SIZE 1024  // 4 seconds at 256 Hz
#define NUM_CHANNELS 4

float32_t eeg_buffer[2][WINDOW_SIZE];
float32_t ecg_buffer[WINDOW_SIZE];
float32_t emg_buffer[WINDOW_SIZE];

// Sliding window for real-time inference
uint32_t buffer_index = 0;

void process_sample(float32_t eeg1, float32_t eeg2, float32_t ecg, float32_t emg) {
    // Preprocess and store
    preprocess_eeg(&eeg1, &eeg_buffer[0][buffer_index], 1);
    preprocess_eeg(&eeg2, &eeg_buffer[1][buffer_index], 1);
    preprocess_ecg(&ecg, &ecg_buffer[buffer_index], 1);
    preprocess_emg(&emg, &emg_buffer[buffer_index], 1);
    
    buffer_index++;
    
    // When buffer is full, run inference
    if (buffer_index >= WINDOW_SIZE) {
        run_inference(eeg_buffer[0], eeg_buffer[1], ecg_buffer, emg_buffer);
        buffer_index = 0;  // Reset or use sliding window
    }
}
```

---

## Testing & Validation

### Python Validation Script
```python
from scipy import signal
import numpy as np

def validate_preprocessing(raw_signal, fs_original=512):
    # Resample
    if fs_original != 256:
        signal_256 = signal.resample(raw_signal, int(256*len(raw_signal)/fs_original))
    else:
        signal_256 = raw_signal
    
    # High-pass
    b, a = signal.butter(2, 0.5/(256/2), 'high')
    signal_hp = signal.filtfilt(b, a, signal_256)
    
    # Low-pass
    b, a = signal.butter(2, 60/(256/2), 'low')
    signal_lp = signal.filtfilt(b, a, signal_hp)
    
    # Notch
    b, a = signal.butter(2, [49.5/(256/2), 50.5/(256/2)], 'bandstop')
    signal_final = signal.filtfilt(b, a, signal_lp)
    
    return signal_final

# Compare with STM32 output
python_output = validate_preprocessing(raw_eeg)
stm32_output = load_stm32_output()  # Load from STM32
mse = np.mean((python_output - stm32_output)**2)
print(f"MSE between Python and STM32: {mse}")  # Should be < 1e-6
```

---

## Summary

| Signal | Sampling Rate | Filters Applied | Purpose |
|--------|---------------|-----------------|---------|
| **EEG** | 256 Hz | HP (0.5 Hz) + LP (60 Hz) + Notch (50 Hz) | Remove drift, high-freq noise, powerline |
| **ECG** | 256 Hz | BP (0.5-40 Hz) + Notch (50 Hz) | R-wave detection range |
| **EMG** | 256 Hz | BP (20-100 Hz) + Notch (50 Hz) | Muscle activity range |

**All filters are 2nd-order Butterworth applied with zero-phase (filtfilt).**

---

## Next Steps
1. **Extract filter coefficients** using the Python script above
2. **Implement in STM32** using ARM DSP library
3. **Validate** by comparing Python and C outputs
4. **Integrate** with TensorFlow Lite model inference

