# utils/load_modalities.py
"""
Minimal, non-invasive helper to load EEG+ECG+EMG+MOV for a single run.
It uses pyedflib to read EDF files, optionally resamples non-matching sampling rates
to the EEG sampling rate, and concatenates channels along the channel axis.

Usage (example):
  python utils\load_modalities.py --base "C:\Adhi\ds005873-download" --sub sub-001 --ses ses-01 --run 01
"""

import os
import glob
import argparse
import numpy as np
import pyedflib
from scipy.signal import resample

MODALITIES = ["eeg", "ecg", "emg", "mov"]

def find_first_edf_for_run(base_dir, sub, ses, modality, run_token=None):
    """
    Find the edf file for the given modality and run. If run_token=None, just take first file.
    """
    d = os.path.join(base_dir, sub, ses, modality)
    if not os.path.isdir(d):
        return None
    # typical filename pattern includes run-XX
    pattern = os.path.join(d, "*.edf")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    if run_token:
        for f in files:
            if run_token in os.path.basename(f):
                return f
        return None
    return files[0]

def read_edf_signals(edf_path):
    """
    Read EDF using pyedflib and return (signals, sample_rate).
    signals is numpy array with shape (n_channels, n_samples).
    """
    if edf_path is None:
        return None, None
    try:
        f = pyedflib.EdfReader(edf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open EDF {edf_path}: {e}")
    try:
        n = f.signals_in_file
        if n == 0:
            return None, None
        sigs = []
        for i in range(n):
            s = f.readSignal(i)
            sigs.append(s)
        sigs = np.vstack(sigs)  # shape: (n_channels, n_samples)
        # sample frequency per channel: getSampleFrequencies not always available, use first channel
        try:
            fs = int(f.getSampleFrequency(0))
        except Exception:
            # fallback estimate using length and header duration if available
            fs = None
        return sigs, fs
    finally:
        f._close()  # ensure file closed
        del f

def resample_to_target(sig_array, orig_fs, target_fs):
    """
    sig_array: np.ndarray shape (n_channels, n_samples)
    returns array resampled to target_fs along axis=1
    """
    if sig_array is None:
        return None
    if orig_fs is None or target_fs is None:
        # can't resample if unknown; return original and user will decide
        return sig_array
    if orig_fs == target_fs:
        return sig_array
    n_samples = sig_array.shape[1]
    new_n = int(round(n_samples * (target_fs / orig_fs)))
    # resample per channel using scipy.signal.resample
    res = resample(sig_array, new_n, axis=1)
    return res

def load_modalities_for_run(base_dir, sub, ses, run_token=None, prefer_fs=None):
    """
    Load EEG + ECG + EMG + MOV for a single subject/session/run.
    Returns (combined_signals, fs_used, modality_info)
    combined_signals: np.ndarray shape (total_channels, n_samples)
    fs_used: sampling rate used (int or None)
    modality_info: dict { modality: (path, channels, fs) }
    """
    modality_info = {}
    modality_arrays = []
    target_fs = prefer_fs  # if user passed desired FS

    # step 1: read EEG first (prefer it as master sampling rate)
    eeg_path = find_first_edf_for_run(base_dir, sub, ses, "eeg", run_token)
    eeg_sigs, eeg_fs = read_edf_signals(eeg_path)
    modality_info["eeg"] = (eeg_path, None if eeg_sigs is None else eeg_sigs.shape[0], eeg_fs)
    if eeg_sigs is not None:
        modality_arrays.append(eeg_sigs)
    if target_fs is None:
        target_fs = eeg_fs  # pick eeg sampling rate as target by default

    # step 2: load other modalities and resample if needed
    for mod in ["ecg", "emg", "mov"]:
        p = find_first_edf_for_run(base_dir, sub, ses, mod, run_token)
        s, fs = read_edf_signals(p)
        modality_info[mod] = (p, None if s is None else s.shape[0], fs)
        if s is None:
            continue
        if target_fs is not None and fs is not None and fs != target_fs:
            s = resample_to_target(s, fs, target_fs)
        elif target_fs is None and fs is not None:
            # if no target yet, adopt this fs
            target_fs = fs
        modality_arrays.append(s)

    if len(modality_arrays) == 0:
        return None, None, modality_info

    # align lengths: truncate to shortest length among arrays (to be conservative)
    lengths = [arr.shape[1] for arr in modality_arrays if arr is not None]
    min_len = min(lengths)
    trimmed = [arr[:, :min_len] for arr in modality_arrays]
    combined = np.concatenate(trimmed, axis=0)  # channels stacked
    return combined, target_fs, modality_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base dataset dir (e.g. C:\\Adhi\\ds005873-download)")
    parser.add_argument("--sub", required=True, help="subject folder name (e.g. sub-001)")
    parser.add_argument("--ses", required=True, help="session folder (e.g. ses-01)")
    parser.add_argument("--run", default=None, help="optional run token (e.g. run-01 or 01)")
    args = parser.parse_args()

    combined, fs, info = load_modalities_for_run(args.base, args.sub, args.ses, args.run)
    print("Modalities info:")
    for k,v in info.items():
        print(f"  {k}: path={v[0]}, channels={v[1]}, fs={v[2]}")
    if combined is None:
        print("No signals loaded for the requested run.")
    else:
        print("Combined shape (channels, samples):", combined.shape, "fs:", fs)
