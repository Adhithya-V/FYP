# utils/load_all_modalities.py
from utils.load_modalities import load_modalities_for_run
from classes.annotation import Annotation
import numpy as np

SEGMENT_SIZE = 2 * 250  # 2 sec windows at 250 Hz
STRIDE = int(SEGMENT_SIZE / 2)  # 50% overlap

def segment_signals(signals, fs, annotations):
    """
    Segments multi-modal signals into fixed windows and assigns pre-ictal labels
    """
    n_channels, n_samples = signals.shape
    segments = []
    labels = []

    for start in range(0, n_samples - SEGMENT_SIZE, STRIDE):
        seg = signals[:, start:start + SEGMENT_SIZE]
        seg_start_sec = start / fs
        seg_end_sec = (start + SEGMENT_SIZE) / fs
        label = 0  # default inter-ictal

        for seizure in annotations.seizures:
            pre_start = seizure['onset'] - seizure.get('pre_ictal_window', 60)
            pre_end = seizure['onset']
            if seg_start_sec >= pre_start and seg_end_sec <= pre_end:
                label = 1
                break

        segments.append(seg)
        labels.append(label)

    return np.stack(segments, axis=0), np.array(labels)

def load_and_segment(sub, run, data_path, fs=250, max_samples=5_000_000):
    """
    Loads all modalities and segments them
    """
    ses = "ses-01"
    signals, sig_fs, info = load_modalities_for_run(
        base_dir=data_path,
        sub=sub,
        ses=ses,
        run_token=run,
        prefer_fs=fs
    )
    if signals is None:
        return None, None

    # Cap samples
    if signals.shape[1] > max_samples:
        signals = signals[:, :max_samples]

    annotations = Annotation.loadAnnotation(data_path, [sub, run])
    X, y = segment_signals(signals, fs, annotations)
    return X, y
