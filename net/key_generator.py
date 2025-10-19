import numpy as np
import random
from tqdm import tqdm
from classes.annotation import Annotation


def generate_data_keys_sequential(config, recs_list, verbose=True):
    """Create data segment keys in a sequential time manner. The keys are 4 element lists corresponding to the file index in the 'recs_list', the start and stop in seconds of the segment and it's label.

        Args:
            config (cls): config object with the experiment's parameters.
            recs_list (list[list[str]]): a list of recording IDs in the format [sub-xxx, run-xx]
        Returns:
            segments: a list of data segment keys with [recording index, start, stop, label]
    """
    
    segments = []

    for idx, f in tqdm(enumerate(recs_list), disable = not verbose):
        annotations = Annotation.loadAnnotation(config.data_path, f)

        if not annotations.events:
            n_segs = int(np.floor((np.floor(annotations.rec_duration) - config.frame)/config.stride))
            seg_start = np.arange(0, n_segs)*config.stride
            seg_stop = seg_start + config.frame

            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            if len(annotations.events) == 1:
                ev = annotations.events[0]
                n_segs = int(np.floor((ev[0])/config.stride)-1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                n_segs = int(np.floor(np.floor(annotations.rec_duration - ev[1])/config.stride)-1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
            else:
                for e, ev in enumerate(annotations.events):
                    if e == 0:
                        n_segs = int(np.floor((ev[0])/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))
                    else:
                        n_segs = int(np.floor((ev[0] - annotations.events[e-1][1])/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + annotations.events[e-1][1]
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                    if e == len(annotations.events)-1:
                        n_segs = int(np.floor(np.floor(annotations.rec_duration - ev[1])/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

    return segments


def generate_data_keys_sequential_window(config, recs_list, t_add, verbose=True):
    """Create data segment keys in a sequential time manner with a time window added.

        Args:
            config (cls): config object with the experiment's parameters.
            recs_list (list[list[str]]): a list of recording IDs in the format [sub-xxx, run-xx]
            t_add (float): time window to add in seconds
        Returns:
            segments: a list of data segment keys with [recording index, start, stop, label]
    """
    
    segments = []

    for idx, f in tqdm(enumerate(recs_list), disable = not verbose):
        annotations = Annotation.loadAnnotation(config.data_path, f)

        if not annotations.events:
            n_segs = int(np.floor((np.floor(annotations.rec_duration) - config.frame)/config.stride))
            seg_start = np.arange(0, n_segs)*config.stride
            seg_stop = seg_start + config.frame

            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            if len(annotations.events) == 1:
                ev = annotations.events[0]
                n_segs = int(np.floor((ev[0] - t_add)/config.stride)-1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                n_segs = int(np.floor((ev[1] + t_add - ev[0] + t_add)/config.stride) + 1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride + ev[0] - t_add - config.stride
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                n_segs = int(np.floor(np.floor(annotations.rec_duration - ev[1] - t_add)/config.stride)-1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride + ev[1] + t_add
                    seg_stop = seg_start + config.frame
                    segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
            else:
                for e, ev in enumerate(annotations.events):
                    if e == 0:
                        n_segs = int(np.floor((ev[0] - t_add)/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                        n_segs = int(np.floor((ev[1] + t_add - ev[0] + t_add)/config.stride) + 1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[0] - t_add - config.stride
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))
                    else:
                        n_segs = int(np.floor((ev[0] - t_add - annotations.events[e-1][1] - t_add)/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + annotations.events[e-1][1] + t_add
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                        n_segs = int(np.floor((ev[1] + t_add - ev[0] + t_add)/config.stride) + 1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[0] - t_add - config.stride
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                    if e == len(annotations.events)-1:
                        n_segs = int(np.floor(np.floor(annotations.rec_duration - ev[1] - t_add)/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[1] + t_add
                            seg_stop = seg_start + config.frame
                            segments.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

    return segments


def generate_data_keys_subsample(config, recs_list, verbose=True):
    """Create data segment keys with subsampling for balanced classes.

        Args:
            config (cls): config object with the experiment's parameters.
            recs_list (list[list[str]]): a list of recording IDs in the format [sub-xxx, run-xx]
        Returns:
            segments: a list of data segment keys with [recording index, start, stop, label]
    """
    
    segments = []
    segments_NS = []  # Non-seizure segments
    segments_S = []   # Seizure segments

    for idx, f in tqdm(enumerate(recs_list), disable = not verbose):
        annotations = Annotation.loadAnnotation(config.data_path, f)

        if not annotations.events:
            n_segs = int(np.floor((np.floor(annotations.rec_duration) - config.frame)/config.stride))
            if n_segs < 0:
                n_segs = 0
            if n_segs > 0:
                seg_start = np.arange(0, n_segs)*config.stride
                seg_stop = seg_start + config.frame
                segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        else:
            if len(annotations.events) == 1:
                ev = annotations.events[0]
                n_segs = int(np.floor((ev[0])/config.stride)-1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride
                    seg_stop = seg_start + config.frame
                    segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                    seg_stop = seg_start + config.frame
                    segments_S.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                n_segs = int(np.floor(np.floor(annotations.rec_duration - ev[1])/config.stride)-1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                    seg_stop = seg_start + config.frame
                    segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
            else:
                for e, ev in enumerate(annotations.events):
                    if e == 0:
                        n_segs = int(np.floor((ev[0])/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride
                            seg_stop = seg_start + config.frame
                            segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                            seg_stop = seg_start + config.frame
                            segments_S.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))
                    else:
                        n_segs = int(np.floor((ev[0] - annotations.events[e-1][1])/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + annotations.events[e-1][1]
                            seg_stop = seg_start + config.frame
                            segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))

                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                            seg_stop = seg_start + config.frame
                            segments_S.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                    if e == len(annotations.events)-1:
                        n_segs = int(np.floor(np.floor(annotations.rec_duration - ev[1])/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                            seg_stop = seg_start + config.frame
                            segments_NS.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.zeros(n_segs))))
        
    # Subsample non-seizure segments
    if len(segments_NS) > len(segments_S) * config.factor:
        segments_NS = random.sample(segments_NS, len(segments_S) * config.factor)

    segments = segments_NS + segments_S
    random.shuffle(segments)

    return segments


def generate_data_keys_preictal(config, recs_list, verbose=True):
    """Create 3-class data segment keys: pre-ictal (0), ictal (1), inter-ictal (2).

        Args:
            config (cls): config object with the experiment's parameters.
            recs_list (list[list[str]]): a list of recording IDs in the format [sub-xxx, run-xx]
        Returns:
            segments: a list of data segment keys with [recording index, start, stop, label]
    """
    
    segments = []
    segments_interictal = []  # Class 2: Inter-ictal
    segments_ictal = []       # Class 1: Ictal  
    segments_preictal = []    # Class 0: Pre-ictal

    for idx, f in tqdm(enumerate(recs_list), disable = not verbose):
        annotations = Annotation.loadAnnotation(config.data_path, f)

        if not annotations.events:
            # No seizures - all inter-ictal
            n_segs = int(np.floor((np.floor(annotations.rec_duration) - config.frame)/config.stride))
            if n_segs < 0:
                n_segs = 0
            if n_segs > 0:
                seg_start = np.arange(0, n_segs)*config.stride
                seg_stop = seg_start + config.frame
                segments_interictal.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.full(n_segs, 2))))
        else:
            if len(annotations.events) == 1:
                ev = annotations.events[0]
                
                # Inter-ictal before first seizure
                n_segs = int(np.floor((ev[0])/config.stride)-1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride
                    seg_stop = seg_start + config.frame
                    segments_interictal.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.full(n_segs, 2))))

                # Pre-ictal window before seizure (configurable window)
                preictal_window = getattr(config, 'pre_ictal_window', 30)  # 30 seconds default
                preictal_start = max(0, ev[0] - preictal_window)
                n_preictal_segs = int(np.floor((ev[0] - preictal_start)/config.stride))
                if n_preictal_segs < 0:
                    n_preictal_segs = 0
                if n_preictal_segs > 0:
                    seg_start = np.arange(0, n_preictal_segs)*config.stride + preictal_start
                    seg_stop = seg_start + config.frame
                    segments_preictal.extend(np.column_stack(([idx]*n_preictal_segs, seg_start, seg_stop, np.zeros(n_preictal_segs))))

                # Ictal segments
                n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                    seg_stop = seg_start + config.frame
                    segments_ictal.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                # Inter-ictal after seizure
                n_segs = int(np.floor(np.floor(annotations.rec_duration - ev[1])/config.stride)-1)
                if n_segs < 0:
                    n_segs = 0
                if n_segs > 0:
                    seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                    seg_stop = seg_start + config.frame
                    segments_interictal.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.full(n_segs, 2))))
            else:
                # Multiple seizures
                for e, ev in enumerate(annotations.events):
                    if e == 0:
                        # Inter-ictal before first seizure
                        n_segs = int(np.floor((ev[0])/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride
                            seg_stop = seg_start + config.frame
                            segments_interictal.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.full(n_segs, 2))))

                        # Pre-ictal before first seizure
                        preictal_window = getattr(config, 'pre_ictal_window', 30)
                        preictal_start = max(0, ev[0] - preictal_window)
                        n_preictal_segs = int(np.floor((ev[0] - preictal_start)/config.stride))
                        if n_preictal_segs < 0:
                            n_preictal_segs = 0
                        if n_preictal_segs > 0:
                            seg_start = np.arange(0, n_preictal_segs)*config.stride + preictal_start
                            seg_stop = seg_start + config.frame
                            segments_preictal.extend(np.column_stack(([idx]*n_preictal_segs, seg_start, seg_stop, np.zeros(n_preictal_segs))))

                        # Ictal segments
                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                            seg_stop = seg_start + config.frame
                            segments_ictal.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))
                    else:
                        # Inter-ictal between seizures
                        n_segs = int(np.floor((ev[0] - annotations.events[e-1][1])/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + annotations.events[e-1][1]
                            seg_stop = seg_start + config.frame
                            segments_interictal.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.full(n_segs, 2))))

                        # Pre-ictal before current seizure
                        preictal_window = getattr(config, 'pre_ictal_window', 30)
                        preictal_start = max(annotations.events[e-1][1], ev[0] - preictal_window)
                        n_preictal_segs = int(np.floor((ev[0] - preictal_start)/config.stride))
                        if n_preictal_segs < 0:
                            n_preictal_segs = 0
                        if n_preictal_segs > 0:
                            seg_start = np.arange(0, n_preictal_segs)*config.stride + preictal_start
                            seg_stop = seg_start + config.frame
                            segments_preictal.extend(np.column_stack(([idx]*n_preictal_segs, seg_start, seg_stop, np.zeros(n_preictal_segs))))

                        # Ictal segments
                        n_segs = int(np.floor((ev[1] - ev[0])/config.stride) + 1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[0] - config.stride
                            seg_stop = seg_start + config.frame
                            segments_ictal.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.ones(n_segs))))

                    if e == len(annotations.events)-1:
                        # Inter-ictal after last seizure
                        n_segs = int(np.floor(np.floor(annotations.rec_duration - ev[1])/config.stride)-1)
                        if n_segs < 0:
                            n_segs = 0
                        if n_segs > 0:
                            seg_start = np.arange(0, n_segs)*config.stride + ev[1]
                            seg_stop = seg_start + config.frame
                            segments_interictal.extend(np.column_stack(([idx]*n_segs, seg_start, seg_stop, np.full(n_segs, 2))))

    # Apply class balancing with emphasis on pre-ictal (but maintain temporal order)
    interictal_subsample_factor = getattr(config, 'inter_ictal_subsample_factor', 0.3)
    if len(segments_interictal) > len(segments_preictal) * interictal_subsample_factor:
        # Subsample interictal segments but maintain temporal order
        subsample_size = int(len(segments_preictal) * interictal_subsample_factor)
        step = len(segments_interictal) // subsample_size
        segments_interictal = segments_interictal[::max(1, step)][:subsample_size]

    # Combine all segments in temporal order (NO shuffling for biosignals!)
    segments = segments_interictal + segments_ictal + segments_preictal
    # Sort by recording index, then by start time to maintain temporal sequence
    segments = sorted(segments, key=lambda x: (x[0], x[1]))

    return segments