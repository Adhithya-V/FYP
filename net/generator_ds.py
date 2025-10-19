import numpy as np
from tensorflow import keras
from net.utils import apply_preprocess_eeg, apply_preprocess_multimodal
from tqdm import tqdm
from classes.data import Data
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


class SequentialGenerator(keras.utils.Sequence):
    ''' Class where a keras sequential data generator is built (the data segments are continuous and aligned in time).

    Args:
        config (cls): config object with the experiment parameters
        recs (list[list[str]]): list of recordings in the format [sub-xxx, run-xx]
        segments: list of keys (each key is a list [1x4] containing the recording index in the rec list,
                  the start and stop of the segment in seconds and the label of the segment)
        batch_size: batch size of the generator
        shuffle: boolean, if True, the segments are randomly mixed in every batch
    
    '''

    def __init__(self, config, recs, segments, batch_size=32, shuffle=False, verbose=True):
        
        'Initialization'
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data_segs = np.empty(shape=[len(segments), int(config.frame*config.fs), config.CH])
        # Support for 3-class prediction
        num_classes = getattr(config, 'num_classes', 2)
        self.labels = np.empty(shape=[len(segments), num_classes])
        self.verbose = verbose
        
        pbar = tqdm(total = len(segments)+1, disable = not self.verbose)

        count = 0
        prev_rec = int(segments[0][0])

        # Use multi-modal loading if specified in config
        modalities = getattr(config, 'modalities', ['eeg'])
        if len(modalities) > 1:
            rec_data_raw = Data.loadData(config.data_path, recs[prev_rec], modalities=modalities)
            rec_data = apply_preprocess_multimodal(config, rec_data_raw)
        else:
            rec_data_raw = Data.loadData(config.data_path, recs[prev_rec], modalities=['eeg'])
            rec_data = apply_preprocess_eeg(config, rec_data_raw)
        
        for s in segments:
            curr_rec = int(s[0])
            
            if curr_rec != prev_rec:
                if len(modalities) > 1:
                    rec_data_raw = Data.loadData(config.data_path, recs[curr_rec], modalities=modalities)
                    rec_data = apply_preprocess_multimodal(config, rec_data_raw)
                else:
                    rec_data_raw = Data.loadData(config.data_path, recs[curr_rec], modalities=['eeg'])
                    rec_data = apply_preprocess_eeg(config, rec_data_raw)
                prev_rec = curr_rec

            start_seg = int(s[1]*config.fs)
            stop_seg = int(s[2]*config.fs)

            # Ensure segment is within bounds and not empty
            if start_seg >= 0 and stop_seg <= len(rec_data[0]) and stop_seg > start_seg:
                # Fill all channels dynamically based on config.CH
                for ch_idx in range(min(config.CH, len(rec_data))):
                    self.data_segs[count, :, ch_idx] = rec_data[ch_idx][start_seg:stop_seg]
            else:
                # Fill with zeros if segment is invalid
                for ch_idx in range(config.CH):
                    self.data_segs[count, :, ch_idx] = np.zeros(config.fs*config.frame)

            # 3-class label encoding
            label = int(s[3])
            if num_classes == 3:
                if label == 0:  # Pre-ictal
                    self.labels[count, :] = [1, 0, 0]
                elif label == 1:  # Ictal
                    self.labels[count, :] = [0, 1, 0]
                elif label == 2:  # Inter-ictal
                    self.labels[count, :] = [0, 0, 1]
            else:  # 2-class (backward compatibility)
                if label == 1:
                    self.labels[count, :] = [0, 1]
                elif label == 0:
                    self.labels[count, :] = [1, 0]

            count += 1
            pbar.update(1)

        
        self.key_array = np.arange(len(self.labels))

        self.on_epoch_end()


    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = np.arange(start=index * self.batch_size, stop=(index + 1) * self.batch_size)
        x, y = self.__data_generation__(keys)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        if self.config.model == 'DeepConvNet' or self.config.model == 'EEGnet':
            out = self.data_segs[self.key_array[keys], :, :, np.newaxis].transpose(0,2,1,3), self.labels[self.key_array[keys]]
        else:
            out = self.data_segs[self.key_array[keys], :, :], self.labels[self.key_array[keys]]
        return out



class SegmentedGenerator(keras.utils.Sequence):
    ''' Class where the keras segmented data generator is built, implemented as a more efficient way to load segments that were subsampled from multiple recordings.

    Args:
        config (cls): config object with the experiment parameters
        recs (list[list[str]]): list of recordings in the format [sub-xxx, run-xx]
        segments: list of keys (each key is a list [1x4] containing the recording index in the rec list,
                  the start and stop of the segment in seconds and the label of the segment)
        batch_size: batch size of the generator
        shuffle: boolean, if True, the segments are randomly mixed in every batch
    
    '''

    def __init__(self, config, recs, segments, batch_size=32, shuffle=True, verbose=True, num_workers=4):
        
        'Initialization'
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.num_workers = num_workers

        self.data_segs = np.empty(shape=[len(segments), int(config.frame*config.fs), config.CH])
        # Support for 3-class prediction
        num_classes = getattr(config, 'num_classes', 2)
        self.labels = np.empty(shape=[len(segments), num_classes])
        segs_to_load = segments

        pbar = tqdm(total = len(segs_to_load)+1, disable=self.verbose)
        count = 0

        # CONSERVATIVE: Process recordings sequentially to avoid OOM
        # Use multi-modal loading if specified in config
        modalities = getattr(config, 'modalities', ['eeg'])
        
        while segs_to_load:
            curr_rec = int(segs_to_load[0][0])
            comm_recs = [i for i, x in enumerate(segs_to_load) if x[0] == curr_rec]

            if len(modalities) > 1:
                rec_data_raw = Data.loadData(config.data_path, recs[curr_rec], modalities=modalities)
                rec_data = apply_preprocess_multimodal(config, rec_data_raw)
            else:
                rec_data_raw = Data.loadData(config.data_path, recs[curr_rec], modalities=['eeg'])
                rec_data = apply_preprocess_eeg(config, rec_data_raw)

            for r in comm_recs:
                start_seg = int(segs_to_load[r][1]*config.fs)
                stop_seg = int(segs_to_load[r][2]*config.fs)
                
                # Ensure segment is within bounds and not empty
                if start_seg >= 0 and stop_seg <= len(rec_data[0]) and stop_seg > start_seg:
                    # Fill all channels dynamically based on config.CH
                    for ch_idx in range(min(config.CH, len(rec_data))):
                        self.data_segs[count, :, ch_idx] = rec_data[ch_idx][start_seg:stop_seg]
                else:
                    # Fill with zeros if segment is invalid
                    for ch_idx in range(config.CH):
                        self.data_segs[count, :, ch_idx] = np.zeros(config.fs*config.frame)

                # 3-class label encoding
                label = int(segs_to_load[r][3])
                if num_classes == 3:
                    if label == 0:  # Pre-ictal
                        self.labels[count, :] = [1, 0, 0]
                    elif label == 1:  # Ictal
                        self.labels[count, :] = [0, 1, 0]
                    elif label == 2:  # Inter-ictal
                        self.labels[count, :] = [0, 0, 1]
                else:  # 2-class (backward compatibility)
                    if label == 1:
                        self.labels[count, :] = [0, 1]
                    elif label == 0:
                        self.labels[count, :] = [1, 0]
                
                count += 1
                pbar.update(1)
                
            segs_to_load = [s for i, s in enumerate(segs_to_load) if i not in comm_recs]
        
        self.key_array = np.arange(len(self.labels))

        self.on_epoch_end()


    def __len__(self):
        return len(self.key_array) // self.batch_size

    def __getitem__(self, index):
        keys = np.arange(start=index * self.batch_size, stop=(index + 1) * self.batch_size)
        x, y = self.__data_generation__(keys)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        if self.config.model == 'DeepConvNet' or self.config.model == 'EEGnet':
            out = self.data_segs[self.key_array[keys], :, :, np.newaxis].transpose(0,2,1,3), self.labels[self.key_array[keys]]
        else:
            out = self.data_segs[self.key_array[keys], :, :], self.labels[self.key_array[keys]]
        return out


    
    
