import os
import gc

import pandas as pd
import time
import h5py
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

import tensorflow as tf

from net.key_generator import generate_data_keys_sequential, generate_data_keys_subsample, generate_data_keys_sequential_window, generate_data_keys_preictal
from net.generator_ds import SegmentedGenerator, SequentialGenerator
from net.routines import train_net, predict_net
from net.utils import apply_preprocess_eeg, get_metrics_scoring

from classes.data import Data

def train(config, load_generators, save_generators, X=None, y=None):
    """ Routine to run the model's training routine.

        Args:
            config (cls): a config object with the data input type and model parameters
            load_generators (bool): boolean to load the training and validation generators from file
            save_generators (bool): boolean to save the training and validation generators
            X (np.ndarray, optional): pre-segmented data array shape (n_segments, channels, samples)
            y (np.ndarray, optional): labels shape (n_segments,)
    """

    name = config.get_name()

    if config.model == 'ChronoNet':
        from net.ChronoNet import net
    elif config.model == 'EEGnet':
        from net.EEGnet import net
    elif config.model == 'DeepConvNet':
        from net.DeepConv_Net import net

    # make directories if necessary
    if not os.path.exists(os.path.join(config.save_dir, 'models')):
        os.mkdir(os.path.join(config.save_dir, 'models'))

    model_save_path = os.path.join(config.save_dir, 'models', name)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    weights_dir = os.path.join(model_save_path, 'Weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir, exist_ok=True)

    config_path = os.path.join(config.save_dir, 'models', name, 'configs')
    if not os.path.exists(config_path):
        os.mkdir(config_path)

    # always save config for reproducibility
    try:
        config.save_config(save_path=config_path)
    except Exception:
        # some configs may not implement save_config, ignore if missing
        pass

    #######################################################################################################################
    # If X,y provided --> train directly on arrays (fast path). Otherwise use original generator pipeline.
    #######################################################################################################################
    if X is not None and y is not None:
        print("[INFO] Training on provided X,y arrays (direct training path).")
        # X expected shape: (n_segments, n_channels, n_samples)
        # ChronoNet / existing code expects (channels, samples) per example; we will reshape into (n_segments, channels, samples)
        # Keras generally expects (batch, time, features) or (batch, channels, time). To avoid making intrusive changes we will
        # use the model returned by net(config) and rely on its input shape expectation.
        # Create model using existing factory and compile
        model = net(config)

        # compile model (binary pre-ictal vs inter-ictal assumed)
        optimizer = tf.keras.optimizers.Adam(learning_rate=getattr(config, 'lr', 0.001))
        try:
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        except Exception as e:
            # Some custom nets may expose different compile expectations
            print("[WARNING] model.compile failed:", e)
            # attempt a generic compile anyway
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Keras expects input shape (batch, timesteps, channels) or (batch, channels, timesteps) depending on model.
        # We will try to detect which one works by attempting to fit; if it errors we transpose segments.
        X_train = np.array(X)  # (n_segments, channels, samples)
        y_train = np.array(y)

        # Attempt fit; if it fails due to shape mismatch, try transposed
        history = None
        try:
            history = model.fit(X_train, y_train,
                                batch_size=getattr(config, 'batch_size', 64),
                                epochs=getattr(config, 'nb_epochs', 10),
                                validation_split=0.1,
                                shuffle=True,
                                verbose=1)
        except Exception as e:
            # try transposed shape
            print("[INFO] Fit failed with shape (n, ch, samples). Trying transposed X (n, samples, ch). Error:", e)
            try:
                X_train_t = np.transpose(X_train, (0, 2, 1))
                history = model.fit(X_train_t, y_train,
                                    batch_size=getattr(config, 'batch_size', 64),
                                    epochs=getattr(config, 'nb_epochs', 10),
                                    validation_split=0.1,
                                    shuffle=True,
                                    verbose=1)
                # keep the model and continue
                X_train = X_train_t
            except Exception as e2:
                print("[ERROR] Both shapes failed to fit model. Exception:", e2)
                raise e2

        # attach model to config for saving later
        config.model_instance = model

        # save weights in expected location (so predict() can use them)
        weights_path = os.path.join(weights_dir, name + '.h5')
        try:
            # prefer saving TensorFlow SavedModel if model.save works
            model.save(weights_path)
        except Exception:
            # fallback: save weights only
            model.save_weights(weights_path)
        print(f"[INFO] Saved trained model to {weights_path}")

        # Build TrainResult
        class TrainResult:
            def __init__(self, model, history):
                self.model = model
                self.history = history
                self.loss = history.history.get('loss', [])
                self.accuracy = history.history.get('accuracy', [])

        return TrainResult(model, history)

    #######################################################################################################################
    # ELSE: original generator-based training (unchanged)
    #######################################################################################################################
    # The rest of your original generator-based code follows unchanged:
    if config.cross_validation == 'fixed':
        
        if config.dataset == 'SZ2':

            train_pats_list = pd.read_csv(os.path.join('net', 'datasets', 'SZ2_training.tsv'), sep = '\t', header = None, skiprows = [0,1,2])
            train_pats_list = train_pats_list[0].to_list()
            train_recs_list = [[s, r.split('_')[-2]] for s in train_pats_list for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg')) if 'edf' in r]

            if load_generators:
                print('Loading generators...')
                name = config.dataset + '_frame-' + str(config.frame) + '_sampletype-' + config.sample_type
                with open(os.path.join('net/generators', 'gen_train_' + name + '.pkl'), 'rb') as inp:
                    gen_train = pickle.load(inp)

                with open('net/generators/gen_val.pkl', 'rb') as inp:
                    gen_val = pickle.load(inp)

            else:
                # Use preictal generator for 3-class, subsample for 2-class
                num_classes = getattr(config, 'num_classes', 2)
                if num_classes == 3:
                    train_segments = generate_data_keys_preictal(config, train_recs_list)
                elif config.sample_type == 'subsample':
                    train_segments = generate_data_keys_subsample(config, train_recs_list)

                print('Generating training segments...')
                # Use SegmentedGenerator but with shuffle=False to maintain temporal order
                gen_train = SegmentedGenerator(config, train_recs_list, train_segments, 
                                              batch_size=config.batch_size, shuffle=False, 
                                              num_workers=4, verbose=False)

                if save_generators:
                    name = config.dataset + '_frame-' + str(config.frame) + '_sampletype-' + config.sample_type
                    if not os.path.exists('net/generators'):
                        os.mkdir('net/generators')

                    with open(os.path.join('net/generators', 'gen_train_' + name + '.pkl'), 'wb') as outp:
                        pickle.dump(gen_train, outp, pickle.HIGHEST_PROTOCOL)

                val_pats_list = pd.read_csv(os.path.join('net', 'datasets', 'SZ2_validation.tsv'), sep = '\t', header = None, skiprows = [0,1,2])
                val_pats_list = val_pats_list[0].to_list()
                val_recs_list = [[s, r.split('_')[-2]] for s in val_pats_list for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg')) if 'edf' in r]
                
                # Use preictal generator for 3-class validation too
                if num_classes == 3:
                    val_segments = generate_data_keys_preictal(config, val_recs_list)
                else:
                    val_segments = generate_data_keys_sequential_window(config, val_recs_list, 5*60)

                print('Generating validation segments...')
                gen_val = SequentialGenerator(config, val_recs_list, val_segments, batch_size=600, shuffle=False)

                if save_generators:
                    with open('net/generators/gen_val.pkl', 'wb') as outp:
                        pickle.dump(gen_val, outp, pickle.HIGHEST_PROTOCOL)

            print('### Training model....')
            
            model = net(config)

            start_train = time.time()
            
            history = train_net(config, model, gen_train, gen_val, model_save_path)
            
            end_train = time.time() - start_train
            print('Total train duration = ', end_train / 60)
            
            return history  # from model.fit()



#######################################################################################################################
#######################################################################################################################


def predict(config):

    name = config.get_name()

    model_save_path = os.path.join(config.save_dir, 'models', name)

    if not os.path.exists(os.path.join(config.save_dir, 'predictions')):
        os.mkdir(os.path.join(config.save_dir, 'predictions'))
    if not os.path.exists(os.path.join(config.save_dir, 'predictions', name)):
        os.mkdir(os.path.join(config.save_dir, 'predictions', name))

    test_pats_list = pd.read_csv(os.path.join('net', 'datasets', config.dataset + '_test.tsv'), sep = '\t', header = None, skiprows = [0,1,2])
    test_pats_list = test_pats_list[0].to_list()
    test_recs_list = [[s, r.split('_')[-2]] for s in test_pats_list for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg')) if 'edf' in r]

    model_weights_path = os.path.join(model_save_path, 'Weights', name + '.h5')

    config.load_config(config_path=os.path.join(model_save_path, 'configs'), config_name=name+'.cfg')
        
    if config.model == 'DeepConvNet':
        from net.DeepConv_Net import net
    elif config.model == 'ChronoNet':
        from net.ChronoNet import net
    elif config.model == 'EEGnet':
        from net.EEGnet import net

    for rec in tqdm(test_recs_list):
        if os.path.isfile(os.path.join(config.save_dir, 'predictions', name, rec[0] + '_' + rec[1] + '_preds.h5')):
            print(rec[0] + ' ' + rec[1] + ' exists. Skipping...')
        else:
            
            with tf.device('/cpu:0'):
                segments = generate_data_keys_sequential(config, [rec], verbose=False)

                gen_test = SequentialGenerator(config, [rec], segments, batch_size=len(segments), shuffle=False, verbose=False)

                model = net(config)

                y_pred, y_true = predict_net(gen_test, model_weights_path, model)

            with h5py.File(os.path.join(config.save_dir, 'predictions', name, rec[0] + '_' + rec[1] + '_preds.h5'), 'w') as f:
                f.create_dataset('y_pred', data=y_pred)
                f.create_dataset('y_true', data=y_true)

            gc.collect()

   
#######################################################################################################################
#######################################################################################################################


def evaluate(config):

    name = config.get_name()

    pred_path = os.path.join(config.save_dir, 'predictions', name)
    pred_fs = 1

    thresholds = list(np.around(np.linspace(0,1,51),2))

    x_plot = np.linspace(0, 200, 200)

    if not os.path.exists(os.path.join(config.save_dir, 'results')):
        os.mkdir(os.path.join(config.save_dir, 'results'))

    result_file = os.path.join(config.save_dir, 'results', name + '.h5')

    sens_ovlp = []
    prec_ovlp = []
    fah_ovlp = []
    sens_ovlp_plot = []
    prec_ovlp_plot = []
    f1_ovlp = []

    sens_epoch = []
    spec_epoch = []
    prec_epoch = []
    fah_epoch = []
    f1_epoch = []

    score = []

    pred_files = [x for x in os.listdir(pred_path)]
    pred_files.sort()

    for file in tqdm(pred_files):
        with h5py.File(os.path.join(pred_path, file), 'r') as f:
            y_pred = list(f['y_pred'])
            y_true = list(f['y_true'])

        sens_ovlp_th = []
        prec_ovlp_th = []
        fah_ovlp_th = []
        f1_ovlp_th = []

        sens_epoch_th = []
        spec_epoch_th = []
        prec_epoch_th = []
        fah_epoch_th = []
        f1_epoch_th = []

        score_th = []

        rec = [file.split('_')[0], file.split('_')[1]]

        rec_data = Data.loadData(config.data_path, rec, modalities=['eeg'])

        [ch_focal, ch_cross] = apply_preprocess_eeg(config, rec_data)

        rmsa_f = [np.sqrt(np.mean(ch_focal[start:start+2*config.fs]**2)) for start in range(0, len(ch_focal) - 2*config.fs + 1, 1*config.fs)]
        rmsa_c = [np.sqrt(np.mean(ch_cross[start:start+2*config.fs]**2)) for start in range(0, len(ch_focal) - 2*config.fs + 1, 1*config.fs)]
        rmsa_f = [1 if 13 < rms < 150 else 0 for rms in rmsa_f]
        rmsa_c = [1 if 13 < rms < 150 else 0 for rms in rmsa_c]
        rmsa = rmsa_f and rmsa_c
        
        if len(y_pred) != len(rmsa):
            rmsa = rmsa[:len(y_pred)]
        y_pred = np.where(np.array(rmsa) == 0, 0, y_pred)

        for th in thresholds:
            sens_ovlp_rec, prec_ovlp_rec, FA_ovlp_rec, f1_ovlp_rec, sens_epoch_rec, spec_epoch_rec, prec_epoch_rec, FA_epoch_rec, f1_epoch_rec = get_metrics_scoring(y_pred, y_true, pred_fs, th)

            sens_ovlp_th.append(sens_ovlp_rec)
            prec_ovlp_th.append(prec_ovlp_rec)
            fah_ovlp_th.append(FA_ovlp_rec)
            f1_ovlp_th.append(f1_ovlp_rec)
            sens_epoch_th.append(sens_epoch_rec)
            spec_epoch_th.append(spec_epoch_rec)
            prec_epoch_th.append(prec_epoch_rec)
            fah_epoch_th.append(FA_epoch_rec)
            f1_epoch_th.append(f1_epoch_rec)
            score_th.append(sens_ovlp_rec*100-0.4*FA_epoch_rec)

        sens_ovlp.append(sens_ovlp_th)
        prec_ovlp.append(prec_ovlp_th)
        fah_ovlp.append(fah_ovlp_th)
        f1_ovlp.append(f1_ovlp_th)

        sens_epoch.append(sens_epoch_th)
        spec_epoch.append(spec_epoch_th)
        prec_epoch.append(prec_epoch_th)
        fah_epoch.append(fah_epoch_th)
        f1_epoch.append(f1_epoch_th)

        score.append(score_th)

        to_cut = np.argmax(fah_ovlp_th)
        fah_ovlp_plot_rec = fah_ovlp_th[to_cut:]
        sens_ovlp_plot_rec = sens_ovlp_th[to_cut:]
        prec_ovlp_plot_rec = prec_ovlp_th[to_cut:]

        y_plot = np.interp(x_plot, fah_ovlp_plot_rec[::-1], sens_ovlp_plot_rec[::-1])
        sens_ovlp_plot.append(y_plot)
        y_plot = np.interp(x_plot, sens_ovlp_plot_rec[::-1], prec_ovlp_plot_rec[::-1])
        prec_ovlp_plot.append(y_plot)

    score_05 = [x[25] for x in score]

    print('Score: ' + "%.2f" % np.nanmean(score_05))

    with h5py.File(result_file, 'w') as f:
        f.create_dataset('sens_ovlp', data=sens_ovlp)
        f.create_dataset('prec_ovlp', data=prec_ovlp)
        f.create_dataset('fah_ovlp', data=fah_ovlp)
        f.create_dataset('f1_ovlp', data=f1_ovlp)
        f.create_dataset('sens_ovlp_plot', data=sens_ovlp_plot)
        f.create_dataset('prec_ovlp_plot', data=prec_ovlp_plot)
        f.create_dataset('x_plot', data=x_plot)
        f.create_dataset('sens_epoch', data=sens_epoch)
        f.create_dataset('spec_epoch', data=spec_epoch)
        f.create_dataset('prec_epoch', data=prec_epoch)
        f.create_dataset('fah_epoch', data=fah_epoch)
        f.create_dataset('f1_epoch', data=f1_epoch)
        f.create_dataset('score', data=score)

