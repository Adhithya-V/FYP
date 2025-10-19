import os
import numpy as np
from net.utils import weighted_focal_loss, sens, spec, sens_ovlp, fah_ovlp, fah_epoch, faRate_epoch, score, decay_schedule
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import LearningRateScheduler


def train_net(config, model, gen_train, gen_val, model_save_path):
    ''' Routine to train the model with the desired configurations.

        Args:
            config: configuration object containing all parameters
            model: Keras Model object
            gen_train: a keras data generator containing the training data
            gen_val: a keras data generator containing the validation data
            model_save_path: path to the folder to save the models' weights
    '''

    K.set_image_data_format('channels_last') 

    model.summary()

    name = config.get_name()

    optimizer = Adam(learning_rate=config.lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    # For 3-class, use categorical crossentropy instead of custom focal loss
    num_classes = getattr(config, 'num_classes', 2)
    if num_classes == 3:
        loss = 'categorical_crossentropy'
        metrics = ['accuracy']  # Simple metrics for 3-class
    else:
        loss = [weighted_focal_loss]
        auc = AUC(name = 'auc')
        metrics = ['accuracy', sens, spec,sens_ovlp, fah_ovlp, fah_epoch, faRate_epoch, score, auc]

    # For 3-class, monitor validation accuracy instead of score
    if num_classes == 3:
        monitor = 'val_accuracy'
        monitor_mode = 'max'
    else:
        monitor = 'val_score'
        monitor_mode = 'max'

    early_stopping = False
    patience = 50

    if not os.path.exists(os.path.join(model_save_path, 'Callbacks')):
        os.mkdir(os.path.join(model_save_path, 'Callbacks'))

    if not os.path.exists(os.path.join(model_save_path, 'History')):
        os.mkdir(os.path.join(model_save_path, 'History'))

    if not os.path.exists(os.path.join(model_save_path, 'Weights')):
        os.mkdir(os.path.join(model_save_path, 'Weights'))


    cb_model = os.path.join(model_save_path, 'Callbacks', name + '_{epoch:02d}.weights.h5')
    # DISABLE CSV logging to save disk space
    # csv_logger = CSVLogger(os.path.join(model_save_path, 'History', name + '.csv'), append=True)

    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

    mc = ModelCheckpoint(cb_model,
                         monitor=monitor,
                         verbose=1,
                         save_weights_only=True,
                         save_freq='epoch',
                         save_best_only=True,  # ONLY save best model to save disk space
                         mode=monitor_mode)


    if early_stopping:
        es = EarlyStopping(monitor=monitor,
                           patience=patience,
                           verbose=1,
                           mode='min')

    lr_sched = LearningRateScheduler(decay_schedule)

    if early_stopping:
        callbacks_list = [mc, es, lr_sched]  # Removed csv_logger
    else:
        callbacks_list = [mc, lr_sched]  # Removed csv_logger

    hist = model.fit(gen_train, validation_data=gen_val,
                     epochs=config.nb_epochs,
                     callbacks=callbacks_list,
                     shuffle=False,
                     verbose=1,
                     class_weight=config.class_weights)

    # serialize weights to HDF5 (STM Nucleo compatible format)
    best_model = model
    # For 3-class, use val_accuracy as the metric to find best epoch
    if num_classes == 3:
        best_epoch = np.argmax(hist.history['val_accuracy']) + 1
    else:
        best_epoch = np.argmax(hist.history['val_score']) + 1
    best_model.load_weights(cb_model.format(epoch=best_epoch))
    best_model.save_weights(os.path.join(model_save_path, 'Weights', name + ".weights.h5"))

    print("Saved model to disk")
    
    return hist


def predict_net(generator, model_weights_path, model):
    ''' Routine to obtain predictions from the trained model with the desired configurations.

    Args:
        generator: a keras data generator containing the data to predict
        model_weights_path: path to the folder containing the models' weights
        model: keras model object

    Returns:
        y_pred: array with the probability of seizure occurences (0 to 1) of each consecutive
                window of the recording.
        y_true: analogous to y_pred, the array contains the label of each segment (0 or 1)
    '''

    K.set_image_data_format('channels_last')

    model.load_weights(model_weights_path)

    y_aux = []
    for j in range(len(generator)):
        _, y = generator[j]
        y_aux.append(y)
    true_labels = np.vstack(y_aux)

    prediction = model.predict(generator, verbose=0)
    
    # For 3-class, return full probability matrix; for 2-class, return binary probabilities
    num_classes = prediction.shape[1] if len(prediction.shape) > 1 else 2
    
    if num_classes == 3:
        # 3-class: return full probability matrix and true class indices
        y_pred = prediction  # Shape: (n_samples, 3)
        y_true = np.argmax(true_labels, axis=1)  # Convert one-hot to class indices
    else:
        # 2-class: return probability of class 1 (backward compatibility)
        y_pred = np.empty(len(prediction), dtype='float32')
        for j in range(len(y_pred)):
            y_pred[j] = prediction[j][1]

        y_true = np.empty(len(true_labels), dtype='uint8')
        for j in range(len(y_true)):
            y_true[j] = true_labels[j][1]

    return y_pred, y_true
