# IMPORT LIBRARIES
import numpy as np
import h5py, os
from src.EEG_Inception import EEGInception
import src.unicorn_utils as utils
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import time
import pandas as pd
import scipy.io
from scipy import signal
import tensorflow.keras
import pickle

pd.options.mode.chained_assignment = None  # default='warn'

input_time = 120
fs = 250
n_cha = 8
filters_per_branch = 8
scales_time = (500, 250, 125)
dropout_rate = 0.25
activation = 'elu'
n_classes = 3
learning_rate = 0.001

#%% LOAD DATASET
# INIT
sampling_frequency = 250 
subject = 'X02_wet'
# testing here for 8 electrodes:
electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
n_electrodes = len(electrode_names)
folder_path = Path(f'./data/pilots/intermediate_datafiles/preprocess/{subject}')
result_path = Path(f'./results/intermediate_datafiles/pilots/{subject}')
result_path.mkdir(exist_ok=True, parents=True)
results_fname = f'inception_UL_multiclass.csv'
num_classes = 3
results = {}

for instance in os.scandir(folder_path):
    if 'deep' in instance.path: 
        print(f'Running for {instance.path}...')
        a_file = open(instance.path, "rb")
        data_dict = pickle.load(a_file)
        X = data_dict['data']
        y = data_dict['labels']
        window_size = int(instance.path.split("ws",1)[1][:3]) 
        train_acc_cv, val_acc_cv, train_f1_cv, val_f1_cv, acc_classes_cv = [], [], [], [], []
        for cross_val in range(list(X)[-1] + 1):
            X_train, y_train, X_val, y_val = [], [], [], []
            for df in X: 
                for segment in range(len(X[df])): 
                    if df == cross_val:
                        X_val.append(X[df][segment])
                        y_val.append(y[df][segment]) 
                    else:
                        X_train.append(X[df][segment])
                        y_train.append(y[df][segment]) 
                print(f'Current length of X train: {len(X_train)}.')
                print(f'Current length of X val: {len(X_val)}.')
            X_train_np = np.stack(X_train)
            X_val_np = np.stack(X_val)
            y_train_np = np.array(y_train)
            y_val_np = np.array(y_val)
            print(f"shape training set: {X_train_np.shape}")
            print(f"shape validation set: {X_val_np.shape}")

            # PREPARE FEATURES AND LABELS
            # Reshape epochs for EEG-Inception
            # [nsamples x n_channels]
            # X_np for me is [samplesxchannelsxtime]
            features = X_train_np.reshape(
                (X_train_np.shape[0], X_train_np.shape[2],
                X_train_np.shape[1], 1)
            )

            val_features = X_val_np.reshape(
                (X_val_np.shape[0], X_val_np.shape[2],
                X_val_np.shape[1], 1)
            )
            
            # One hot encoding of labels
            def one_hot_labels(caategorical_labels):
                enc = OneHotEncoder(handle_unknown='ignore')
                on_hot_labels = enc.fit_transform(
                    caategorical_labels.reshape(-1, 1)).toarray()
                return on_hot_labels

            train_erp_labels = one_hot_labels(y_train_np)
            val_erp_labels = one_hot_labels(y_val_np)

            #  TRAINING
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            train = True
            if train:
                # Create model
                model = EEGInception(
                    input_time=window_size, fs=fs, ncha=n_cha, filters_per_branch=filters_per_branch,
                    scales_time=scales_time, dropout_rate=dropout_rate,
                    activation=activation, n_classes=n_classes, learning_rate=learning_rate)

                # Print model summary
                #model.summary()

                # Callbacks
                early_stopping = EarlyStopping(
                    monitor='val_loss', min_delta=0.0001,
                    mode='min', patience=10, verbose=1,
                    restore_best_weights=True)

                # Fit model
                fit_hist = model.fit(features,
                                    train_erp_labels,
                                    epochs=250,
                                    batch_size=64,
                                    validation_split=0.2,
                                    callbacks=[early_stopping])
                # Save
                DL_result = model.evaluate(val_features, val_erp_labels, batch_size=256)
                print("test loss, test acc:", DL_result)
                results[f"crossval_{instance.path}"] = {'test_acc': DL_result[1]} 
results_df = pd.DataFrame.from_dict(results, orient='index').sort_values('test_acc', ascending=False)  
results_df.to_csv(result_path / results_fname)
print('Finished')

'''
else: 
    # here we first re-train model on the new session before eval on validation set
    reconstructed_model = tensorflow.keras.models.load_model("inception_model_7static")
    reconstructed_model.fit(features, train_erp_labels, epochs=1, batch_size=256, validation_split=0.2)
    results = reconstructed_model.evaluate(val_features, val_erp_labels, batch_size=256)
    print("test loss, test acc:", results)
'''
