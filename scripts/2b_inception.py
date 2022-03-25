import argparse
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
import random

pd.options.mode.chained_assignment = None  # default='warn'

input_time = 120
fs = 250
n_cha = 8
filters_per_branch = 8
scales_time = (500, 250, 125)
dropout_rate = 0.25
activation = 'elu'
learning_rate = 0.001

def execution(subject, type):
    # INIT
    sampling_frequency = 250 
    # testing here for 8 electrodes:
    electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    n_electrodes = len(electrode_names)

    if type == 'arms':
        folder_path = Path(f'./data/pilots/intermediate_datafiles/preprocess/{subject}_leftA_rightA')
        result_path = Path(f'./results/intermediate_datafiles/pilots/{subject}_leftA_rightA')
        result_path.mkdir(exist_ok=True, parents=True)
        results_fname = f'inception_UL_{type}.csv'
        n_classes = 2
    elif type == 'multiclass':
        folder_path = Path(f'./data/pilots/intermediate_datafiles/preprocess/{subject}')
        result_path = Path(f'./results/intermediate_datafiles/pilots/{subject}')
        result_path.mkdir(exist_ok=True, parents=True)
        results_fname = f'inception_UL_{type}.csv'
        n_classes = 3
    results = {}

    for instance in os.scandir(folder_path):
        if 'deep' in instance.path: 
            all_results = []
            print(f'Running for {instance.path}...')
            a_file = open(instance.path, "rb")
            data_dict = pickle.load(a_file)
            X = data_dict['data']
            y = data_dict['labels']
            window_size = int(instance.path.split("ws",1)[1][:3]) 
            train_acc_cv, val_acc_cv, train_f1_cv, val_f1_cv, acc_classes_cv = [], [], [], [], []
            for cross_val in range(list(X)[-1] + 1): #+1
                X_train, y_train, X_val, y_val = [], [], [], []
                seg_list = [0,1,2,3,4,5,6,7,8,9]
                random.shuffle(seg_list)
                print(seg_list)
                for df in seg_list: 
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
                                        batch_size=32,
                                        validation_split=0.2,
                                        callbacks=[early_stopping])
                    # Save
                    DL_result = model.evaluate(val_features, val_erp_labels, batch_size=256)
                    all_results.append(DL_result[1])
                    print("test loss, test acc:", DL_result)
            results[f"crossval_{instance.path}"] = {'mean_test_acc': np.array(all_results).mean(),'all_accs': all_results} 
    results_df = pd.DataFrame.from_dict(results, orient='index').sort_values('mean_test_acc', ascending=False)  
    results_df.to_csv(result_path / results_fname)
    print('Finished')

def main():
    for subj in FLAGS.subjects:
        print(f'Running for subject {subj}')
        if 'multiclass' in FLAGS.type:
            print(f'For upper limb multiclass')
            execution(subj, 'multiclass')
        elif 'arms' in FLAGS.type:
            print(f'For binary arms')
            execution(subj, 'arms')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--type", nargs='+', default=['multiclass'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'multiclass', 'arms'")
    parser.add_argument("--subjects", nargs='+', default=['X02_wet'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are in the data folder.")
    FLAGS, unparsed = parser.parse_known_args()
    main()

'''
else: 
    # here we first re-train model on the new session before eval on validation set
    reconstructed_model = tensorflow.keras.models.load_model("inception_model_7static")
    reconstructed_model.fit(features, train_erp_labels, epochs=1, batch_size=256, validation_split=0.2)
    results = reconstructed_model.evaluate(val_features, val_erp_labels, batch_size=256)
    print("test loss, test acc:", results)
'''
