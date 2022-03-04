#%% IMPORT LIBRARIES
import numpy as np
import h5py, os
from src.EEG_Inception import EEGInception
import src.utils as utils
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import time
import pandas as pd
import scipy.io
from scipy import signal
import tensorflow.keras

pd.options.mode.chained_assignment = None  # default='warn'

input_time = 120
fs = 200
n_cha = 8
filters_per_branch = 8
scales_time = (400, 200, 100)
dropout_rate = 0.25
activation = 'elu'
n_classes = 2
learning_rate = 0.001

#%% LOAD DATASET
all_electrode_names = ['F3', 'FZ', 'FC1','FCZ','C1','CZ','CP1','CPZ', 
        'FC5', 'FC3','C5','C3','CP5','CP3','P3','PZ','F4','FC2','FC4','FC6','C2',
        'C4','CP2','CP4','C6','CP6','P4','HR' ,'HL', 'VU', 'VD']
selected_electrodes_names = ['FZ', 'C3', 'CZ', 'C4', 'CPZ', 'P3', 'PZ', 'P4'] 
n_electrodes = len(selected_electrodes_names)
folder_path = Path('./data/offline/20210519_MI_atencion_online_static_online/')
subject = '20210519_MI_atencion_online_static_online'   
result_path = Path('./data/offline/intermediate_datafiles/inception/')
result_path.mkdir(exist_ok=True, parents=True)
results_fname = f'inception_{time.time()}_experiments_{subject}.csv'

dataset_full = {}
trials_amount = 0
for instance in os.scandir(folder_path):
    if instance.path.endswith('.mat'): 
        trials_amount +=1
        print(f'adding_{instance} to dataset...')
        sig = scipy.io.loadmat(instance.path)
        X = sig['session']['data_EEG'][0][0][:-1,:].T
        y = sig['session']['task_EEG'][0][0].T 
        dataset = pd.DataFrame(X, columns=all_electrode_names)
        labels = pd.DataFrame(y, columns=['label'])
        dataset = dataset[selected_electrodes_names]    
        data_relax = dataset.loc[labels['label'] == 402] 
        data_MI = dataset.loc[labels['label'] == 404]
        data_relax['label'] = 0
        data_MI['label'] = 1
        dataset_full[str(instance)] = pd.concat([data_relax,data_MI], axis=0)
        dataset_full[str(instance)].reset_index(drop=True, inplace=True)

results = {}
sample_duration = 400
list_of_freq_lim = [[[4,35]]]
freq_limits_names_list = [['4-35Hz']]

freq_limits = np.asarray(list_of_freq_lim[0]) 
freq_limits_names = freq_limits_names_list[0]
filter_order = 2
sampling_frequency = 200
filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filter_order)
X_train = []
y_train = []  
X_val = []
y_val = []
df_num = 0
print(f'experimenting with filter order of {filter_order}, freq limits of {freq_limits_names}, \
and ws of {sample_duration}.')
for df in dataset_full:
    X_temp, y_temp = utils.segmentation_overlap_withfilt(dataset_full[df], sample_duration, filters,
    selected_electrodes_names, freq_limits_names, 'deep', window_hop=100)
    for segment in range(len(X_temp)): 
        # add to x_all without initiating lists into lists
        if df_num ==4: 
            X_val.append(X_temp[segment])
            y_val.append(y_temp[segment]) 
        else:
            X_train.append(X_temp[segment])
            y_train.append(y_temp[segment]) 
    df_num += 1
    print(f'Current length of X train: {len(X_train)}.')
    print(f'Current length of X val: {len(X_val)}.')

# transform to np for use in ML-pipeline
X_train_np = np.stack(X_train)
X_val_np = np.stack(X_val)
y_train_np = np.array(y_train)
y_val_np = np.array(y_val)
print(f"shape training set: {X_train_np.shape}")
print(f"shape validation set: {X_val_np.shape}")


#%% PREPARE FEATURES AND LABELS
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

#%%  TRAINING
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
train = False
if train:
    # Create model
    model = EEGInception(
        input_time=400, fs=fs, ncha=n_cha, filters_per_branch=filters_per_branch,
        scales_time=scales_time, dropout_rate=dropout_rate,
        activation=activation, n_classes=n_classes, learning_rate=learning_rate)

    # Print model summary
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0.0001,
        mode='min', patience=10, verbose=1,
        restore_best_weights=True)

    # Fit model
    fit_hist = model.fit(features,
                        train_erp_labels,
                        epochs=250,
                        batch_size=256,
                        validation_split=0.2,
                        callbacks=[early_stopping])
    # Save
    model.save('inception_model_7static')
else: 
    # here we first re-train model on the new session before eval on validation set
    reconstructed_model = tensorflow.keras.models.load_model("inception_model_7static")
    reconstructed_model.fit(features, train_erp_labels, epochs=1, batch_size=256, validation_split=0.2)
    results = reconstructed_model.evaluate(val_features, val_erp_labels, batch_size=256)
    print("test loss, test acc:", results)
