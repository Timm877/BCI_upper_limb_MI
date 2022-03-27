from pathlib import Path
import os
import scipy.io
# INIT
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import moabb
from moabb.datasets import PhysionetMI, Schirrmeister2017, Weibo2014
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
from sklearn.pipeline import Pipeline
import src.unicorn_utils as utils
import src.utils_deep as utils_deep
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix

dataset = Weibo2014()
data = dataset.get_data(subjects=[1])
subject, session, run = 1, "session_0", "run_0"
raw = data[subject][session][run]
#fig = raw.plot_sensors(show_names=True, block=True)
#dataset.subject_list = [3]
n_classes = 3
paradigm = MotorImagery(fmin = 8, fmax = 35,events=['left_hand', 'rest', 'right_hand'], n_classes=n_classes, tmin = 0.0,
 tmax = 2.495,channels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'])
X, labels, meta = paradigm.get_data(dataset=dataset)

print(np.unique(labels))
print("The EEG is stored a (n_trials x n_channels x n_samples) array: {}".format(X.shape))
print(X.dtype)
print("The labels are also stored as an array: {} ...".format(labels[:5]))
print()
print("The metadata are store in a DataFrame:")
print(meta.head(10))

labels = np.where(labels == 'rest', 0, labels)
labels = np.where(labels == 'right_hand', 1, labels)
labels = np.where(labels == 'left_hand', 2, labels)
labels = labels.astype(np.float)

print(np.unique(labels))
print(np.count_nonzero(labels == 2))
print(np.count_nonzero(labels == 1))
print(np.count_nonzero(labels == 0))

from imblearn.under_sampling import RandomUnderSampler
# define undersample strategy
undersample = RandomUnderSampler()
undersample.fit_resample(X[:,:,0], labels)
X = X[undersample.sample_indices_]
labels = labels[undersample.sample_indices_]

print(X.shape)
print(np.count_nonzero(labels == 2))
print(np.count_nonzero(labels == 1))
print(np.count_nonzero(labels == 0))

#TODO add the 2 channel split here
channels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
pairs = [['C3', 'C4'], ['PO7', 'PO8'], ['Cz', 'Oz'], ['Fz', 'Pz']]
X1 = X[:,[1,3],:]
#X2 = X[:,[5,7],:]
X3 = X[:,[2,4],:]
#X4 = X[:,[0,6],:]
X_full = np.concatenate((X1, X3))
y_full = np.concatenate((labels, labels))# labels, labels))
print(X_full.shape)
print(y_full.shape)
oneDCNN = True
if oneDCNN:
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.1, random_state=42, shuffle=True)
else:
    #split EEG and labels into train val set randomly
    X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42, shuffle=True)

# in good shape
# then apply deep pipeline
trainloader, valloader = utils_deep.data_setup(X_train, y_train, X_val, y_val) 
lr = 0.0001
receptive_field = 50 # chosen by experimentation (see deeplearn_experiment folder) 
# In paper1, they also use 65, but also collect more samples (3seconds)
filter_sizing = 40 # Chosen by experimentation. Depth of conv layers; 40 was used in appers
mean_pool = 15 # Chosen by experimentation. 15 was used in papers
window_size = 500
n_electrodes = 8

train_accuracy, val_accuracy, train_f1, val_f1, train_classacc, val_classacc, \
training_precision, training_recall, validation_precision, validation_recall, validation_roc_auc = \
    utils_deep.run_model(trainloader, valloader, lr, window_size, n_electrodes, receptive_field, 
    filter_sizing, mean_pool, num_classes = n_classes)

print(f"Classification accuracy validation set: {np.around(np.array(val_accuracy).mean(),3)}, prec: {np.around(np.array(validation_precision).mean(),3)}\
    recall: {np.around(np.array(validation_recall).mean(),3)}, f1: {np.around(np.array(val_f1).mean(),3)}")

# riemannian pipeline for check
pipe = Pipeline(steps=[('cov', Covariances("oas")), 
                                            ('tg', TangentSpace(metric="riemann")),
                                            ('svm', SVC(decision_function_shape='ovo'))])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_val)
acc = np.mean(preds == y_val)
print(f"Classification accuracy: {acc} and per class:")
print(confusion_matrix(y_val, preds))

'''
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

features = X_train.reshape(
                    (X_train.shape[0], X_train.shape[2],
                    X_train.shape[1], 1)
                )

val_features = X_val.reshape(
    (X_val.shape[0], X_val.shape[2],
    X_val.shape[1], 1)
)

# One hot encoding of labels
def one_hot_labels(caategorical_labels):
    enc = OneHotEncoder(handle_unknown='ignore')
    on_hot_labels = enc.fit_transform(
        caategorical_labels.reshape(-1, 1)).toarray()
    return on_hot_labels

train_erp_labels = one_hot_labels(y_train)
val_erp_labels = one_hot_labels(y_val)

#  TRAINING
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
train = True

n_cha = 8
filters_per_branch=8
scales_time = (500, 250, 125)
dropout_rate = 0.25
activation = 'elu'
learning_rate = 0.001
if train:
    # Create model
    model = EEGInception(
        input_time=500, fs=200, ncha=n_cha, filters_per_branch=filters_per_branch,
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
    print("test loss, test acc:", DL_result)
'''