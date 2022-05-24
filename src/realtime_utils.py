import copy
import time
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from mne.decoding import CSP
from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy import signal, stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def init_filters(freq_lim, sample_freq, filt_type = 'bandpass', order=2, state_space=True):
    filters = []
    for f in range(freq_lim.shape[0]):
        A, B, C, D, Xnn  = init_filt_coef_statespace(freq_lim[f], fs=sample_freq, filtype=filt_type, order=order)
        filters.append([A, B, C, D, Xnn ])
    return filters

def init_filt_coef_statespace(cuttoff, fs, filtype, order,len_selected_electrodes=8):
    if filtype == 'lowpass':
        b,a = signal.butter(order, cuttoff[0]/(fs/2), filtype)
    elif filtype == 'bandpass':
        b,a = signal.butter(order, [cuttoff[0]/(fs/2), cuttoff[1]/(fs/2)], filtype)
    elif filtype == 'highpass':
        b,a = signal.butter(order, cuttoff[0]/(fs/2), filtype)
    # getting matrices for the state-space form of the filter from scipy and init state vector
    A, B, C, D = signal.tf2ss(b,a)
    Xnn = np.zeros((1,len_selected_electrodes,A.shape[0],1))
    return A, B, C, D, Xnn

def apply_filter_statespace(sig, A, B, C, D, Xnn):
    # State space with scipy's matrices
    filt_sig = np.array([])
    for sample in sig: 
        filt_sig = np.append(filt_sig, C@Xnn + D * sample)
        Xnn = A@Xnn + B * sample
    return filt_sig, Xnn

def pre_processing(curr_segment, selected_electrodes_names, filters, sample_duration, freq_limits_names, 
                    sampling_frequency):
    outlier = 0
    # 1. notch filt
    b_notch, a_notch = signal.iirnotch(50, 30, sampling_frequency)
    for column in curr_segment.columns:
        curr_segment.loc[:,column] = signal.filtfilt(b_notch, a_notch, curr_segment.loc[:,column])
    curr_segment = curr_segment.T

    # 2 OUTLIER DETECTION
    for i, j in curr_segment.iterrows():
        if stats.kurtosis(j) > 4*np.std(j) or (abs(j - np.mean(j)) > 125).any():
            if stats.kurtosis(j) > 4*np.std(j):
                print('due to kurtosis')
            outlier +=1
    
    # 3 APPLY COMMON AVERAGE REFERENCE (CAR) per segment only for deep learning pipeline   
    curr_segment -= curr_segment.mean()

    # 4 FILTERING filter bank / bandpass
    segment_filt, filters = filter_1seg_statespace(curr_segment, selected_electrodes_names, filters, sample_duration, 
    freq_limits_names)

    return segment_filt, outlier, filters

def filter_1seg_statespace(segment, selected_electrodes_names,filters, sample_duration, freq_limits_names):
    # filters dataframe with 1 segment of 1 sec for all given filters
    # returns a dataframe with columns as electrode-filters
    filter_results = {}
    segment = segment.transpose()
    for electrode in range(len(selected_electrodes_names)):
        for f in range(len(filters)):
            A, B, C, D, Xnn = filters[f] 
            filter_results[selected_electrodes_names[electrode] + '_' + freq_limits_names[f]] = []
            if segment.shape[0] == sample_duration:      
                # apply filter ánd update Xnn state vector       
                filt_result_temp, Xnn[0,electrode] = apply_filter_statespace(segment[selected_electrodes_names[electrode]], 
                A, B, C, D, Xnn[0,electrode])         
                for data_point in filt_result_temp:
                    filter_results[selected_electrodes_names[electrode] + '_' + freq_limits_names[f]].append(data_point) 
            filters[f] = [A, B, C, D, Xnn]
    filtered_dataset = pd.DataFrame.from_dict(filter_results).transpose()    
    return filtered_dataset, filters

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class EEGNET(nn.Module):
    def __init__(self):
        super(EEGNET,self).__init__()
        sample_duration = 500
        channel_amount = 8
        num_classes = 3
        receptive_field = 64
        filter_sizing = 8
        dropout = 0.25
        D = 2

        self.temporal=nn.Sequential(
            nn.Conv2d(1,filter_sizing,kernel_size=[1,receptive_field],stride=1, bias=False,\
                padding='same'), 
            nn.BatchNorm2d(filter_sizing),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing*D,kernel_size=[channel_amount,1],bias=False,\
                groups=filter_sizing),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )

        self.seperable=nn.Sequential(
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,16],\
                padding='same',groups=filter_sizing*D, bias=False),
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,1], padding='same',groups=1, bias=False),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )
        self.avgpool1 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)   
        self.avgpool2 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)
        self.dropout = nn.Dropout(dropout)
        self.view = nn.Sequential(Flatten())

        endsize = 320
        self.fc2 = nn.Linear(endsize, num_classes)


    def forward(self,x):
        out = self.temporal(x)
        out = self.spatial(out)
        out = self.avgpool1(out)
        out = self.dropout(out)
        out = self.seperable(out)
        out = self.avgpool2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        prediction = self.fc2(out)
        return prediction





