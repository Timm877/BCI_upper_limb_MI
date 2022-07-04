import numpy as np
import pandas as pd
from scipy import signal, stats
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# preprocessing functions
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

# neural net functions
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

# closed loop functions
def movedot(prediction,focus,pos,size):
    if prediction ==0 :
        size = (size[0]+0.01,size[1]+0.01)
        focus.setSize(size)
    elif prediction ==1 :
        pos = (pos[0]+0.05,pos[1])
        focus.setPos(pos)
    elif prediction ==2:
        pos = (pos[0]-0.05,pos[1])
        focus.setPos(pos)
    else :
        focus.setPos(pos)
        focus.setSize(size)
        
    return focus, pos, size

def movedotwhen(prediction,focus,pos,size,cue):
    if cue == 1:
        if prediction ==1 : # right
            pos = (pos[0]+0.01,pos[1])
            focus.setPos(pos)
        else :
            focus.setPos(pos)
            focus.setSize(size)
    elif cue == 2:
        if prediction ==2: # left
            pos = (pos[0]-0.01,pos[1])
            focus.setPos(pos)
        else :
            focus.setPos(pos)
            focus.setSize(size)
    elif cue == 0:
        if prediction ==0 : # relax
            size = (size[0]+0.01,size[1]+0.01)
            focus.setSize(size)
        else :
            focus.setPos(pos)
            focus.setSize(size)
        
    return focus, pos, size

import random
def Genrandom(num):
    randomlist = []
    for i in range(0,num):
        n = random.randint(0,3)
        randomlist.append(n)
    print(randomlist)
    return randomlist

def concatdata(current_seg, segment_filt):
    if current_seg.shape[1] == 0:
        current_seg = segment_filt
        
    elif current_seg.shape[1] == 500:
        current_seg = pd.concat([current_seg.iloc[: , 125:].reset_index(drop=True), segment_filt.reset_index(drop=True)],
                    axis=1, ignore_index=True)        
    else:
        current_seg = pd.concat([current_seg.reset_index(drop=True), segment_filt.reset_index(drop=True)],
                    axis=1, ignore_index=True)
    return current_seg

#updating database with captured data from EEG
def update_data(data,res):
    i = 0
    for key in list(data.keys()):
        data[key].append(res[i])
        i = i +1
    return data

def is_MI_segment(labels):
    #label_count = labels.shape[0]
    label_num = labels.label.unique()[0]
    if label_num in [0,1,2]:
        return True, label_num
    else:
        return False, 'noMI'

def do_prediction(current_seg, net):
    current_tensor = torch.from_numpy(np.array(current_seg))
    current_tensor = current_tensor[np.newaxis, np.newaxis, :, :]
    output = net(current_tensor.float())
    _, predicted = torch.max(output.data, 1)
    return predicted

def segment_dict(initial, final, hop, data):
    electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    temp_dict = dict((k, []) for k in electrode_names)
    #gives out 125 length dicts
    for channel in electrode_names :
        temp_dict[channel] = data[channel][initial:final]
    initial = initial + hop
    final = final + hop
    df = pd.DataFrame.from_dict(temp_dict)
    return df, initial, final