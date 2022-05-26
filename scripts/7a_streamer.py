from pylsl import StreamInfo, StreamOutlet
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pickle

class SignalsOutlet:
    def __init__(self, trans_type,
                 fs, channels, name='NFBLab_data1'):
        self.info = StreamInfo(name=name, type= trans_type, channel_count= channels, source_id='nfblab42',
                               nominal_srate=fs)
        self.info.desc().append_child_value("manufacturer", "BioSemi")
        #channels = self.info.desc().append_child("channels")
        #for c in signals:
        #    channels.append_child("channel").append_child_value("name", c)
        self.outlet = StreamOutlet(self.info)

    def push_sample(self, data):
        self.outlet.push_sample(data)

    def push_repeated_chunk(self, data, n=1):
        #chunk = repeat(data, n).reshape(-1, n).T.tolist()
        #self.outlet.push_chunk(chunk)
        for k in range(n):
            self.outlet.push_sample(data)

    def push_chunk(self, data, n=1):
        self.outlet.push_chunk(data)

sOut1 = SignalsOutlet('EEG1',250,17) 
# load data into dict entry
sampling_frequency = 250 
# testing here for 8 electrodes:
electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8','AccX','AccY','AccZ','Gyro1','Gyro2','Gyro3',
                                  'Battery','Counter','Validation']
dataset_full = {}
trials_amount = 0

subjects = ['X01']#,'X02','X03','X04','X05','X06','X07','X08','X09']
#subjects = ['X02'] #dry subjects
for subject in subjects :
    folder_path = Path(f'./data/openloop/{subject}/openloop')
    #folder_path = Path(f'./data/dry/{subject}/openloop')
    for instance in os.scandir(folder_path):
        if instance.path.endswith('.csv'): 
            trials_amount +=1
            #print(f'adding_{instance} to dataset...')
            sig = pd.read_csv(instance.path)
            X = sig.loc[:,electrode_names]
            y = sig.loc[:,'Class']
            dataset_full[str(instance)] = pd.concat([X], axis=1)
    print(f'{subject} data loaded to dataset & total length - {len(dataset_full)}')

sub2stream = 'X01'
for dir_entry in list(dataset_full.keys()):
    if sub2stream in dir_entry:
        for i in range(0, dataset_full[dir_entry].shape[0]):
            sOut1.push_sample(list(dataset_full[dir_entry].iloc[i]))
            print(i)
            if i == dataset_full[dir_entry].shape[0]:
                print('stream done')