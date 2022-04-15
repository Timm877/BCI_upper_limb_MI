import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import src.unicorn_utils as utils
from scipy import signal, stats
pd.options.mode.chained_assignment = None  # default='warn'

sampling_frequency = 250 
# testing here for 8 electrodes:
electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
n_electrodes = len(electrode_names)
folder_path = Path(f'./data/openloop/intermediate_datafiles/preprocess/X07')
result_path = Path(f'./results/intermediate_datafiles/openloop/X07')
result_path.mkdir(exist_ok=True, parents=True)

for instance in os.scandir(folder_path):
    if 'riemann' in instance.path: 
        print(f'Running for {instance.path}...')
        a_file = open(instance.path, "rb")
        data_dict = pickle.load(a_file)
        X = data_dict['data']
        y = data_dict['labels']
        window_size = int(instance.path.split("ws",1)[1][:3])
        for df in X: 
            segments=0
            for segment in range(len(X[df])):
                if segments == 0:
                    data = pd.DataFrame(X[df][segment].T)
                    if y[df][segment] == 0:
                        label = pd.DataFrame({'label_relax': pd.Series(1 for i in range(500)),
                        'label_right_arm': pd.Series(0 for i in range(500)),
                        'label_left_arm': pd.Series(0 for i in range(500))})
                    elif y[df][segment] == 1:
                        label = pd.DataFrame({'label_relax': pd.Series(0 for i in range(500)),
                        'label_right_arm': pd.Series(1 for i in range(500)),
                        'label_left_arm': pd.Series(0 for i in range(500))})
                    elif y[df][segment] == 2:
                        label = pd.DataFrame({'label_relax': pd.Series(0 for i in range(500)),
                        'label_right_arm': pd.Series(0 for i in range(500)),
                        'label_left_arm': pd.Series(1 for i in range(500))})

                    data = pd.concat([data, label], axis=1)
                    data.columns = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8', 'label_relax', 
                    'label_right_arm','label_left_arm']
                elif segments % 5 == 0:
                    data2 = pd.DataFrame(X[df][segment].T)

                    if y[df][segment] == 0:
                        label = pd.DataFrame({'label_relax': pd.Series(1 for i in range(500)),
                        'label_right_arm': pd.Series(0 for i in range(500)),
                        'label_left_arm': pd.Series(0 for i in range(500))})
                    elif y[df][segment] == 1:
                        label = pd.DataFrame({'label_relax': pd.Series(0 for i in range(500)),
                        'label_right_arm': pd.Series(1 for i in range(500)),
                        'label_left_arm': pd.Series(0 for i in range(500))})
                    elif y[df][segment] == 2:
                        label = pd.DataFrame({'label_relax': pd.Series(0 for i in range(500)),
                        'label_right_arm': pd.Series(0 for i in range(500)),
                        'label_left_arm': pd.Series(1 for i in range(500))})
                    
                    #label = label.replace({0: 'relax', 1: 'right arm', 2: 'left arm'})
                    data2 = pd.concat([data2, label], axis=1)
                    data2.columns = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8', 'label_relax', 
                    'label_right_arm','label_left_arm']
                    
                    data = pd.concat([data, data2], ignore_index=True)
                segments+=1
            utils.plot_dataset(data, ['CZ', 'C3', 'C4', 'label_'],
                                  ['like', 'like', 'like', 'like'],
                                  ['line','line', 'line', 'points'])