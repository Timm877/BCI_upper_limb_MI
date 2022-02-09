import os
from pathlib import Path

import matplotlib
import src.utils as utils
import argparse
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import KFold, cross_val_score
import mne

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
     # 1. get files
     folder_path = Path('./data/offline/')
     result_path = Path('./data/offline/intermediate_datafiles/1/')
     result_path.mkdir(exist_ok=True, parents=True)
     
     for instance in os.scandir(folder_path): # go through all data files  
          # when having only 1 file this is not needed
          if instance.path.endswith('.mat'):
               #TODO add multiple sessions
               sig = scipy.io.loadmat(instance.path)
               X = sig['session']['data_EEG']
               y = sig['session']['task_EEG']

               X = X[0][0][:-1,:].T #get relevant data and transpose
               y = y[0][0].T #get relevant data and transpose

     # 2. INIT
     sampling_frequency = 200 # 250 for ours, 200 for Laura's
     sample_duration = 200 # 1 second
     all_electrode_names = ['F3', 'FZ', 'FC1','FCZ','C1','CZ','CP1','CPZ', 
     'FC5', 'FC3','C5','C3','CP5','CP3','P3','PZ','F4','FC2','FC4','FC6','C2',
     'C4','CP2','CP4','C6','CP6','P4','HR' ,'HL', 'VU', 'VD']
     deleted_electrodes_names = ['F3','C5', 'F4','C2']
     selected_electrodes_names = [x for x in all_electrode_names if x not in deleted_electrodes_names]
     # for us: ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
     n_electrodes = len(selected_electrodes_names)
     # transform np array to pandas dataframe
     dataset = pd.DataFrame(X, columns=all_electrode_names)
     # IMPORTANT: DELETE! F3, C5, F4, C2 --> only used for eye artifact correction!
     dataset.drop(deleted_electrodes_names, axis=1, inplace=True)


     labels = pd.DataFrame(y, columns=['label'])
     data_relax = dataset.loc[labels['label'] == 402] 
     # TODO relax is made of 2 segments --> dumb concat may lead to weird transition? for now, don't care as it's only influencing 1 segment
     data_MI = dataset.loc[labels['label'] == 404]


     # init filters 
     # TODO add more filter experimentation
     # order of 2, 3, 4
     # freq limits: big - [[5, 15], [15, 25], [25, 35]],          normal [[5, 10], [10, 15], [15, 20], [20, 25]], 
     #  low [[0.5, 5], [5, 10], [10, 15]],         high  [[5, 10], [10, 15], [15, 20], [20, 25], [25, 30], [30, 35]], 

     freq_limits = np.asarray([[5, 10], [10, 15], [15, 20], [20, 25]]) 
     freq_limits_names = ['5_10Hz', '10_15Hz','15_20Hz','20_25Hz']
     filter_order = 2
     filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filter_order)

     # 3. split data into segments, apply filter bank (FB), set up for pipeline 
     filtered_dataset_relax = utils.segmentation_and_filter(data_relax, selected_electrodes_names,filters, sample_duration, freq_limits_names)
     filtered_dataset_MI = utils.segmentation_and_filter(data_MI, selected_electrodes_names, filters, sample_duration, freq_limits_names)
     filtered_dataset_relax['label'] = 1
     filtered_dataset_MI['label'] = 2
     filtered_dataset_full = pd.concat([filtered_dataset_relax,filtered_dataset_MI], axis=0)
     filtered_dataset_full.reset_index(drop=True, inplace=True)

     X_segmented, y = utils.segmentation_for_ML(filtered_dataset_full,sample_duration)
     # we have 70 trials of size 108channelsx200samples, with 70 corresponding labels

     # 4. init pipeline by parser args, and apply to data.
     chosen_pipelines = utils.init_pipelines(FLAGS.p)
     cv = KFold(n_splits=3, shuffle=True, random_state=42)
     results = {}

     # TODO: add all results to pd dataframe: save results as dicts of dict --> categorize on f1 score
     results = {}
     scoring = ['accuracy', 'f1_score', 'precision', 'recall']
     for clf in chosen_pipelines:
          start_time = time.time()
          accuracy = cross_val_score(chosen_pipelines[clf], X_segmented, y, cv=cv, n_jobs=-1).mean()
          #TODO add f1 p r
          elapsed_time = time.time() - start_time
          results[clf] = {'filter_order': filter_order, 'filter_limits': freq_limits_names,
                'accuracy': accuracy, 'time (seconds)': elapsed_time,  }   
     print(results) #and save

if __name__ == '__main__':
     parser = argparse.ArgumentParser(description="Run offline BCI analysis")
     parser.add_argument("--p", type=list, default=['csp+lda'], help="The pipeline used. \
     This variable is a list containing all names of selected pipelines. Options currently are: ")

     FLAGS, unparsed = parser.parse_known_args()
     main()
