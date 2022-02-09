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

from sklearn.model_selection import KFold, cross_validate
import mne

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
    # INIT
    sampling_frequency = 200 # 250 for ours, 200 for Laura's
    sample_duration = 200 # 1 second
    all_electrode_names = ['F3', 'FZ', 'FC1','FCZ','C1','CZ','CP1','CPZ', 
    'FC5', 'FC3','C5','C3','CP5','CP3','P3','PZ','F4','FC2','FC4','FC6','C2',
    'C4','CP2','CP4','C6','CP6','P4','HR' ,'HL', 'VU', 'VD']
    deleted_electrodes_names = ['F3','C5', 'F4','C2']
    selected_electrodes_names = [x for x in all_electrode_names if x not in deleted_electrodes_names]
    # for us: ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    n_electrodes = len(selected_electrodes_names)


    folder_path = Path('./data/offline/202100722_MI_atencion_online/')
    result_path = Path('./data/offline/intermediate_datafiles/1/')
    result_path.mkdir(exist_ok=True, parents=True)

    X_all = []
    y_all = []
    
    for instance in os.scandir(folder_path): # go through all data files  
        # when having only 1 file this is not needed
        if instance.path.endswith('.mat'):
            #add multiple sessions --> segment them individually, then add the segments together
            sig = scipy.io.loadmat(instance.path)
            X = sig['session']['data_EEG']
            y = sig['session']['task_EEG']

            X = X[0][0][:-1,:].T #get relevant data and transpose
            y = y[0][0].T #get relevant data and transpose

            dataset = pd.DataFrame(X, columns=all_electrode_names)
            # IMPORTANT: DELETE! F3, C5, F4, C2 --> only used for eye artifact correction!
            dataset.drop(deleted_electrodes_names, axis=1, inplace=True)
            labels = pd.DataFrame(y, columns=['label'])

            data_relax = dataset.loc[labels['label'] == 402] 
            data_MI = dataset.loc[labels['label'] == 404]

            #--------
            # below filtering part --> maybe later add this to pipeline function?
            freq_limits = np.asarray([[5, 10], [10, 15], [15, 20], [20, 25]]) 
            freq_limits_names = ['5_10Hz', '10_15Hz','15_20Hz','20_25Hz']
            filter_order = 2
            filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filter_order)
            #-----------

            data_relax.loc['label'] = 0
            data_MI['label'] = 1

            dataset_full = pd.concat([data_relax,data_MI], axis=0)
            dataset_full.reset_index(drop=True, inplace=True)
            print(dataset_full.shape) #TODO why 14001 and not 14000
            X_segmented, y = utils.segmentation_all(dataset_full,sample_duration)
            for segment in range(len(X_segmented)):
                segment_filt = utils.filter_1seg(X_segmented[segment].transpose(), selected_electrodes_names,filters, sample_duration, freq_limits_names)
                segment_filt = segment_filt.transpose()
                X_all.append(X_segmented[segment])
                y_all.append(y[segment])
    #TODO check if filtering goes well
    X_np = np.stack(X_all)
    y_np = np.array(labels).ravel()
    #print(X_np.shape)

    # apply pipelines
    chosen_pipelines = utils.init_pipelines(FLAGS.p)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    scoring = {'f1': 'f1', 
        'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro',
        }

    for clf in chosen_pipelines:
        start_time = time.time()
        scores = cross_validate(chosen_pipelines[clf], X_segmented, y, cv=cv, n_jobs=-1, scoring=scoring, return_train_score=True)
        elapsed_time = time.time() - start_time
        results[clf] = {'filter_order': filter_order, 'filter_limits': freq_limits_names,
            'test_accuracy': scores['test_acc'].mean(),'test_f1': scores['test_f1'].mean(),'test_prec': scores['test_prec_macro'].mean(),
                'test_rec': scores['test_rec_macro'].mean(), 'time (seconds)': elapsed_time}  
        print(results[clf]) #TODO check results for this
    results_df = pd.DataFrame.from_dict(results, orient='index').sort_values('test_f1', ascending=False) 
    print(results_df)

if __name__ == '__main__':
     parser = argparse.ArgumentParser(description="Run offline BCI analysis")
     parser.add_argument("--p", type=list, default=['csp+lda'], help="The pipeline used. \
     This variable is a list containing all names of selected pipelines. Options currently are: ")

     FLAGS, unparsed = parser.parse_known_args()
     main()