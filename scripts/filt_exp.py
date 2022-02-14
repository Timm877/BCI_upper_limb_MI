import argparse
import os
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.io
import src.utils as utils
from sklearn.model_selection import KFold, cross_validate
from scipy import signal

pd.options.mode.chained_assignment = None  # default='warn'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
    print('Initializing...')
    # INIT
    sampling_frequency = 200 # 250 for ours, 200 for Laura's
    all_electrode_names = ['F3', 'FZ', 'FC1','FCZ','C1','CZ','CP1','CPZ', 
    'FC5', 'FC3','C5','C3','CP5','CP3','P3','PZ','F4','FC2','FC4','FC6','C2',
    'C4','CP2','CP4','C6','CP6','P4','HR' ,'HL', 'VU', 'VD']
    deleted_electrodes_names = ['F3','C5', 'F4','C2']
    selected_electrodes_names = [x for x in all_electrode_names if x not in deleted_electrodes_names]
    # for us: ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    n_electrodes = len(selected_electrodes_names)

    folder_path = Path('./data/offline/20210517_MI_atencion_online/')
    result_path = Path('./data/offline/intermediate_datafiles/')
    result_path.mkdir(exist_ok=True, parents=True)
    
    if 'csp' in FLAGS.p:
        results_fname = f'csp_{time.time()}_filtering_experiments.csv'
        list_of_freq_lim = [[[5, 10], [10, 15], [15, 20], [20, 25]], 
        [[5, 15], [15, 25], [25, 35]],
        [[4, 8], [8, 12], [12, 16], [16, 20], [20,24], [24,28], [28,32], [32,36], [36,40]],
        [[5, 10], [10, 15], [15, 20], [20, 25], [25, 30], [30, 35]]]
        freq_limits_names_list = [['5_10Hz', '10_15Hz','15_20Hz','20_25Hz'],
        ['5_15Hz', '15_25Hz','25_35Hz'],
        ['4_8Hz', '8_12Hz','12_16Hz','16_20Hz', '20_24Hz','24_28Hz', '28_32Hz', '32_36Hz', '36_40Hz'],
        ['5_10Hz', '10_15Hz','15_20Hz','20_25Hz', '25_30Hz','30_35Hz']]
        filt_orders = [2,3,4]
        sample_durations = [50, 100, 200, 400] # 0.25, 0.5, 1 or 2 seconds

    if 'riemann' in FLAGS.p:
        results_fname = f'riemann_{time.time()}_filtering_experiments.csv'
        list_of_freq_lim = [[[3, 30]], 
        [[3, 35]],
        [[4,35]], [[4,30]], [[8,35]]]
        freq_limits_names_list = [['3_30Hz'],
        ['3_35Hz'], ['4-35Hz'], ['4-30Hz'],['8-35hz']]
        filt_orders = [2,3,4]
        sample_durations = [50, 100, 200, 400] # 0.25, 0.5, 1 or 2 seconds

    if 'deep' in FLAGS.p:
        results_fname = f'deepl_{time.time()}_filtering_experiments.csv'
        list_of_freq_lim = [[[3, 30]], 
        [[3, 35]],
        [[4,35]], [[4,30]], [[8,35]]]
        freq_limits_names_list = [['3_30Hz'],
        ['3_35Hz'], ['4-35Hz'], ['4-30Hz'],['8-35hz']]
        filt_orders = [2,3,4]
        sample_durations = [50, 100, 200, 400] # 0.25, 0.5, 1 or 2 seconds

    results = {}

    dataset_full = {}
    for instance in os.scandir(folder_path): 
        if instance.path.endswith('.mat'):
            print(f'adding_{instance} to dataset...')
            sig = scipy.io.loadmat(instance.path)
            X = sig['session']['data_EEG'][0][0][:-1,:].T #get relevant data and transpose
            y = sig['session']['task_EEG'][0][0].T #get relevant data and transpose
            dataset = pd.DataFrame(X, columns=all_electrode_names)
            labels = pd.DataFrame(y, columns=['label'])

            # IMPORTANT: DELETE! F3, C5, F4, C2 --> only used for eye artifact correction!
            dataset.drop(deleted_electrodes_names, axis=1, inplace=True)
            
            data_relax = dataset.loc[labels['label'] == 402] 
            data_MI = dataset.loc[labels['label'] == 404]
            data_relax['label'] = 0
            data_MI['label'] = 1
            dataset_full[str(instance)] = pd.concat([data_relax,data_MI], axis=0)
            dataset_full[str(instance)].reset_index(drop=True, inplace=True)

    for filt_ord in range(len(filt_orders)):
        for freq_limit_instance in range(len(list_of_freq_lim)):
            for sample_duration in sample_durations:
                freq_limits = np.asarray(list_of_freq_lim[freq_limit_instance]) 
                freq_limits_names = freq_limits_names_list[freq_limit_instance]
                filter_order = filt_orders[filt_ord]
                print(f'experimenting with filter order of {filter_order}, freq limits of {freq_limits_names}, and ws of {sample_duration}.')

                filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filter_order)
                X_all = []
                y_all = []

                for df in dataset_full:
                    X_segmented, y = utils.segmentation_all(dataset_full[df],sample_duration)
                    for segment in range(len(X_segmented)):
                        segment_filt = utils.filter_1seg(X_segmented[segment].transpose(), selected_electrodes_names,filters, sample_duration,
                        freq_limits_names)
                        segment_filt = segment_filt.transpose()

                        # append each segment-df to a list of dfs
                        X_all.append(segment_filt)
                        y_all.append(y[segment])
                    print(f'Current length of X: {len(X_all)}.')

                # transform to np for use in ML-pipeline
                X_np = np.stack(X_all)
                y_np = np.array(y_all).ravel()

                chosen_pipelines = utils.init_pipelines(FLAGS.p)
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                scoring = {'f1': 'f1', 
                    'acc': 'accuracy',
                    'prec_macro': 'precision_macro',
                    'rec_macro': 'recall_macro',
                    }

                for clf in chosen_pipelines:
                    print(f'applying {clf}...')
                    start_time = time.time()
                    scores = cross_validate(chosen_pipelines[clf], X_np, y_np, cv=cv, n_jobs=-1, scoring=scoring, return_train_score=True)
                    elapsed_time = time.time() - start_time
                    results[f"{clf}_{filter_order}_{freq_limits_names}_{sample_duration}"] = {'clf': clf, 'filter_order' : filter_order, 
                    'freq_limits' : freq_limits_names, 'windowsize' : sample_duration, 
                        'test_accuracy': scores['test_acc'].mean(),'test_f1': scores['test_f1'].mean(),
                        'test_prec': scores['test_prec_macro'].mean(), 'test_rec': scores['test_rec_macro'].mean(), 
                        'time (seconds)': elapsed_time, }  
    results_df = pd.DataFrame.from_dict(results, orient='index').sort_values('test_f1', ascending=False) 
    results_df.to_csv(result_path / results_fname)
    print('Finished. Boem')

if __name__ == '__main__':
    # TODO update filter parameters after each segment?
    parser = argparse.ArgumentParser(description="Run offline BCI analysis")
    parser.add_argument("--p", type=str, default=['csp'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'csp', 'riemann', 'deep'")

    FLAGS, unparsed = parser.parse_known_args()
    main()
