import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
from sklearn.model_selection import KFold, cross_validate
import matplotlib.pyplot as plt

import src.utils as utils
import src.utils_deep as utils_deep

pd.options.mode.chained_assignment = None  # default='warn'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
    print(f'Initializing for {FLAGS.p} experimentation...')
    # INIT
    sampling_frequency = 200 # 250 for ours, 200 for Laura's
    sample_duration = 200 # ws of 200 seems to be best comrpomize between descision speed and enough data
    all_electrode_names = ['F3', 'FZ', 'FC1','FCZ','C1','CZ','CP1','CPZ', 
    'FC5', 'FC3','C5','C3','CP5','CP3','P3','PZ','F4','FC2','FC4','FC6','C2',
    'C4','CP2','CP4','C6','CP6','P4','HR' ,'HL', 'VU', 'VD']
    deleted_electrodes_names = ['F3','C5', 'F4','C2']
    selected_electrodes_names = [x for x in all_electrode_names if x not in deleted_electrodes_names]
    # for us: ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    n_electrodes = len(selected_electrodes_names)
    pre_filtering = False
    folder_path = Path('./data/offline/202100722_MI_atencion_online/')
    subject = '202100722_MI_atencion_online'
    result_path = Path('./data/offline/intermediate_datafiles/')
    result_path.mkdir(exist_ok=True, parents=True)
    results_fname = f'{FLAGS.p}_{time.time()}_filtering_experiments_{subject}.csv'

    if 'csp' in FLAGS.p:
        # filterbank
        list_of_freq_lim = [[[5, 10], [10, 15], [15, 20], [20, 25]], [[5, 15], [15, 25], [25, 35]],
        [[4, 8], [8, 12], [12, 16], [16, 20], [20,24], [24,28], [28,32], [32,36], [36,40]],
        [[5, 10], [10, 15], [15, 20], [20, 25], [25, 30], [30, 35]]]
        freq_limits_names_list = [['5_10Hz', '10_15Hz','15_20Hz','20_25Hz'], ['5_15Hz', '15_25Hz','25_35Hz'],
        ['4_8Hz', '8_12Hz','12_16Hz','16_20Hz', '20_24Hz','24_28Hz', '28_32Hz', '32_36Hz', '36_40Hz'],
        ['5_10Hz', '10_15Hz','15_20Hz','20_25Hz', '25_30Hz','30_35Hz']]
        filt_orders = [2,3,4]
    if 'riemann' in FLAGS.p:
        # only 1 bandpass filter
        list_of_freq_lim = [[[4, 30]], [[4, 35]], [[4,40]], [[8,30]], [[8,35]], [[8,40]]]
        freq_limits_names_list = [['4_30Hz'], ['4_35Hz'], ['4-40Hz'], ['8-30Hz'], ['8-35Hz'], ['8-40hz']]
        filt_orders = [2,3,4]
    if 'deep' in FLAGS.p:
        list_of_freq_lim = [[[4, 30]],[[4,40]], [[8,30]], [[8,40]], [[1,60]]]
        freq_limits_names_list = [['4_30Hz'], ['4-40Hz'], ['8-30Hz'], ['8-40hz'], ['1-60Hz']]
        filt_orders = [2,3,4]

    dataset_full = {}
    # getting data of all trials
    for instance in os.scandir(folder_path): 
        if instance.path.endswith('.mat'):
            print(f'adding_{instance} to dataset...')
            sig = scipy.io.loadmat(instance.path)
            X = sig['session']['data_EEG'][0][0][:-1,:].T
            y = sig['session']['task_EEG'][0][0].T 
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

    results = {}
    for filt_ord in range(len(filt_orders)):
        for freq_limit_instance in range(len(list_of_freq_lim)):
            freq_limits = np.asarray(list_of_freq_lim[freq_limit_instance]) 
            freq_limits_names = freq_limits_names_list[freq_limit_instance]
            filter_order = filt_orders[filt_ord]
            filters, z = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filter_order, 
            state_space = FLAGS.s)
            X_all = []
            y_all = []  
            print(f'experimenting with filter order of {filter_order}, freq limits of {freq_limits_names}, and ws of {sample_duration}.')
            for df in dataset_full:
                X_segmented, y = utils.segmentation_all(dataset_full[df],sample_duration)
                outliers = 0           
                for segment in range(len(X_segmented)):
                    # apply pre-processing and update filter state space vector in filters
                    segment_filt, outlier, filters = utils.pre_processing(X_segmented[segment], selected_electrodes_names, filters, 
                    sample_duration, freq_limits_names, z, state_space = FLAGS.s)
                    outliers += outlier
                    X_all.append(segment_filt)
                    y_all.append(y[segment])           
                print(f'Current length of X: {len(X_all)}.')
                print(f'Current amount of outliers: {outliers}')
            # transform to np for use in ML-pipeline
            X_np = np.stack(X_all)
            y_np = np.array(y_all).ravel()

            if 'deep' in FLAGS.p:
                # deep learning pipeline
                start_time = time.time()
                trainloader, valloader = utils_deep.data_setup(X_np, y_np, val_size=0.2)   
                train_accuracy, val_accuracy = utils_deep.run_model(trainloader, valloader)
                #plt.plot(train_accuracy,c= 'b')
                #plt.plot(val_accuracy,c= 'r')
                #plt.show()
                print(f'trainacc: {train_accuracy}')
                print(f'valacc: {val_accuracy}')
                elapsed_time = time.time() - start_time
                results[f"DL_{filter_order}_{freq_limits_names}"] = {'final_train_accuracy': train_accuracy[-1],'test_accuracy': val_accuracy[-1], 'filter_order' : filter_order, 
                        'freq_limits' : freq_limits_names, 'windowsize' : sample_duration, 
                            'time (seconds)': elapsed_time}                   
            else:
                if FLAGS.g ==True: 
                    # gridsearch experimentation
                    chosen_pipelines = utils.init_pipelines_grid(FLAGS.p)
                    for clf in chosen_pipelines:
                        print(f'applying {clf} with gridsearch...')
                        acc, elapsed_time, chosen_pipelines = utils.grid_search_execution(X_np, y_np, chosen_pipelines, clf)
                        results[f"grid_search_{clf}_{filter_order}_{freq_limits_names}_{sample_duration}"] = {'clf': clf, 'filter_order' : filter_order, 
                        'freq_limits' : freq_limits_names, 'windowsize' : sample_duration, 
                            'test_accuracy': acc, 'time (seconds)': elapsed_time, 'bestParams': chosen_pipelines[clf].best_params_ }                         
                else: 
                    # no gridsearch
                    # data leakage due to double cross val w/ gridsearch and here --> watch out
                    chosen_pipelines = utils.init_pipelines(FLAGS.p)
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    scoring = {'f1': 'f1', 'acc': 'accuracy','prec_macro': 'precision_macro','rec_macro': 'recall_macro'}
                    for clf in chosen_pipelines:
                        print(f'applying {clf}...')
                        start_time = time.time()
                        scores = cross_validate(chosen_pipelines[clf], X_np, y_np, cv=cv, n_jobs=-1, scoring=scoring, return_train_score=True)
                        elapsed_time = time.time() - start_time
                        results[f"{clf}_{filter_order}_{freq_limits_names}_{sample_duration}"] = {'clf': clf, 'filter_order' : filter_order, 
                        'freq_limits' : freq_limits_names, 'windowsize' : sample_duration, 
                            'test_accuracy': scores['test_acc'].mean(),'test_f1': scores['test_f1'].mean(),
                            'test_prec': scores['test_prec_macro'].mean(), 'test_rec': scores['test_rec_macro'].mean(), 
                            'time (seconds)': elapsed_time, 'train_accuracy': scores['train_acc'].mean() }  
    results_df = pd.DataFrame.from_dict(results, orient='index').sort_values('test_accuracy', ascending=False) 
    results_df.to_csv(result_path / results_fname)
    print('Finished.')

if __name__ == '__main__':
    #TODO change main to a Class --> possible to e.g. run both csp and riemann after each other with self. etc.
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--p", type=str, default=['csp'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'csp', 'riemann', 'deep'")
    parser.add_argument("--g", type=bool, default=False, help="Option to experiment with gridsearch pipelines \
    or without. This is a boolean variable. Default is False.")
    parser.add_argument("--s", type=bool, default=True, help="Option to experiment run filter with state space implemenation, \
    or without (using filtfilt). This is a boolean variable. Default is True.")
    FLAGS, unparsed = parser.parse_known_args()
    main()
