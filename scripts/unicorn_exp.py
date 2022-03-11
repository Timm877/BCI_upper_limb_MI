import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from meegkit.asr import ASR
import scipy.io
from scipy import signal
from sklearn.model_selection import KFold, cross_validate
import matplotlib.pyplot as plt

import src.unicorn_utils as utils
import src.utils_deep as utils_deep

pd.options.mode.chained_assignment = None  # default='warn'

def execution(pipeline_type, list_of_freq_lim, freq_limits_names_list, filt_ord, window_size):
    print(f'Initializing for {pipeline_type} experimentation...')
    # INIT
    sampling_frequency = 250 
    # testing here for 8 electrodes:
    electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    #file_elec_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8']
    n_electrodes = len(electrode_names)

    subject = 'X02_wet'  
    folder_path = Path(f'./data/unicorn/pilots/expdata/Subjects/{subject}/openloop')
    print(folder_path)
    env_noise_path = Path(f'./data/unicorn/pilots/expdata/Subjects/{subject}/Envdata')
    result_path = Path(f'./results/intermediate_datafiles/pilots/{subject}')
    result_path.mkdir(exist_ok=True, parents=True)
    results_fname = f'multiclass_{pipeline_type}_{time.time()}_filtering_experiments_{subject}.csv'
    num_classes = 3
    
    dataset_full = {}
    trials_amount = 0

    asr = None
    for instance in os.scandir(env_noise_path):
        if instance.path.endswith('.csv'): 
            env_noise_X = pd.read_csv(instance.path)
    env_noise_X = env_noise_X.loc[:,electrode_names].T
    #apply 0.5 butter filt
    b_env,a_env = signal.butter(2, 0.5/(sampling_frequency/2), 'highpass')
    env_noise_X_filt = signal.filtfilt(b_env, a_env, env_noise_X)
    # Initialize ASR!
    asr = ASR(method='euclid')
    asr.fit(env_noise_X_filt)

    for instance in os.scandir(folder_path):
        if instance.path.endswith('.csv'): 
            trials_amount +=1
            print(f'adding_{instance} to dataset...')
            sig = pd.read_csv(instance.path)
            X = sig.loc[:,electrode_names]
            y = sig.loc[:,'Class']
            dataset_full[str(instance)] = pd.concat([X,y], axis=1)
            #print(pd.value_counts(y))
    results = {}   
    for freq_limit_instance in range(len(list_of_freq_lim)):
        freq_limits = np.asarray(list_of_freq_lim[freq_limit_instance]) 
        freq_limits_names = freq_limits_names_list[freq_limit_instance]
        filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filt_ord)
        X_train, y_train, X_val, y_val  = [], [], [], []
        df_num = 0
        print(f'experimenting with filter order of {filt_ord}, freq limits of {freq_limits_names}, \
                and ws of {window_size}.')
        for df in dataset_full:
            X_temp, y_temp = utils.unicorn_segmentation_overlap_withfilt(dataset_full[df], window_size, filters,
            electrode_names, freq_limits_names, pipeline_type, sampling_frequency, asr)

            for segment in range(len(X_temp)): 
                if df_num == 3 or df_num == 7: # > 0 and df_num % 5 == 0: 
                    X_val.append(X_temp[segment])
                    y_val.append(y_temp[segment]) 
                else:
                    X_train.append(X_temp[segment])
                    y_train.append(y_temp[segment]) 
            df_num += 1
            print(f'Current length of X train: {len(X_train)}.')
            print(f'Current length of X val: {len(X_val)}.')

        X_train_np = np.stack(X_train)
        X_val_np = np.stack(X_val)
        y_train_np = np.array(y_train)
        y_val_np = np.array(y_val)
        print(f"shape training set: {X_train_np.shape}")
        print(f"shape validation set: {X_val_np.shape}")

        if 'deep' in pipeline_type:
            # deep learning pipeline
            start_time = time.time()
            trainloader, valloader = utils_deep.data_setup(X_train_np, y_train_np, X_val_np, y_val_np) 
            lr = 0.0005
            receptive_field = 65 # chosen by experimentation (see deeplearn_experiment folder) 
            # In paper1, they also use 65, but also collect more samples (3seconds)
            filter_sizing = 10 # Chosen by experimentation. Depth of conv layers; 40 was used in appers
            mean_pool = 15 # Chosen by experimentation. 15 was used in papers
            train_accuracy, val_accuracy, train_f1, val_f1, train_classacc_iters, val_classacc_iters = utils_deep.run_model(
                trainloader, valloader, lr, window_size, n_electrodes, receptive_field, filter_sizing, mean_pool, num_classes)
            print(f'trainacc: {train_accuracy}')
            print(f'valacc: {val_accuracy}')
            elapsed_time = time.time() - start_time
            results[f"{freq_limits_names}_DL_rf{receptive_field}_filtersize{filter_sizing}_meanpool{mean_pool}"] = {'final_train_accuracy': np.array(train_accuracy).mean(),
            'test_accuracy': np.array(val_accuracy).mean(), 'final_train_f1': np.array(train_f1).mean(),
            'test_f1': np.array(val_f1).mean(), 'train_classacc': train_classacc_iters, 'val_classacc': val_classacc_iters, 
            'filter_order' : filt_ord, 'freq_limits' : freq_limits_names, 
            'windowsize' : window_size, 'time (seconds)': elapsed_time}                   
        else:
            # gridsearch experimentation
            chosen_pipelines = utils.init_pipelines_grid(pipeline_type)
            for clf in chosen_pipelines:
                print(f'applying {clf} with gridsearch...')
                acc, acc_classes, f1, elapsed_time, chosen_pipelines = utils.grid_search_execution(X_train_np, y_train_np, 
                X_val_np, y_val_np, chosen_pipelines, clf)
                results[f"grid_search_{clf}_{filt_ord}_{freq_limits_names}_{window_size}"] = {
                'clf': clf, 'filter_order' : filt_ord, 'freq_limits' : freq_limits_names, 
                'windowsize' : window_size, 'test_accuracy': acc, 'acc_classes': acc_classes, 
                'test_f1' : f1, 'time (seconds)': elapsed_time}

    results_df = pd.DataFrame.from_dict(results, orient='index').sort_values('test_accuracy', ascending=False) 
    results_df.to_csv(result_path / results_fname)
    print('Finished.')


def main():
    if 'csp' in FLAGS.pline:
        # filterbank
        list_of_freq_lim = [[[4, 8], [8, 12], [12, 16], [16, 20], [20,24], [24,28], [28,32], [32,36], [36,40]],]
        #[[5, 10], [10, 15], [15, 20], [20, 25]]]
        freq_limits_names_list = [['4_8Hz', '8_12Hz','12_16Hz','16_20Hz', '20_24Hz','24_28Hz', '28_32Hz', '32_36Hz', '36_40Hz'],]
        #['5_10Hz', '10_15Hz','15_20Hz','20_25Hz']]
        filt_orders = 2
        window_sizes = 500
        execution('csp', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes)
    if 'riemann' in FLAGS.pline:
        # only 1 bandpass filter
        list_of_freq_lim = [[[4, 35]]]#, [[8, 35]], [[4, 40]]]
        freq_limits_names_list = [['4_35Hz']]#, ['8_35Hz'], ['4-40Hz']]
        filt_orders = 2
        window_sizes = 500
        execution('riemann', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes)
    if 'deep' in FLAGS.pline:
        list_of_freq_lim = [[[4,35]], [[8, 35]], [[4,40]]]
        freq_limits_names_list = [['4-35Hz'], ['8_35Hz'], ['4-40Hz']]
        filt_orders = 2
        window_sizes = 500
        execution('deep', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--pline", nargs='+', default=['csp'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'csp', 'riemann', 'deep'")
    FLAGS, unparsed = parser.parse_known_args()
    main()

    
