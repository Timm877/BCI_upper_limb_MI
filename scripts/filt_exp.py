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

def execution(pipeline_type, list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes):
    print(f'Initializing for {pipeline_type} experimentation...')
    print(FLAGS)
    # INIT
    sampling_frequency = 200 # 250 for ours, 200 for Laura's
    all_electrode_names = ['F3', 'FZ', 'FC1','FCZ','C1','CZ','CP1','CPZ', 
        'FC5', 'FC3','C5','C3','CP5','CP3','P3','PZ','F4','FC2','FC4','FC6','C2',
        'C4','CP2','CP4','C6','CP6','P4','HR' ,'HL', 'VU', 'VD']
    if FLAGS.less:
        # testing here for 8 electrodes:
        selected_electrodes_names = ['FZ', 'C3', 'CZ', 'C4', 'CPZ', 'P3', 'PZ', 'P4'] 
        # somewhat similar to Unicorn: ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
        n_electrodes = len(selected_electrodes_names)
    else:
        # IMPORTANT: DELETE! F3, C5, F4, C2 --> only used for eye artifact correction!
        deleted_electrodes_names = ['F3','C5', 'F4','C2']
        selected_electrodes_names = [x for x in all_electrode_names if x not in deleted_electrodes_names]
        n_electrodes = len(selected_electrodes_names)

    folder_path = Path('./data/offline/202100722_MI_atencion_online/')
    subject = '202100722_MI_atencion_online'   
    result_path = Path('./data/offline/intermediate_datafiles/transfer_learn/')
    result_path.mkdir(exist_ok=True, parents=True)
    results_fname = f'{pipeline_type}_{time.time()}_filtering_experiments_{subject}.csv'

    dataset_full = {}
    trials_amount = 0
    for instance in os.scandir(folder_path):
        if instance.path.endswith('.mat'): 
            trials_amount +=1
            print(f'adding_{instance} to dataset...')
            sig = scipy.io.loadmat(instance.path)
            X = sig['session']['data_EEG'][0][0][:-1,:].T
            y = sig['session']['task_EEG'][0][0].T 
            dataset = pd.DataFrame(X, columns=all_electrode_names)
            labels = pd.DataFrame(y, columns=['label'])
            dataset = dataset[selected_electrodes_names]    
            data_relax = dataset.loc[labels['label'] == 402] 
            data_MI = dataset.loc[labels['label'] == 404]
            data_relax['label'] = 0
            data_MI['label'] = 1
            dataset_full[str(instance)] = pd.concat([data_relax,data_MI], axis=0)
            dataset_full[str(instance)].reset_index(drop=True, inplace=True)

    results = {}
    for sample_duration in window_sizes:
        for filt_ord in range(len(filt_orders)):
            for freq_limit_instance in range(len(list_of_freq_lim)):
                freq_limits = np.asarray(list_of_freq_lim[freq_limit_instance]) 
                freq_limits_names = freq_limits_names_list[freq_limit_instance]
                filter_order = filt_orders[filt_ord]
                filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filter_order)
                X_train = []
                y_train = []  
                X_val = []
                y_val = []
                df_num = 0
                overlap = FLAGS.overlap
                print(f'experimenting with filter order of {filter_order}, freq limits of {freq_limits_names}, \
                and ws of {sample_duration}.')
                for df in dataset_full:
                    if overlap:
                        X_temp, y_temp = utils.segmentation_overlap_withfilt(dataset_full[df], sample_duration, filters,
                        selected_electrodes_names, freq_limits_names, pipeline_type, window_hop=100)
                        for segment in range(len(X_temp)): 
                            # add to x_all without initiating lists into lists
                            if df_num >1:# > 0 and df_num % 5 == 0: 
                                X_val.append(X_temp[segment])
                                y_val.append(y_temp[segment]) 
                            else:
                                X_train.append(X_temp[segment])
                                y_train.append(y_temp[segment]) 
                    else:
                        X_segmented, y = utils.segmentation_all(dataset_full[df],sample_duration) 
                        i=0    
                        for segment in range(len(X_segmented)):
                            # apply pre-processing and update filter state space vector in filters
                            segment_filt, outlier, filters = utils.pre_processing(X_segmented[segment], selected_electrodes_names, 
                            filters, sample_duration, freq_limits_names, pipeline_type)
                            if outlier > 0 or i == 0: 
                                # when i ==0, state space filters are (re-)initiated so signal is destroyed, 
                                # so segment is considered as outlier; this is at the start of each new df in data_set_full
                                # When outlier > 0, we have 1 or more 1 bad channel so segment is outlier.
                                print(f'A segment was considered as an outlier due to bad signal in {outlier} channels')
                            else:
                                if df_num > 0 and df_num % 5 == 0: 
                                    X_val.append(segment_filt)
                                    y_val.append(y[segment]) 
                                else: 
                                    X_train.append(segment_filt)
                                    y_train.append(y[segment]) 
                            i+=1
                    df_num += 1
                    print(f'Current length of X train: {len(X_train)}.')
                    print(f'Current length of X val: {len(X_val)}.')

                # transform to np for use in ML-pipeline
                X_train_np = np.stack(X_train)
                X_val_np = np.stack(X_val)
                if overlap:
                    y_train_np = np.array(y_train)
                    y_val_np = np.array(y_val)
                else:
                    y_train_np = np.array(y_train).ravel()
                    y_val_np = np.array(y_val).ravel()
                print(f"shape training set: {X_train_np.shape}")
                print(f"shape validation set: {X_val_np.shape}")

                # check continuity of signals
                #X_concat = np.concatenate((X_np[5, 5, :],X_np[6, 5, :]))
                #plt.plot(np.arange(0,100), X_np[0, 5, 100:])
                #plt.plot(np.arange(100,300), X_np[1, 5, :])
                #plt.plot(np.arange(400, 600), X_np[2, 5, :])
                #plt.show()

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
                        trainloader, valloader, lr, sample_duration, n_electrodes, receptive_field, filter_sizing, mean_pool)
                    print(f'trainacc: {train_accuracy}')
                    print(f'valacc: {val_accuracy}')
                    elapsed_time = time.time() - start_time

                    results[f"DL_rf{receptive_field}_filtersize{filter_sizing}_meanpool{mean_pool}"] = {'final_train_accuracy': np.array(train_accuracy).mean(),
                    'test_accuracy': np.array(val_accuracy).mean(), 'final_train_f1': np.array(train_f1).mean(),
                    'test_f1': np.array(val_f1).mean(), 'train_classacc': train_classacc_iters, 'val_classacc': val_classacc_iters, 
                    'filter_order' : filter_order, 'freq_limits' : freq_limits_names, 
                    'windowsize' : sample_duration, 'time (seconds)': elapsed_time}                   
                else:
                    if FLAGS.grid == True: 
                        # gridsearch experimentation
                        chosen_pipelines = utils.init_pipelines_grid(pipeline_type)
                        for clf in chosen_pipelines:
                            print(f'applying {clf} with gridsearch...')
                            acc, acc_classes, f1, elapsed_time, chosen_pipelines = utils.grid_search_execution(X_train_np, y_train_np, 
                            X_val_np, y_val_np, chosen_pipelines, clf)

                            results[f"grid_search_{clf}_{filter_order}_{freq_limits_names}_{sample_duration}"] = {
                            'clf': clf, 'filter_order' : filter_order, 'freq_limits' : freq_limits_names, 
                            'windowsize' : sample_duration, 'test_accuracy': acc, 'acc_classes': acc_classes, 
                            'test_f1' : f1, 'time (seconds)': elapsed_time}#, 'bestParams': chosen_pipelines[clf].best_params_ }                         
                    else: 
                        chosen_pipelines = utils.init_pipelines(pipeline_type)
                        cv = KFold(n_splits=5, shuffle=True, random_state=42)
                        scoring = {'f1': 'f1', 'acc': 'accuracy','prec_macro': 'precision_macro','rec_macro': 'recall_macro'}
                        for clf in chosen_pipelines:
                            print(f'applying {clf}...')
                            start_time = time.time()
                            scores = cross_validate(chosen_pipelines[clf], X_np, y_np, cv=cv, n_jobs=-1, scoring=scoring, return_train_score=True)
                            elapsed_time = time.time() - start_time
                            results[f"{clf}_{filter_order}_{freq_limits_names}_{sample_duration}"] = {'clf': clf, 
                            'filter_order' : filter_order, 'freq_limits' : freq_limits_names, 'windowsize' : sample_duration, 
                                'test_accuracy': scores['test_acc'].mean(),'test_f1': scores['test_f1'].mean(),
                                'test_prec': scores['test_prec_macro'].mean(), 'test_rec': scores['test_rec_macro'].mean(), 
                                'time (seconds)': elapsed_time, 'train_accuracy': scores['train_acc'].mean() }  
    results_df = pd.DataFrame.from_dict(results, orient='index').sort_values('test_accuracy', ascending=False) 
    results_df.to_csv(result_path / results_fname)
    print('Finished.')

def main():
    if 'csp' in FLAGS.pline:
        # filterbank
        list_of_freq_lim = [[[5, 10], [10, 15], [15, 20], [20, 25]]]
        #[[4, 8], [8, 12], [12, 16], [16, 20], [20,24], [24,28], [28,32], [32,36], [36,40]],]
        freq_limits_names_list = [['5_10Hz', '10_15Hz','15_20Hz','20_25Hz']]
        #['4_8Hz', '8_12Hz','12_16Hz','16_20Hz', '20_24Hz','24_28Hz', '28_32Hz', '32_36Hz', '36_40Hz']]
        filt_orders = [2]
        window_sizes = [400]
        execution('csp', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes)
    if 'riemann' in FLAGS.pline:
        # only 1 bandpass filter
        list_of_freq_lim = [[[4, 35]]]#,[[8, 35]]]
        freq_limits_names_list = [['4_35Hz']]#, ['8_35Hz']]
        filt_orders = [2]
        window_sizes = [400]
        execution('riemann', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes)
    if 'deep' in FLAGS.pline:
        list_of_freq_lim = [[[4,35]]]#, [[8, 35]], [[4,40]]]
        freq_limits_names_list = [['4-35Hz']]#, ['8_35Hz'], ['4-40Hz']]
        filt_orders = [2]
        window_sizes = [400]
        execution('deep', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--pline", nargs='+', default=['csp'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'csp', 'riemann', 'deep'")
    parser.add_argument("--grid", type=bool, default=False, help="Option to experiment with gridsearch pipelines \
    or without. This is a boolean variable. Default is False.")
    parser.add_argument("--less", type=bool, default=False, help="Option to experiment with less electrodes or with full setup. \
    This is a boolean variable. Default is False (all electrodes will be used).")
    parser.add_argument("--overlap", type=bool, default=False, help="Option to experiment with overlap of 0.5 between segments. \
    This is a boolean variable. Default is True (with overlap).")
    FLAGS, unparsed = parser.parse_known_args()
    main()

    
