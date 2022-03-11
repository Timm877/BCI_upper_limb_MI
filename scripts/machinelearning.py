import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from meegkit.asr import ASR
import pickle
import scipy.io
from scipy import signal
from sklearn.model_selection import KFold, cross_validate
import matplotlib.pyplot as plt

import src.unicorn_utils as utils
import src.utils_deep as utils_deep

pd.options.mode.chained_assignment = None  # default='warn'

def execution(pipeline_type, subject):
    print(f'Initializing for {pipeline_type} machine learning...')
    # INIT
    sampling_frequency = 250 
    # testing here for 8 electrodes:
    electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    #file_elec_names = ['EEG1', 'EEG2', 'EEG3', 'EEG4', 'EEG5', 'EEG6', 'EEG7', 'EEG8']
    n_electrodes = len(electrode_names)


    folder_path = Path(f'./data/pilots/intermediate_datafiles/preprocess/{subject}')
    result_path = Path(f'./results/intermediate_datafiles/pilots/{subject}')
    result_path.mkdir(exist_ok=True, parents=True)
    results_fname = f'{pipeline_type}_UL_multiclass.csv'
    num_classes = 3
    results = {}
    for instance in os.scandir(folder_path):
        if pipeline_type in instance.path: 
            print(f'Running for {instance.path}...')
            a_file = open(instance.path, "rb")
            data_dict = pickle.load(a_file)
            X = data_dict['data']
            y = data_dict['labels']
            window_size = int(instance.path.split("ws",1)[1][:3]) 
            train_acc_cv, val_acc_cv, train_f1_cv, val_f1_cv, acc_classes_cv = [], [], [], [], []

            for cross_val in range(list(X)[-1] + 1):
                X_train, y_train, X_val, y_val = [], [], [], []
                for df in X: 
                    for segment in range(len(X[df])): 
                        if df == cross_val:
                            X_val.append(X[df][segment])
                            y_val.append(y[df][segment]) 
                        else:
                            X_train.append(X[df][segment])
                            y_train.append(y[df][segment]) 
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
                    train_acc_cv.append(np.array(train_accuracy).mean())
                    val_acc_cv.append(np.array(val_accuracy).mean())
                    train_f1_cv.append(np.array(train_f1).mean())
                    val_f1_cv.append(np.array(val_f1).mean())    
                else:
                    # gridsearch experimentation csp or riemann
                    chosen_pipelines = utils.init_pipelines_grid(pipeline_type)
                    for clf in chosen_pipelines:
                        print(f'applying {clf} with gridsearch...')
                        val_accuracy, acc_classes, f1, elapsed_time, chosen_pipelines = utils.grid_search_execution(X_train_np, y_train_np, 
                        X_val_np, y_val_np, chosen_pipelines, clf)
                        val_acc_cv.append(val_accuracy)
                        val_f1_cv.append(f1)
                        acc_classes_cv.append(acc_classes)
                        print(acc_classes_cv)

            if 'deep' in pipeline_type:    
                results[f"crossval_{instance.path}"] = {'final_train_accuracy': np.array(train_acc_cv).mean(),
                'final_val_accuracy': np.array(val_acc_cv).mean(), 'final_train_f1': np.array(train_f1_cv).mean(),
                'final_val_f1': np.array(val_f1_cv).mean(), 'full_trainacc': train_acc_cv, 'full_valacc': val_acc_cv}         
            else:
                results[f"crossval_{instance.path}"] = {'final_val_accuracy': np.array(val_acc_cv).mean(), 
                    'final_val_f1': np.array(val_f1_cv).mean(), 'full_valacc': val_acc_cv, 'full_acc_classes': acc_classes_cv}  
            print('Finished 1 pipeline')
    results_df = pd.DataFrame.from_dict(results, orient='index').sort_values('final_val_accuracy', ascending=False)  
    results_df.to_csv(result_path / results_fname)
    print('Finished')

def main():
    for subj in FLAGS.subjects:
        print(subj)
        if 'csp' in FLAGS.pline:
            execution('csp', subj)
        if 'riemann' in FLAGS.pline:
            execution('riemann', subj)
        if 'deep' in FLAGS.pline:
            execution('deep', subj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--pline", nargs='+', default=['csp'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'csp', 'riemann', 'deep'")
    parser.add_argument("--subjects", nargs='+', default=['X02_wet'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are in the data folder.")
    FLAGS, unparsed = parser.parse_known_args()
    main()

    
