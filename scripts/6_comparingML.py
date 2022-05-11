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
    folder_path = Path(f'./data/openloop/intermediate_datafiles/preprocess/{subject}_TLcompare')
    result_path = Path(f'./results/intermediate_datafiles/TLcompare/{subject}_TLcompare_all')
    result_path.mkdir(exist_ok=True, parents=True)
    results_fname = f'{pipeline_type}_{subject}_TLcompare.csv'

    results = {}
    for instance in os.scandir(folder_path):
        if pipeline_type[:4] in instance.path: 
            print(f'Running for {instance.path}...')
            a_file = open(instance.path, "rb")
            data_dict = pickle.load(a_file)
            X = data_dict['data']
            y = data_dict['labels']
            train_acc_cv, val_acc_cv, val_prec_cv, val_rec_cv, train_f1_cv, val_f1_cv, \
            val_roc_auc_cv, acc_classes_cv = {}, {}, {}, {}, {}, {}, {}, {}

    for trial_num in range(1,5):
        X_train, y_train, X_test, y_test = [], [], [], []
        for df in X:
            for segment in range(len(X[df])): 
                # upperlimb classification
                if y[df][segment] == 0 or y[df][segment] == 1 or y[df][segment] == 2:
                    if df < (trial_num): 
                        # earlier trials in training
                        X_train.append(X[df][segment])
                        y_train.append(y[df][segment])  
                    elif df > 4:
                        X_test.append(X[df][segment])
                        y_test.append(y[df][segment])  
            print(f'Current length of X train: {len(X_train)}.')
            print(f'Current length of X test: {len(X_test)}.')
        X_train_np = np.stack(X_train)
        X_test_np = np.stack(X_test)
        y_train_np = np.array(y_train)
        y_test_np = np.array(y_test)

        # gridsearch experimentation csp or riemann
        chosen_pipelines = utils.init_pipelines_grid(pipeline_type)
        for clf in chosen_pipelines:
            print(f'applying {clf} with gridsearch...')
            test_accuracy, prec, rec, roc_auc, acc_classes, f1, elapsed_time, chosen_pipelines = utils.grid_search_execution(
                X_train_np, y_train_np, X_test_np, y_test_np, chosen_pipelines, clf)
            if clf not in val_acc_cv:
                val_acc_cv[clf],val_prec_cv[clf],val_rec_cv[clf],val_f1_cv[clf],val_roc_auc_cv[clf],\
                    acc_classes_cv[clf] = [],[],[],[],[],[]
            val_acc_cv[clf].append(test_accuracy)
            val_prec_cv[clf].append(prec)
            val_rec_cv[clf].append(rec)
            val_f1_cv[clf].append(f1)
            val_roc_auc_cv[clf].append(np.array(roc_auc).mean()) 
            acc_classes_cv[clf].append(acc_classes)
        
            results[f"trialnum_{trial_num}_{clf}_crossval_{instance.path}"] = {'final_test_accuracy': np.around(np.array(val_acc_cv[clf]).mean(),3), 
                'final_test_f1': np.around(np.array(val_f1_cv[clf]).mean(),3), 'test_prec': np.around(np.array(val_prec_cv[clf]).mean(),3), 
            'test_rec': np.around(np.array(val_rec_cv[clf]).mean(),3), 'test_roc_auc': np.around(np.array(val_roc_auc_cv[clf]).mean(),3), 
            'full_testacc': val_acc_cv[clf], 'full_acc_classes': acc_classes_cv[clf]}  
        print('Finished 1 pipeline')
    results_df = pd.DataFrame.from_dict(results, orient='index').sort_values('final_test_accuracy', ascending=False)  
    results_df.to_csv(result_path / results_fname)
    print('Finished')

def main():
    for subj in FLAGS.subjects:
        print(subj)
        for pline in FLAGS.pline:
            print(pline)
            execution(pline, subj)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--pline", nargs='+', default=['csp'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'csp', 'riemann', 'deep', \
    'deep_1dcnn, 'deep_inception'")
    parser.add_argument("--subjects", nargs='+', default=['X02_wet'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are in the data folder.")  
    FLAGS, unparsed = parser.parse_known_args()
    main()