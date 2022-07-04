import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from mne.decoding import CSP
import mne
import matplotlib.pyplot as plt
import src.utils_preprocess as utils
import src.utils_deep as utils_deep

pd.options.mode.chained_assignment = None  # default='warn'

def main():
    for subj in FLAGS.subjects:
        print(subj)
        for type in FLAGS.type:
            print(type)
            for pline in FLAGS.pline:
                print(pline)
                execution(pline, subj, type)

def execution(pipeline_type, subject, type):
    print(f'Initializing for {pipeline_type} machine learning...')
    # INIT
    electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    n_electrodes = len(electrode_names)
    folder_path = Path(f'./data/openloop/intermediate_datafiles/preprocess/csp_1band')
    result_path = Path(f'./results/intermediate_datafiles/openloop/{subject}_freqbands')
    result_path.mkdir(exist_ok=True, parents=True)
    window_sie = 500
    if type == 'arms':
        results_fname = f'{pipeline_type}_{type}.csv'
        num_classes = 2
    elif type == 'binary':
        results_fname = f'{pipeline_type}_{type}.csv'
        num_classes = 2
    elif type == 'multiclass':
        results_fname = f'{pipeline_type}_STD_{type}.csv'
        num_classes = 3
    elif type == 'all':
        results_fname = f'{pipeline_type}_{type}.csv'
        num_classes = 4
    results = {}

    for instance in os.scandir(folder_path):
        if pipeline_type[:4] in instance.path: 
            print(f'Running for {instance.path}...')
            a_file = open(instance.path, "rb")
            data_dict = pickle.load(a_file)
            X = data_dict['data']
            y = data_dict['labels']
            #window_size = int(instance.path.split("ws",1)[1][:3]) 

            if pipeline_type[:4] == 'deep':
                train_acc_cv, val_acc_cv, val_prec_cv, val_rec_cv, train_f1_cv, val_f1_cv, \
                val_roc_auc_cv, acc_classes_cv, train_acc_std_cv, val_acc_std_cv = \
                    [], [], [], [], [], [], [], [],[],[]
            else:
                train_acc_cv, val_acc_cv, val_prec_cv, val_rec_cv, train_f1_cv, val_f1_cv, \
                val_roc_auc_cv, acc_classes_cv = {}, {}, {}, {}, {}, {}, {}, {}

            samp_num = 0
            for cross_val in range(1):#list(X)[-1] + 1):
                X_train, y_train, X_val, y_val = [], [], [], []
                for df in X: 
                    for segment in range(len(X[df])): 
                        samp_num +=1
                        if type == 'arms':
                            # only arms
                            if y[df][segment] == 1 or y[df][segment] == 2:
                                if df == cross_val:
                                    X_val.append(X[df][segment])
                                    y_val.append(y[df][segment] -1) 
                                else:
                                    X_train.append(X[df][segment])
                                    y_train.append(y[df][segment] -1) 
                        elif type == 'binary':
                            if (y[df][segment] == 1 or y[df][segment] == 2) and samp_num % 2 > 0:
                                #print(samp_num)
                                # only select half of the samples for left arm / right arm
                                # and make label just 1
                                # to have balanced MI vs relax
                                if df == cross_val:
                                    X_val.append(X[df][segment])
                                    y_val.append(1) 
                                else:
                                    X_train.append(X[df][segment])
                                    y_train.append(1) 
                            elif y[df][segment] == 0:
                                if df == cross_val:
                                    X_val.append(X[df][segment])
                                    y_val.append(0) 
                                else:
                                    X_train.append(X[df][segment])
                                    y_train.append(0)
                        elif type == 'multiclass':
                            # only upperlimb
                            if y[df][segment] == 1 or y[df][segment] == 2:# or y[df][segment] == 2:
                                if df == cross_val:
                                        X_val.append(X[df][segment])
                                        y_val.append(y[df][segment]) 
                                else:
                                    X_train.append(X[df][segment])
                                    y_train.append(y[df][segment]) 
                                    
                        elif type == 'all':
                            # classify everything (relax, left arm, right arm, legs)
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
                    receptive_field = 50 
                    filter_sizing = 20 
                    mean_pool = 15 

                    train_accuracy, val_accuracy, train_f1, val_f1, train_classacc, val_classacc, \
                    training_precision, training_recall, validation_precision, validation_recall, validation_roc_auc = \
                        utils_deep.run_model(trainloader, valloader, lr, window_size, n_electrodes, receptive_field, 
                        filter_sizing, mean_pool, num_classes, pipeline_type)

                    print(f'trainacc: {train_accuracy}')
                    print(f'valacc: {val_accuracy}')
                    train_acc_cv.append(np.array(train_accuracy).mean())
                    train_acc_std_cv.append(np.array(train_accuracy).std())
                    val_acc_cv.append(np.array(val_accuracy).mean())
                    val_acc_std_cv.append(np.array(val_accuracy).std())
                    train_f1_cv.append(np.array(train_f1).mean())
                    val_f1_cv.append(np.array(val_f1).mean())   
                    val_prec_cv.append(np.array(validation_precision).mean())
                    val_rec_cv.append(np.array(validation_recall).mean()) 
                    val_roc_auc_cv.append(np.array(validation_roc_auc).mean()) 
                else:
                    # gridsearch experimentation csp or riemann
                    chosen_pipelines = utils.init_pipelines_grid(pipeline_type)
                    
                    csp = CSP(n_components=3)
                    info = mne.create_info(['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8'], 250, ch_types='eeg')
                    info.set_montage('standard_1020')

                    epochs = mne.EpochsArray(X_train_np, info)
                    csp.fit_transform(X_train_np, y_train_np)
                    fig  = csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)
                    fig.savefig(f'topomap_csp_rightleft_{instance.path[-10:-7]}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--pline", nargs='+', default=['csp'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'csp', 'riemann', 'deep', \
    'deep_1dcnn, 'deep_inception'")
    parser.add_argument("--subjects", nargs='+', default=['X02_wet'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are in the data folder.")
    parser.add_argument("--type", nargs='+', default=['multiclass'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'all', 'multiclass', 'arms', 'binary'")
    FLAGS, unparsed = parser.parse_known_args()
    main()