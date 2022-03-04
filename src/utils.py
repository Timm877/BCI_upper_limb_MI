import copy
import glob
import os
import shutil
import time
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from mne.decoding import CSP
from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy import signal, stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix

def init_filters(freq_lim, sample_freq, filt_type = 'bandpass', order=2, state_space=True):
    filters = []
    for f in range(freq_lim.shape[0]):
        A, B, C, D, Xnn  = init_filt_coef_statespace(freq_lim[f], fs=sample_freq, filtype=filt_type, order=order)
        filters.append([A, B, C, D, Xnn ])
    return filters

def init_filt_coef_statespace(cuttoff, fs, filtype, order,len_selected_electrodes=27):
    if filtype == 'lowpass':
        b,a = signal.butter(order, cuttoff[0]/(fs/2), filtype)
    elif filtype == 'bandpass':
        b,a = signal.butter(order, [cuttoff[0]/(fs/2), cuttoff[1]/(fs/2)], filtype)
    elif filtype == 'highpass':
        b,a = signal.butter(order, cuttoff[0]/(fs/2), filtype)
    # getting matrices for the state-space form of the filter from scipy and init state vector
    A, B, C, D = signal.tf2ss(b,a)
    Xnn = np.zeros((1,len_selected_electrodes,A.shape[0],1))
    return A, B, C, D, Xnn 

def apply_filter_statespace(sig, A, B, C, D, Xnn):
    # Substract mean for centering around 0
    # as the signal has to be centered at zero before filtering (for better comparison in plots).
    # https://stackoverflow.com/questions/69728320/setting-parameters-for-a-butterworth-filter
    #sig -= sig.mean()
    # State space with scipy's matrices
    filt_sig = np.array([])

    for sample in sig: 
        filt_sig = np.append(filt_sig, C@Xnn + D * sample)
        Xnn = A@Xnn + B * sample
    return filt_sig, Xnn

def pre_processing(segment,selected_electrodes_names ,filters, sample_duration, freq_limits_names, pipeline_type):
    curr_segment = segment
    outlier = 0
    # TODO add notch filter 50Hz for Unicorn BCI experiments??? --> is needed?

    # 1 OUTLIER DETECTION --> https://www.mdpi.com/1999-5903/13/5/103/html#B34-futureinternet-13-00103
    for i, j in curr_segment.iterrows():
        if stats.kurtosis(j) > 4*np.std(j) or (abs(j - np.mean(j)) > 125).any():
            if stats.kurtosis(j) > 4*np.std(j):
                print('due to kurtosis')
            outlier +=1
    # 2 APPLY COMMON AVERAGE REFERENCE (CAR) per segment only for deep learning pipeline   
    #CAR doesnt work for csp, for riemann gives worse results, so only use for deep
    if 'deep' in pipeline_type: 
        curr_segment -= curr_segment.mean()
    # 3 FILTERING filter bank / bandpass
    segment_filt, filters = filter_1seg_statespace(curr_segment, selected_electrodes_names, filters, sample_duration, 
    freq_limits_names)

    return segment_filt, outlier, filters

def filter_1seg_statespace(segment, selected_electrodes_names,filters, sample_duration, freq_limits_names):
    # filters dataframe with 1 segment of 1 sec for all given filters
    # returns a dataframe with columns as electrode-filters
    filter_results = {}
    segment = segment.transpose()
    for electrode in range(len(selected_electrodes_names)):
        for f in range(len(filters)):
            A, B, C, D, Xnn = filters[f] 
            filter_results[selected_electrodes_names[electrode] + '_' + freq_limits_names[f]] = []
            if segment.shape[0] == sample_duration:      
                # apply filter ánd update Xnn state vector       
                filt_result_temp, Xnn[0,electrode] = apply_filter_statespace(segment[selected_electrodes_names[electrode]], 
                A, B, C, D, Xnn[0,electrode])         
                for data_point in filt_result_temp:
                    filter_results[selected_electrodes_names[electrode] + '_' + freq_limits_names[f]].append(data_point) 
            filters[f] = [A, B, C, D, Xnn]
    filtered_dataset = pd.DataFrame.from_dict(filter_results).transpose()    
    return filtered_dataset, filters
    
def segmentation_all(dataset,sample_duration):
    segments = []
    labels = []
    true = 0
    false = 0
    dataset_c = copy.deepcopy(dataset)
    for _, segment in dataset_c.groupby(np.arange(len(dataset)) // sample_duration):
        segments.append(segment.iloc[:,:-1].transpose())
        labels.append(segment['label'].mode()) 
        if segment['label'].mode()[0] == 1:
            true +=1
        else:
            false +=1
    # print(f'true:{true}, false: {false}')
    return segments, labels

def segmentation_overlap_withfilt(dataset, sample_duration, filters, selected_electrodes_names, freq_limits_names, 
    pipeline_type, window_hop=100):
    segments = []
    labels = []
    dataset_c = copy.deepcopy(dataset)
    true = 0
    false = 0
    i = 0
    outliers = 0
    for frame_idx in range(sample_duration, dataset_c.shape[0], window_hop):
        if i == 0:
            temp_dataset = dataset_c.iloc[frame_idx-sample_duration:frame_idx, :-1].transpose()   
            # here apply filtering
            segment_filt, outlier, filters = pre_processing(temp_dataset, selected_electrodes_names, filters, 
                        sample_duration, freq_limits_names, pipeline_type)
        else:
            # here only get 0.5 seconds extra, filter only that part, than concat with new part
            temp_dataset = dataset_c.iloc[frame_idx-window_hop:frame_idx, :-1].transpose()  
            # here apply filtering   
            segment_filt_new, outlier, filters = pre_processing(temp_dataset, selected_electrodes_names, filters, 
                        window_hop, freq_limits_names, pipeline_type)
            if window_hop == sample_duration:
                segment_filt = segment_filt_new
            else:
                segment_filt = pd.concat([segment_filt.iloc[:,-(sample_duration-window_hop):], segment_filt_new], axis=1, 
                ignore_index=True)
            #plt.plot(np.arange(100,300), segment_filt.iloc[5, :])
            #plt.show()
        if outlier > 0 or i == 0: 
            #when i ==0, filters are initiated so signal is destroyed. Dont use.
            print(f'A segment was considered as an outlier due to bad signal in {outlier} channels')
            outliers +=1
        else:
            label_row = dataset_c.iloc[frame_idx-sample_duration:frame_idx, -1]
            label = label_row.sum() / len(label_row)
            #print(f'shape of segment filtered: {segment_filt.shape}')
            # transition from MI state to relax --> signal timestamp differs so don't use that segment as well.
            if (label == 1 or label == 0) and segment_filt.shape[1] == sample_duration:
                segments.append(segment_filt)
                labels.append(label) 
                if labels[-1] == 1:
                    true +=1
                elif labels[-1] == 0:
                    false +=1
        i += 1
    print(f'true:{true}, false: {false}')
    return segments, labels

def init_pipelines(pipeline_name = ['csp']):
    pipelines = {}
    if 'csp' in pipeline_name:
    # standard / Laura's approach + variations
        for n_comp in [10, 12]:
            #pipelines["csp+lda_n" + str(n_comp)] = Pipeline(steps=[('csp', CSP(n_components=n_comp)), 
             #                               ('lda', LDA())])
                                            
            # Using Ledoit-Wolf lemma shrinkage, which is said to outperform standard LDA
            pipelines["csp+s_lda_n" + str(n_comp)] = Pipeline(steps=[('csp', CSP(n_components=n_comp)), 
                                            ('slda', LDA(solver = 'lsqr', shrinkage='auto'))])
            
            pipelines["csp+svm_n" + str(n_comp)] = Pipeline(steps=[('csp', CSP(n_components=n_comp)), 
                                                ('svm', SVC(gamma = 0.05, C=10))])

            pipelines["csp+rf_n" + str(n_comp)] = Pipeline(steps=[('csp', CSP(n_components=n_comp)), 
                                                ('rf', RFC(random_state=42))])
    
    # Riemannian approaches
    if 'riemann' in pipeline_name:
        pipelines["tgsp+svm"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                            ('tg', TangentSpace(metric="riemann")),
                                            ('svm', SVC(gamma = 0.05, C=10))])
                                
        pipelines["tgsp+rf"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                            ('tg', TangentSpace(metric="riemann")),
                                            ('rf', RFC(random_state=42))])
        
        # Minimum distance to riemanian mean (MDRM) --> directly on the manifold --> parameter-free!
        pipelines["mdm"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                    ('mdm', MDM(metric="riemann"))])
        pipelines["fgmdm"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                    ('mdm', FgMDM(metric="riemann"))])

    return pipelines

def init_pipelines_grid(pipeline_name = ['csp']):
    pipelines = {}
    if 'csp' in pipeline_name:
        pipelines["csp_12+s_lda_n"] = Pipeline(steps=[('csp', CSP(n_components=12)), 
                                            ('slda', LDA(solver = 'lsqr', shrinkage='auto'))])

        pipe = Pipeline(steps=[('csp', CSP()), ('svm', SVC())])
        param_grid = {
            "csp__n_components" :[10,12],
            "svm__C": [1, 10],
            "svm__gamma": [0.1, 0.01, 0.001]
                }
        pipelines["csp+svm"] = GridSearchCV(pipe, param_grid, cv=4, scoring='accuracy',n_jobs=-1)

        pipe = Pipeline(steps=[('csp', CSP(n_components=12)), ('rf', RFC(random_state=42))])
        param_grid = {
            "rf__min_samples_leaf": [1, 2],
            "rf__n_estimators": [50, 100, 200],
            "rf__criterion": ['gini', 'entropy']}
        pipelines["csp+rf"] = GridSearchCV(pipe, param_grid, cv=4, scoring='accuracy',n_jobs=-1)

    if 'riemann' in pipeline_name:  
        pipelines["fgmdm"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                    ('mdm', FgMDM(metric="riemann"))])

        pipelines["tgsp+slda"] = Pipeline(steps=[('cov', Covariances("oas")), 
                ('tg', TangentSpace(metric="riemann")),
                ('slda', LDA(solver = 'lsqr', shrinkage='auto'))])   

        pipe = Pipeline(steps=[('cov', Covariances("oas")), 
                                            ('tg', TangentSpace(metric="riemann")),
                                            ('svm', SVC())])
        param_grid = {"svm__C": [0.1, 1, 10, 100],
            "svm__gamma": [0.1, 0.01, 0.001],
            "svm__kernel": ['rbf', 'linear']
                }
        pipelines["tgsp+svm"] = GridSearchCV(pipe, param_grid, cv=4, scoring='accuracy',n_jobs=-1)  

        pipe = Pipeline(steps=[('cov', Covariances("oas")), 
                                            ('tg', TangentSpace(metric="riemann")),
                                            ('rf', RFC(random_state=42))])
        param_grid = {"rf__min_samples_leaf": [1, 2, 50, 100],
            "rf__n_estimators": [10, 50, 100, 200],
            "rf__criterion": ['gini', 'entropy']}
        pipelines["tgsp+rf"] = GridSearchCV(pipe, param_grid, cv=4, scoring='accuracy',n_jobs=-1)

    return pipelines  

def grid_search_execution(X_train, y_train, X_val, y_val, chosen_pipelines, clf):
    start_time = time.time()
    preds = np.zeros(len(y_val))
    chosen_pipelines[clf].fit(X_train, y_train)
    preds = chosen_pipelines[clf].predict(X_val)

    acc = np.mean(preds == y_val)
    f1 = f1_score(y_val, preds)
    acc_classes = confusion_matrix(y_val, preds, normalize="true").diagonal()
    print(f"Classification accuracy: {acc} and per class: {acc_classes}")
    elapsed_time = time.time() - start_time
    
    return acc, acc_classes, f1, elapsed_time, chosen_pipelines

def plot_dataset(data_table, columns, match='like', display='line'):
    names = list(data_table.columns)

    # Create subplots if more columns are specified.
    if len(columns) > 1:
        f, xar = plt.subplots(len(columns), sharex=True, sharey=False)
    else:
        f, xar = plt.subplots()
        xar = [xar]

    f.subplots_adjust(hspace=0.4)

    # Pass through the columns specified.
    for i in range(0, len(columns)):
        xar[i].set_prop_cycle(color=['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        # if a column match is specified as 'exact', select the column name(s) with an exact match.
        # If it's specified as 'like', select columns containing the name.

        # We can match exact (i.e. a columns name is an exact name of a columns or 'like' for
        # which we need to find columns names in the dataset that contain the name.
        if match[i] == 'exact':
            relevant_cols = [columns[i]]
        elif match[i] == 'like':
            relevant_cols = [name for name in names if columns[i] == name[0:len(columns[i])]]
        else:
            raise ValueError("Match should be 'exact' or 'like' for " + str(i) + ".")

        max_values = []
        min_values = []

        point_displays = ['+', 'x'] #'*', 'd', 'o', 's', '<', '>']
        line_displays = ['-'] #, '--', ':', '-.']
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']


        # Pass through the relevant columns.
        for j in range(0, len(relevant_cols)):
        
            # Create a mask to ignore the NaN and Inf values when plotting:
            mask = data_table[relevant_cols[j]].replace([np.inf, -np.inf], np.nan).notnull()
            max_values.append(data_table[relevant_cols[j]][mask].max())
            min_values.append(data_table[relevant_cols[j]][mask].min())

            # Display point, or as a line
            if display[i] == 'points':
                xar[i].plot(data_table.index[mask], data_table[relevant_cols[j]][mask],
                            point_displays[j%len(point_displays)])
            else:
                xar[i].plot(data_table.index[mask], data_table[relevant_cols[j]][mask],
                            line_displays[j%len(line_displays)])

        xar[i].tick_params(axis='y', labelsize=10)
        xar[i].legend(relevant_cols, fontsize='xx-small', numpoints=1, loc='upper center',
                        bbox_to_anchor=(0.5, 1.3), ncol=len(relevant_cols), fancybox=True, shadow=True)

        xar[i].set_ylim([min(min_values) - 0.1*(max(max_values) - min(min_values)),
                            max(max_values) + 0.1*(max(max_values) - min(min_values))])

    # Make sure we get a nice figure with only a single x-axis and labels there.
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    plt.xlabel('time')
    plt.show()