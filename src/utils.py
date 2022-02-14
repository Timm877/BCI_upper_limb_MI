import copy
import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from mne.decoding import CSP
from pyriemann.classification import MDM, FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def init_filters(freq_lim, sample_freq, filt_type = 'bandpass', order=2):
    filters = []
    for f in range(freq_lim.shape[0]):
        b, a = init_filt_coef(freq_lim[f], fs=sample_freq, filtype=filt_type, order=order)
        filters.append([b, a])
    return filters

def init_filt_coef(cuttoff, fs, filtype, order):
    if filtype == 'lowpass':
        b,a = signal.butter(order, cuttoff[0]/(fs/2), filtype)
    elif filtype == 'bandpass':
        b,a = signal.butter(order, [cuttoff[0]/(fs/2), cuttoff[1]/(fs/2)], filtype)
    elif filtype == 'highpass':
        b,a = signal.butter(order, cuttoff[0]/(fs/2), filtype)
    return b, a 

def apply_filter(sig, b, a):
    # Substract mean and then scaling of signal to 0-1, as original signal is in range of -14000
    # as the signal has to be centered at zero before filtering.
    # https://stackoverflow.com/questions/69728320/setting-parameters-for-a-butterworth-filter
    sig -= sig.mean()
    sig = min_max_scale(sig)
    return signal.filtfilt(b, a, sig)

def min_max_scale(x):
    x -= x.min()
    x = x/x.max()
    return x

def init_pipelines(pipeline_name = ['csp'], n_components = 8):
    pipelines = {}
    if 'csp' in pipeline_name:
    # standard / Laura's approach + variations
        for n_comp in [8, 10, 11, 12, 13]:
            pipelines["csp+lda_n" + str(n_comp)] = Pipeline(steps=[('csp', CSP(n_components=n_comp)), 
                                            ('lda', LDA())])
                                            
            # Using Ledoit-Wolf lemma shrinkage, which is said to outperform standard LDA
            pipelines["csp+s_lda_n" + str(n_comp)] = Pipeline(steps=[('csp', CSP(n_components=n_comp)), 
                                            ('slda', LDA(solver = 'lsqr', shrinkage='auto'))])
            
            pipelines["csp+svm_n" + str(n_comp)] = Pipeline(steps=[('csp', CSP(n_components=n_comp)), 
                                                ('svm', SVC(kernel='linear'))])

            pipelines["csp+rf_n" + str(n_comp)] = Pipeline(steps=[('csp', CSP(n_components=n_comp)), 
                                                ('rf', RFC(random_state=42))])
    
    # Riemannian approaches
    if 'riemann' in pipeline_name:
        pipelines["tgsp+svm"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                            ('tg', TangentSpace(metric="riemann")),
                                            ('svm', SVC(kernel='linear'))])
                                            
        pipelines["tgsp+rf"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                            ('tg', TangentSpace(metric="riemann")),
                                            ('rf', RFC(random_state=42))])
        
        # Minimum distance to riemanian mean (MDRM) --> directly on the manifold --> parameter-free!
        pipelines["mdm"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                    ('mdm', MDM(metric="riemann"))])
        pipelines["fgmdm"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                    ('mdm', FgMDM(metric="riemann"))])

    if 'deep' in pipeline_name:
        #TODO, for now just pass 1 random
        pipelines["fgmdm"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                    ('mdm', FgMDM(metric="riemann"))])


    return pipelines

def segmentation_and_filter(dataset, selected_electrodes_names,filters, sample_duration, freq_limits_names):
    filter_results = {}
    for electrode in selected_electrodes_names:
        for f in range(len(filters)):
            b, a = filters[f] 
            filter_results[electrode + '_' + freq_limits_names[f]] = []
            for _, segment in dataset[electrode].groupby(np.arange(len(dataset)) // sample_duration):
                if segment.shape[0] == sample_duration:                     
                    filt_result_relax = apply_filter(segment,b,a)                     
                    for data_point in filt_result_relax:
                        filter_results[electrode + '_' + freq_limits_names[f]].append(data_point)      
    filtered_dataset = pd.DataFrame.from_dict(filter_results)        
    return filtered_dataset

def filter_1seg(segment, selected_electrodes_names,filters, sample_duration, freq_limits_names):
    # filters dataframe with 1 segment of 1 sec for all given filters
    # returns a dataframe with as electrodexfilters columns
    filter_results = {}
    for electrode in selected_electrodes_names:
        for f in range(len(filters)):
            b, a = filters[f] 
            filter_results[electrode + '_' + freq_limits_names[f]] = []
            if segment.shape[0] == sample_duration: 
                #print('segment\n')
                #print(segment)                    
                filt_result_temp = apply_filter(segment[electrode],b,a)
                #print('filtresult\n')
                #print(filt_result_temp)                     
                for data_point in filt_result_temp:
                    filter_results[electrode + '_' + freq_limits_names[f]].append(data_point) 
                #print('end result\n')  
                #print(filter_results[electrode + '_' + freq_limits_names[f]])   
    filtered_dataset = pd.DataFrame.from_dict(filter_results)        
    return filtered_dataset

def segmentation_for_ML(dataset,sample_duration):
    segments = []
    labels = []
    dataset_c = copy.deepcopy(dataset)
    for _, segment in dataset_c.groupby(np.arange(len(dataset)) // sample_duration):
        segments.append(segment.iloc[:,:-1].transpose())
        labels.append(segment['label'].mode()) 
    return np.stack(segments), np.array(labels).ravel()

def segmentation_all(dataset,sample_duration):
    segments = []
    labels = []
    dataset_c = copy.deepcopy(dataset)
    for _, segment in dataset_c.groupby(np.arange(len(dataset)) // sample_duration):
        segments.append(segment.iloc[:,:-1].transpose())
        labels.append(segment['label'].mode()) 
    return segments, labels

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
