import copy
import time
from collections import Counter
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from pathlib import Path


def init_filters(freq_lim, sample_freq, filt_type = 'bandpass', order=2, state_space=True):
    filters = []
    for f in range(freq_lim.shape[0]):
        A, B, C, D, Xnn  = init_filt_coef_statespace(freq_lim[f], fs=sample_freq, filtype=filt_type, order=order)
        filters.append([A, B, C, D, Xnn ])
    return filters

def init_filt_coef_statespace(cuttoff, fs, filtype, order,len_selected_electrodes=8):
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
    # State space with scipy's matrices
    filt_sig = np.array([])
    for sample in sig: 
        filt_sig = np.append(filt_sig, C@Xnn + D * sample)
        Xnn = A@Xnn + B * sample

    #b,a = signal.butter(4, [8/(250/2), 35/(250/2)], 'bandpass')
    #filt_sig = signal.filtfilt(b, a, sig)
    '''
    plt.plot(filt_sig)
    plt.plot(filt_sig2)
    plt.ylim(ymax = 50, ymin = -50)
    plt.title('EEG signal FZ - 3rd segment - bandpass filtered 8 to 35Hz, order 8')
    plt.xlabel('Datapoints of 0.5 seconds, captured with 250Hz')
    plt.ylabel('EEG signal (\u03BCV)')
    plt.legend(['State space filtered signal', 'Butter filtfilt filtered signal'])
    plt.show()
    '''
    return filt_sig, Xnn

def pre_processing(curr_segment,selected_electrodes_names ,filters, sample_duration, freq_limits_names, pipeline_type, 
                    sampling_frequency=250):
    outlier = 0

    # 1. notch filt
    b_notch, a_notch = signal.iirnotch(50, 30, sampling_frequency)
    for column in curr_segment.columns:
        curr_segment.loc[:,column] = signal.filtfilt(b_notch, a_notch, curr_segment.loc[:,column])
    curr_segment = curr_segment.T

    # 2 OUTLIER DETECTION
    for i, j in curr_segment.iterrows():
        if stats.kurtosis(j) > 4*np.std(j) or (abs(j - np.mean(j)) > 125).any():
            if stats.kurtosis(j) > 4*np.std(j):
                print('due to kurtosis')
            outlier +=1
    
    # 3 APPLY COMMON AVERAGE REFERENCE (CAR) per segment only for deep learning pipeline   
    # CAR doesnt work for csp and riemann, so only use for deep
    if 'deep' in pipeline_type: 
        curr_segment -= curr_segment.mean()
    #plot_dataset(curr_segment.T, ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8'],
    #                              ['like', 'like', 'like','like', 'like', 'like','like', 'like'],
    #                              ['line','line', 'line','line','line', 'line','line','line'])

    # 4 FILTERING filter bank / bandpass
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

def unicorn_segmentation_overlap_withfilt(dataset, sample_duration, filters, selected_electrodes_names, freq_limits_names, 
    pipeline_type, sampling_frequency, subject=None):
    window_hop = sampling_frequency//2
    segments = []
    labels = []
    outlier_labels = []
    dataset_c = copy.deepcopy(dataset)
    i = 0
    outliers = 0
    for frame_idx in range(sample_duration, dataset_c.shape[0], window_hop):
        if i == 0:
            temp_dataset = dataset_c.iloc[frame_idx-sample_duration:frame_idx, :-1] 
            # here apply filtering
            segment_filt, outlier, filters = pre_processing(temp_dataset, selected_electrodes_names, filters, 
                        sample_duration, freq_limits_names, pipeline_type, sampling_frequency)
            
        else:
            # here only get 0.5 seconds extra, filter only that part, than concat with new part
            temp_dataset = dataset_c.iloc[frame_idx-window_hop:frame_idx, :-1] 
            # here apply filtering   
            segment_filt_new, outlier, filters = pre_processing(temp_dataset, selected_electrodes_names, filters, 
                        window_hop, freq_limits_names, pipeline_type, sampling_frequency)
            if window_hop == sample_duration:
                segment_filt = segment_filt_new
            else:
                segment_filt = pd.concat([segment_filt.iloc[:,-(sample_duration-window_hop):].reset_index(drop=True), 
                segment_filt_new.reset_index(drop=True)], axis=1, ignore_index=True)

        if outlier > 0 or i == 0: 
            print(f'A segment was considered as an outlier due to bad signal in {outlier} channels')
            outliers +=1
            label_row = dataset_c.iloc[frame_idx-sample_duration:frame_idx, -1]
            label = label_row.value_counts()[:1]
            if (label[0] == sample_duration) and (label.index.tolist()[0] in ['0', '1','2', '3']):
                # 0 relax 1 right arm 2 left arm 3 legs
                outlier_labels.append(int(label.index.tolist()[0])) 
            #if outlier > 3:
                #print(segment_filt.head())
                #segment_filt = segment_filt.T
                #segment_filt.columns = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
                #plot_dataset(segment_filt, ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8'],
                #                  ['like', 'like', 'like','like', 'like', 'like','like', 'like'],
                #                  ['line','line', 'line','line','line', 'line','line','line'])
            
        else:
            label_row = dataset_c.iloc[frame_idx-sample_duration:frame_idx, -1]
            label = label_row.value_counts()[:1]
            if (label[0] == sample_duration) and (label.index.tolist()[0] in ['0' ,'1','2', '3']):
                # 0 relax 1 right arm 2 left arm 3 legs
                segments.append(segment_filt)
                labels.append(int(label.index.tolist()[0])) 
        i += 1
    label_amounts = Counter(labels)
    outlier_amounts = Counter(outlier_labels)
    print(f'amount of good segments: {len(labels)}')
    print(f"Good - relax: {label_amounts[0]}, rightarm: {label_amounts[1]}, leftarm: {label_amounts[2]},\
         legs: {label_amounts[3]}")
    print(f"Outliers - relax: {outlier_amounts[0]}, rightarm: {outlier_amounts[1]}, leftarm: {outlier_amounts[2]},\
    legs: {outlier_amounts[3]}")

    # save output:
    path_outlier = Path(f"./outliers/{subject}_openloop")
    path_outlier.mkdir(exist_ok=True, parents=True) 
    print(f"Good - relax: {label_amounts[0]}, rightarm: {label_amounts[1]}, leftarm: {label_amounts[2]}, \
        legs: {label_amounts[3]}", 
    file=open(f"outliers/{subject}_openloop/{pipeline_type}_{freq_limits_names}_outliers.txt", "a"))
    print(f"Outliers - relax: {outlier_amounts[0]}, rightarm: {outlier_amounts[1]}, leftarm: {outlier_amounts[2]}, \
        legs: {outlier_amounts[3]}", 
    file=open(f"outliers/{subject}_openloop/{pipeline_type}_{freq_limits_names}_outliers.txt", "a"))

    return segments, labels

def init_pipelines_grid(pipeline_name = ['csp']):
    pipelines = {}
    if 'csp' in pipeline_name:
        pipe= Pipeline(steps=[('csp', CSP()), 
                                            ('slda', LDA(solver = 'lsqr', shrinkage='auto'))])

        param_grid = {
            "csp__n_components" :[8,10,12]
                }
        pipelines["csp+s_lda"] = GridSearchCV(pipe, param_grid, cv=4, scoring='accuracy',n_jobs=-1)
        '''
        pipe = Pipeline(steps=[('csp', CSP()), ('svm', SVC(decision_function_shape='ovo'))])
        param_grid = {
            "csp__n_components" :[8,10,12],
            "svm__C": [1, 10],
            "svm__gamma": [0.1, 0.01, 0.001]
                }
        pipelines["csp+svm"] = GridSearchCV(pipe, param_grid, cv=4, scoring='accuracy',n_jobs=-1)

        pipe = Pipeline(steps=[('csp', CSP()), ('rf', RFC(random_state=42))])
        param_grid = {
            "csp__n_components" :[8,10,12],
            "rf__min_samples_leaf": [1, 2, 3],
            "rf__n_estimators": [50, 100, 200],
            "rf__criterion": ['gini', 'entropy']}
        pipelines["csp+rf"] = GridSearchCV(pipe, param_grid, cv=4, scoring='accuracy',n_jobs=-1)
        '''
    if 'riemann' in pipeline_name:  
        '''
        pipelines["fgmdm"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                    ('mdm', FgMDM(metric="riemann"))])

        pipelines["tgsp+slda"] = Pipeline(steps=[('cov', Covariances("oas")), 
                ('tg', TangentSpace(metric="riemann")),
                ('slda', LDA(solver = 'lsqr', shrinkage='auto'))]) 
        '''  
        pipe = Pipeline(steps=[('cov', Covariances("oas")), 
                                            ('tg', TangentSpace(metric="riemann")),
                                            ('svm', SVC(decision_function_shape='ovo'))])

        param_grid = {"svm__C": [0.1, 1, 10, 100],
            "svm__gamma": [0.1, 0.01, 0.001],
            "svm__kernel": ['rbf', 'linear']
                }
        pipelines["tgsp+svm"] = GridSearchCV(pipe, param_grid, cv=4, scoring='accuracy',n_jobs=-1)  
        '''
        pipe = Pipeline(steps=[('cov', Covariances("oas")), 
                                            ('tg', TangentSpace(metric="riemann")),
                                            ('rf', RFC(random_state=42))])
        param_grid = {"rf__min_samples_leaf": [1, 2, 50, 100],
            "rf__n_estimators": [10, 50, 100, 200],
            "rf__criterion": ['gini', 'entropy']}
        pipelines["tgsp+rf"] = GridSearchCV(pipe, param_grid, cv=4, scoring='accuracy',n_jobs=-1)
        '''
    return pipelines  

def grid_search_execution(X_train, y_train, X_val, y_val, chosen_pipelines, clf):
    start_time = time.time()
    chosen_pipelines[clf].fit(X_train, y_train)
    preds = chosen_pipelines[clf].predict(X_val)

    acc = np.mean(preds == y_val)
    f1 = f1_score(y_val, preds, average='macro')
    precision = precision_score(y_val, preds, average='macro')
    recall = recall_score(y_val, preds, average='macro')
    acc_classes = confusion_matrix(y_val, preds, normalize="true").diagonal()
    roc_auc = 0
    print(f"Classification accuracy: {acc} and per class: {acc_classes}")
    elapsed_time = time.time() - start_time
    return acc, precision, recall, roc_auc, acc_classes, f1, elapsed_time, chosen_pipelines

def plot_dataset(data_table, columns, match='like', display='line'):
    names = list(data_table.columns)
    # Create subplots if more columns are specified.
    if len(columns) > 1:
        f, xar = plt.subplots(len(columns), sharex=True, sharey=False)
    else:
        f, xar = plt.subplots()
        xar = [xar]
    f.subplots_adjust(hspace=0.4)
    f.suptitle(r'Example of EEG segment marked as outlier', fontsize=16)

    # Pass through the columns specified.
    for i in range(0, len(columns)):
        xar[i].set_prop_cycle(color=['b', 'r', 'k', 'y', 'm'])
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
                xar[i].set_xlabel('Datapoints (250Hz)')
                xar[i].set_ylabel('Label Value')
            else:
                xar[i].plot(data_table.index[mask], data_table[relevant_cols[j]][mask],
                            line_displays[j%len(line_displays)])
                #xar[i].set_xlabel('Datapoints (250Hz)')
                #xar[i].set_ylabel('EEG signal (\u03BCV)')

        xar[i].tick_params(axis='y', labelsize=10)
        xar[i].legend(relevant_cols, fontsize='x-small', numpoints=1, loc='upper center',
                        bbox_to_anchor=(0.5, 1.25), ncol=len(relevant_cols), fancybox=True, shadow=True)

        xar[i].set_ylim([min(min_values) - 0.1*(max(max_values) - min(min_values)),
                            max(max_values) + 0.1*(max(max_values) - min(min_values))])

    # Make sure we get a nice figure with only a single x-axis and labels there.
    f.text(0.5, 0.04, 'Datapoints (250Hz)', ha='center')
    f.text(0.04, 0.5, 'EEG signal (\u03BCV)', va='center', rotation='vertical')

    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)


    plt.show()