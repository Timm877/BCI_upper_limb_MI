import os
from pathlib import Path
import src.utils as utils
import argparse
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
     # 1. get files
     folder_path = Path('./data/offline/')
     result_path = Path('./data/offline/intermediate_datafiles/1/')
     result_path.mkdir(exist_ok=True, parents=True)
     
     for instance in os.scandir(folder_path): # go through all data files  
          # when having only 1 file this is not needed
          if instance.path.endswith('.mat'):
               sig = scipy.io.loadmat(instance.path)
               X = sig['session']['data_EEG']
               y = sig['session']['task_EEG']

               #print(X[0][0].shape) # the 31+1 electrodes as rows, the 36750 columns as samples, 36750 / 200 = +- 3 minutes of data
               X = X[0][0][:-1,:].T #get relevant data and transpose
               #print(X.shape)
               #print(y[0][0].shape) # and for each sample a label is available
               y = y[0][0].T #get relevant data and transpose
               #print(y.shape)
     # 2. INIT
     sampling_frequency = 200 # 250 for ours, 200 for Laura's
     sample_duration = 200 # 1 second
     selected_electrodes_names= ['F3', 'FZ', 'FC1','FCZ','C1','CZ','CP1','CPZ', 
     'FC5', 'FC3','C5','C3','CP5','CP3','P3','PZ','F4','FC2','FC4','FC6','C2',
     'C4','CP2','CP4','C6','CP6','P4','HR' ,'HL', 'VU', 'VD']
     # for us: ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
     n_electrodes = len(selected_electrodes_names) #31 for Laura's, 8 for ours

     # transform np array to pandas dataframe
     dataset = pd.DataFrame(X, columns=selected_electrodes_names)
     labels = pd.DataFrame(y, columns=['label'])
     dataset = dataset.join(labels)
     data_relax = dataset.loc[labels['label'] == 402] #reset index?
     data_MI = dataset.loc[labels['label'] == 404]

     # initialize filters
     freq_limits = np.asarray([[5, 10], [10, 15], [15, 20], [20, 25]]) 
     freq_limits_names = ['5_10Hz', '10_15Hz','15_20Hz','20_25Hz']
     filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=2)



     # 3. split data into segments
     # 4. apply filter bank (FB) 
     

     
     # 5. init pipeline by parser args.
     chosen_pipelines = utils.init_pipelines(FLAGS.p)
     for pipeline in chosen_pipelines:
          #print(chosen_pipelines[pipeline])
          pass
               
     '''
     print(dataset[:203])
     print(dataset.columns)
     # 8. some prints
     print_flags()
     print('\n')
     #print(filters)
     #print('\n')
     print(chosen_pipelines)
     '''


if __name__ == '__main__':
     parser = argparse.ArgumentParser(description="Run offline BCI analysis")
     parser.add_argument("--p", type=list, default=['csp+lda'], help="The pipeline used. \
     This variable is a list containing all names of selected pipelines. Options currently are: ")

     FLAGS, unparsed = parser.parse_known_args()
     main()
