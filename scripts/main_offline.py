import os
from pathlib import Path
from src.utils import init_filters, apply_filter, create_pipeline
import argparse
import numpy as np

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
     '''
     # 1. get files
     folder_path = Path('./data/offline/')
     result_path = Path('./data/offline/intermediate_datafiles/1/')
     result_path.mkdir(exist_ok=True, parents=True)
     
     for instance in os.scandir(folder_path): # go through all data files  
          # when having only 1 file this is not needed
          sig = instance.path
     '''

     # 2. INIT
     sampling_frequency = 200 # 250 for ours, 200 for Laura's
     sample_duration = 100 # 0.5 seconds
     selected_electrodes_names= ['F3', 'FZ', 'FC1','FCZ','C1','CZ','CP1','CPZ', 
     'FC5', 'FC3','C5','C3','CP5','CP3','P3','PZ','F4','FC2','FC4','FC6','C2',
     'C4','CP2','CP4','C6','CP6','P4','HR' ,'HL', 'VU', 'VD']
     # for us: ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
     n_electrodes = len(selected_electrodes_names) #31 for Laura's, 8 for ours

     # initialize filters
     freq_limits = np.asarray([[5, 10], [10, 15], [15, 20], [20, 25]]) 
     filters = init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=2)


     # 3. SPLIT DATA 
     # TODO: add a function to split data in windows of 0.5 seconds / 100 datapoints
     # pd.rolling window?


     # 4. apply filter bank (FB)
     '''
     for electrode in range(n_electrodes):
          for f in range(len(filters)):
               b, a = filters[f]
               sig_filtered = apply_filter(sig, b, a)
     '''

     # 5. create pipeline by parser args.
     # for now only do LDA and csp
     pipeline = create_pipeline(FLAGS.p)

     # 6. apply pipeline
     # TODO

     # 7.be excited
     # :)
     
     # 8. some prints
     print_flags()
     print('\n')
     print(filters)
     print('\n')
     print(pipeline)
     


if __name__ == '__main__':
     parser = argparse.ArgumentParser(description="Run offline BCI analysis")
     parser.add_argument("--p", type=str, default='CSP+LDA', help="The spatial filter and ML model pipeline used.")

     FLAGS, unparsed = parser.parse_known_args()
     main()
