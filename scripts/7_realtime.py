import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import src.unicorn_utils as utils
pd.options.mode.chained_assignment = None  # default='warn'

def execution(subject):
    sampling_frequency = 250
    electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    folder_path = Path(f'./data/openloop/{subject}/openloop')
    result_path = Path(f'./data/openloop/intermediate_datafiles/preprocess/closedloop')
    result_path.mkdir(exist_ok=True, parents=True)  
    dataset_full = {}
    trials_amount = 0

    filt_ord = 2
    window_size = 500
    freq_limits = np.asarray([[1,100]]) 
    freq_limits_names = ['1_100Hz']

    filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filt_ord)
    for instance in os.scandir(folder_path):
        if instance.path.endswith('.csv'): 
            trials_amount +=1
            print(f'adding_{instance} to dataset...')
            sig = pd.read_csv(instance.path)
            X = sig.loc[:,electrode_names]
            y = sig.loc[:,'Class']
            dataset_full = pd.concat([X, y], axis=1)

            #start data collection
            #data_segment = add datapoints
            # every 0.5 seconds:
            # TODO change function below
            X_temp, y_temp = utils.realtime_filt(dataset_full, window_size, filters,
                            electrode_names, freq_limits_names, sampling_frequency, subject)
            # print time
            #if seg == 0:
            # add 4x 0.5sec together -->TODO plot data to see if overlap is good
            #else: 
            #remove first 0.5, add new 0.5
            # if y_temp == 0 or 1 or 2:
                # if outlier: predict nothing
                # else: predict
                # ball to left, ball to right, or circle bigger (relax)
                # if predict == y_temp: save data segment to data-dict for further training


def main():
    list_of_freq_lim = [[[1,100]]]
    freq_limits_names_list = [['1_100Hz']]
    filt_orders = [2]
    window_sizes = [500]
    execution('deep', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes, FLAGS.subj[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--subjects", nargs='+', default=['X02_wet'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are in the data folder.")
    FLAGS, unparsed = parser.parse_known_args()
    main()