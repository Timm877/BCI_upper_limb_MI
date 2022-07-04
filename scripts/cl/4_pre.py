import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import src.utils_preprocess as utils
pd.options.mode.chained_assignment = None  # default='warn'

def main():
    for subj in FLAGS.subjects:
        print(subj)
        if 'csp' in FLAGS.pline:
            # filterbank
            list_of_freq_lim = [[[5, 10], [10, 15], [15, 20], [20, 25]]]
            freq_limits_names_list = [['10_15Hz','15_20Hz','20_25Hz', '25_30Hz', '30_35Hz'],]
            filt_orders = [2]
            window_sizes = [500]
            execution('csp', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes, subj, FLAGS.session)
        if 'deep' in FLAGS.pline:
            list_of_freq_lim = [[[1, 100]]]
            freq_limits_names_list = [['1_100Hz']]
            filt_orders = [2]
            window_sizes = [500]
            execution('deep', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes, subj, FLAGS.session)
        if 'riemann' in FLAGS.pline:
            list_of_freq_lim = [[[8, 35]]]
            freq_limits_names_list = [['8_35Hz']]
            filt_orders = [2]
            window_sizes = [500]
            execution('riemann', list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes, subj, FLAGS.session)

def execution(pipeline_type, list_of_freq_lim, freq_limits_names_list, filt_orders, window_sizes, subject, session):
    print(f'Preprocessing for {pipeline_type} experimentation...')

    # INIT
    sampling_frequency = 250 
    electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    folder_path = Path(f'./scripts/cl\Expdata\Subjects\wet\{subject}\{session}\openloop')
    result_path = Path(f'./scripts/cl\intermediate_files/{session}_CSP')
    result_path.mkdir(exist_ok=True, parents=True)  
    dataset_full = {}

    for instance in os.scandir(folder_path):
        if instance.path.endswith('.csv'): 
            print(f'adding_{instance} to dataset...')
            sig = pd.read_csv(instance.path)
            X = sig.loc[:, electrode_names]
            y = sig.loc[:, 'Class']
            dataset_full[str(instance)] = pd.concat([X, y], axis=1)
    
    for window_size in window_sizes:
        for filt_ord in filt_orders:
            for freq_limit_instance in range(len(list_of_freq_lim)):
                freq_limits = np.asarray(list_of_freq_lim[freq_limit_instance]) 
                freq_limits_names = freq_limits_names_list[freq_limit_instance]
                filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filt_ord)

                print(f'experimenting with filter order of {filt_ord}, freq limits of {freq_limits_names}, \
                        and ws of {window_size}.')
                
                data_dict = {'data' : {}, 'labels': {}}
                df_num = 0
                for df in dataset_full:
                    print(f'current df: {df}')
                    data_dict['data'][df_num], data_dict['labels'][df_num] = [], []
                    X_temp, y_temp = utils.unicorn_segmentation_overlap_withfilt(dataset_full[df], window_size, filters,
                    electrode_names, freq_limits_names, pipeline_type, sampling_frequency, subject)
                    for segment in range(len(X_temp)): 
                        data_dict['data'][df_num].append(X_temp[segment])
                        data_dict['labels'][df_num].append(y_temp[segment]) 
                    df_num += 1

                results_fname = f'{subject}_{pipeline_type}.pkl'
                save_file = open(result_path / results_fname, "wb")
                pickle.dump(data_dict, save_file)
                save_file.close()
                print('Finished a preprocess pipeline.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--pline", nargs='+', default=['deep'], help="The variant of pipelines used for after preprocessing. \
    This variable is a list containing the name of the variants. Options are: 'csp', 'riemann', 'deep'")
    parser.add_argument("--subjects", nargs='+', default=['X02'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are in the data folder.")
    parser.add_argument("--session", type=str, default='session4', help="The session number of closed loop experiments.")
    FLAGS, unparsed = parser.parse_known_args()
    main()