import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import src.realtime_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

def main():
    # INIT
    filt_ord = 2
    freq_limits = np.asarray([[1,100]]) 
    freq_limits_names = ['1_100Hz']
    sample_duration = 125
    sampling_frequency = 250
    subject = FLAGS.subjects[0]
    electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
    filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filt_ord)
    segments, labels, predictions = [], [], []
    total_outlier = 0

    # init DL model
    net = utils.EEGNET()
    net.load_state_dict(torch.load(f'finetuned_models/{subject}/EEGNET_trialnum4'))
    net = net.float()
    net.eval()

    folder_path = Path(f'./data/openloop/{subject}/openloop') 
    accuracies, elapsed_times = [], []
    counter = 0

    for instance in os.scandir(folder_path):
        sig = pd.read_csv(instance.path)
        X = sig.loc[:,electrode_names]
        y = sig.loc[:,'Class']
        dataset_full = pd.concat([X, y], axis=1)
        current_seg = pd.DataFrame()
        current_label = pd.DataFrame()
        
        for frame_id in range(sample_duration, dataset_full.shape[0] , sample_duration):
            start = time.time()
            counter +=1

            data_segment = dataset_full.iloc[frame_id-sample_duration:frame_id, :-1] 
            label = dataset_full.iloc[frame_id-sample_duration:frame_id, -1]
            segment_filt, outlier, filters = utils.pre_processing(data_segment, electrode_names, filters, 
                        sample_duration, freq_limits_names, sampling_frequency)

            if len(current_seg) == 0:
                current_seg = segment_filt
                current_label = label

            else:
                current_seg = pd.concat([current_seg.reset_index(drop=True), segment_filt.reset_index(drop=True)],
                    axis=1, ignore_index=True)
                current_label = pd.concat([current_label.reset_index(drop=True), label.reset_index(drop=True)],
                    axis=0, ignore_index=True)

            if len(current_seg.columns) == 500:
                # only when we have 2 second of data, move on
                current_label = current_label.T
                label_count = current_label.value_counts()[:1]
                label_num = label_count.index.tolist()[0]

                if (label_count[0] == 500) and (label_num in ['0', '1', '2']):
                    # 0 relax 1 right arm 2 left arm
                    if outlier > 0:
                        total_outlier +=1
                        print('OUTLIER')
                    else:
                        segments.append(current_seg)
                        labels.append(int(label_num)) 

                        current_tensor = torch.from_numpy(np.array(current_seg))
                        current_tensor = current_tensor[np.newaxis, np.newaxis, :, :]

                        output = net(current_tensor.float())
                        _, predicted = torch.max(output.data, 1)
                        #print(f'PREDICTED CLASS: {predicted[0]}')
                        predictions.append(int(predicted[0]))
                else:
                    pass
                    #print("Not a valid MI segment")

                # TODO save data here to CSV before deleting first 0.5 sec
                # not so difficult with pd.save_csv or smthng

                # lastly delete first 0.5 sec to later add 0.5 sec more
                current_seg = current_seg.iloc[:,125:]
                current_label = current_label.iloc[125:]
                # and calculate elapsed time
                elapsed_times.append(time.time() - start)

        print(f"amount of outliers during MI: {total_outlier}")
        print(f"accuracy: {(sum([x==y for (x,y) in zip(predictions,labels)]))/len(labels)}")
        accuracies.append((sum([x==y for (x,y) in zip(predictions,labels)]))/len(labels))

    print(f"average elapsed time: {sum(elapsed_times) / len(elapsed_times)}")
    print(accuracies)    
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--subjects", nargs='+', default=['X02'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are in the data folder.")
    FLAGS, unparsed = parser.parse_known_args()
    main()