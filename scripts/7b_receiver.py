import numpy as np
import pandas as pd
import src.realtime_utils as utils
import torch
from pylsl import StreamInlet, resolve_stream

# INIT
filt_ord = 2
freq_limits = np.asarray([[1,100]]) 
freq_limits_names = ['1_100Hz']
sample_duration = 125
sampling_frequency = 250
subject = 'X01'
electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filt_ord)
segments, labels, predictions = [], [], []
total_outlier = 0

# init DL model
net = utils.EEGNET()
net.load_state_dict(torch.load(f'final_models/best_finetune\EEGNET_Finetuned_{subject}', map_location=torch.device('cpu')))
net = net.float()
net.eval()

# below code is for initializing the streaming layer which will help us capture data later
finished = False
streams = resolve_stream()
inlet = StreamInlet(streams[0])
sig_tot = ''
i = 0

def update_data(data,res):
    i = 0
    for key in list(data.keys()):
        data[key].append(res[i])
        i = i +1
    return data

def concatdata(current_seg):
    if len(current_seg) == 0:
        current_seg = segment_filt
    else:
        current_seg = pd.concat([current_seg.reset_index(drop=True), segment_filt.reset_index(drop=True)],
                    axis=1, ignore_index=True)
    return current_seg

def is_MI_segment(labels):
    labels = labels.T
    label_count = labels.value_counts()[:1]
    label_num = label_count.index.tolist()[0]
    if (label_count[0] == 500) and (label_num in ['0', '1', '2']):
        return True, label_num
    else:
        return False, 'noMI'

def do_prediction(current_seg, net):
    current_tensor = torch.from_numpy(np.array(current_seg))
    current_tensor = current_tensor[np.newaxis, np.newaxis, :, :]
    output = net(current_tensor.float())
    _, predicted = torch.max(output.data, 1)
    return predicted

columns = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
data = dict((k, []) for k in columns)
labels = []
current_seg = pd.DataFrame()
total_MI_outliers = 0
all_MI_segments, all_MI_labels, predictions = [], [], []
while not finished:
    sample, timestamp = inlet.pull_sample()
    data = update_data(data, sample)
    print(len(data['FZ']))

    labels.append(1) # would normally be put in code/data

    if len(data['FZ']) == 125:
        df = pd.DataFrame.from_dict(data)
        segment_filt, outlier, filters = utils.pre_processing(df, electrode_names, filters, 
                        sample_duration, freq_limits_names, sampling_frequency)
        current_seg = concatdata(current_seg)

    if len(current_seg.columns) == 500:
        labels = pd.DataFrame(labels, columns = 'label')
        MI_state, current_label = is_MI_segment(labels)
        if MI_state:
            if outlier > 0:
                total_MI_outliers +=1
                print('OUTLIER')
            else:
                all_MI_segments.append(current_seg)
                all_MI_labels.append(int(current_label)) 
                prediction = do_prediction(current_seg, net)
                predictions.append(int(prediction[0]))
                print(f"prediction: {prediction}, true label: {current_label}")
        else:
            print(current_label)

    data = dict((k, []) for k in columns)
    current_seg = current_seg.iloc[:,125:]
    labels = labels[125:]

  



