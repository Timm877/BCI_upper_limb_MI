import numpy as np
import pandas as pd
import src.realtime_utils as utils
import torch
import torch.nn as nn
from pylsl import StreamInlet, resolve_stream

# Classes and functions
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class EEGNET(nn.Module):
    def __init__(self, receptive_field=64, filter_sizing=8, dropout=0.25, D=2):
        super(EEGNET,self).__init__()
        channel_amount = 8
        num_classes = 3
        self.temporal=nn.Sequential(
            nn.Conv2d(1,filter_sizing,kernel_size=[1,receptive_field],stride=1, bias=False,\
                padding='same'), 
            nn.BatchNorm2d(filter_sizing),
        )
        self.spatial=nn.Sequential(
            nn.Conv2d(filter_sizing,filter_sizing*D,kernel_size=[channel_amount,1],bias=False,\
                groups=filter_sizing),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )

        self.seperable=nn.Sequential(
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,16],\
                padding='same',groups=filter_sizing*D, bias=False),
            nn.Conv2d(filter_sizing*D,filter_sizing*D,kernel_size=[1,1], padding='same',groups=1, bias=False),
            nn.BatchNorm2d(filter_sizing*D),
            nn.ELU(True),
        )
        self.avgpool1 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)   
        self.avgpool2 = nn.AvgPool2d([1, 5], stride=[1, 5], padding=0)
        self.dropout = nn.Dropout(dropout)
        self.view = nn.Sequential(Flatten())

        endsize = 320
        self.fc2 = nn.Linear(endsize, num_classes)

    def forward(self,x):
        out = self.temporal(x)
        out = self.spatial(out)
        out = self.avgpool1(out)
        out = self.dropout(out)
        out = self.seperable(out)
        out = self.avgpool2(out)
        out = self.dropout(out)
        out = out.view(out.size(0), -1)
        prediction = self.fc2(out)
        return prediction

def update_data(data, res):
    i = 0
    for key in list(data.keys()):
        data[key].append(res[i])
        i+=1
    return data

def concatdata(current_seg):
    if len(current_seg) == 0:
        current_seg = segment_filt
    else:
        current_seg = pd.concat([current_seg.reset_index(drop=True), segment_filt.reset_index(drop=True)],
                    axis=1, ignore_index=True)
    return current_seg

def is_MI_segment(labels):
    label_count = labels.value_counts()[:1]
    label_num = label_count.index.tolist()[0]
    if (label_count.iloc[0] == 500) and (label_num[0] in [0, 1, 2]):
        # NOTE: label_num is here for some reason a tuple, but is normally a string (like '0') for our files!
        # TODO change above thus to let it work for our data stuff.
        return True, label_num[0]
    else:
        return False, 'noMI'

def do_prediction(current_seg, net):
    current_tensor = torch.from_numpy(np.array(current_seg))
    current_tensor = current_tensor[np.newaxis, np.newaxis, :, :]
    output = net(current_tensor.float())
    _, predicted = torch.max(output.data, 1)
    return predicted

# Init variables
filt_ord = 2
freq_limits = np.asarray([[1,100]]) 
freq_limits_names = ['1_100Hz']
sample_duration = 125
sampling_frequency = 250
subject = 'X01'
columns = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
data = dict((k, []) for k in columns)
labels = []
current_seg = pd.DataFrame()
total_MI_outliers = 0
all_MI_segments, all_MI_labels, predictions = [], [], []

filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filt_ord)

# init DL model
net = EEGNET()
net.load_state_dict(torch.load(f'final_models/best_finetune\EEGNET_Finetuned_{subject}', map_location=torch.device('cpu')))
net = net.float()
net.eval()

#  initializing the streaming layer
finished = False
streams = resolve_stream()
inlet = StreamInlet(streams[0])
sig_tot = ''

while not finished:
    sample, timestamp = inlet.pull_sample()
    data = update_data(data, sample)

    # TODO change to actual label:
    labels.append(1) 

    # every 0.5 sec do filtering and concat to current seg
    if len(data['FZ']) == 125:
        df = pd.DataFrame.from_dict(data)
        segment_filt, outlier, filters = utils.pre_processing(df, columns, filters, 
                        sample_duration, freq_limits_names, sampling_frequency)
        current_seg = concatdata(current_seg)

        # re-initialize variables
        data = dict((k, []) for k in columns)
        index = 0

    # every 2 sec: get labels and check if segment is 100% MI segment
    # then: if not outlier, do predictions
    if len(current_seg.columns) == 500:
        labels_df = pd.DataFrame(labels)
        MI_state, current_label = is_MI_segment(labels_df)
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
            pass

        # lastly, delete first 0.5 seconds
        current_seg = current_seg.iloc[:,125:]
        labels = labels[125:]

    #TODO save raw EEG
    #TODO save all_MI_segments, all_MI_labels, and predictions

  



