import numpy as np
from pylsl import StreamInlet, resolve_stream
from datetime import datetime
from utils import *


# init stuff
duration = 4 # --> 4 seconds
sampling_frequency = 250 # why 128 and not 250 as is standard for Unicorn device? --> for now, changed to 250
down_sampling_ratio = 5
n_electrodes = 8

streams = resolve_stream()
inlet = StreamInlet(streams[0])

aborted = False
i = 0
j = 0
    
# initialize filters coefficients to filter freqs to get brain waves
filters = [] 
freq_lim = np.asarray([
    [4, 4], #delta --> funcs: deep sleep
    [4, 8], #theta --> funcs: creativity, insight, deep states, dreams
    [8, 13], #alpha --> physically and mentally relaxed
    [13, 13]]) #beta  --> awake, conciousness, thinking
    # gamma is not used (funcs --> learning, problem solving)
filt_type = ['lowpass', 'bandpass', 'bandpass', 'highpass']
for f in range(freq_lim.shape[0]):
    # I think in gen_coeff fs is set at 50 because fs / down samp ratio == 250/5 = 50
    # output are coefficients: Numerator (b) and denominator (a)
    b, a = gen_coeff(freq_lim[f], filtype=filt_type[f])
    filters.append([b, a])


sig = []
while not aborted:
    sample, timestamp = inlet.pull_sample()

    # Get data and append in sig every 5 datapoints
    if i%down_sampling_ratio==0:
        x = np.asarray([sample[i] for i in range(n_electrodes)])
        sig.append(x)
        j += 1 

    # only when more than enough data is gathered (here: every 4 seconds, as fs is 250Hz and downsampling ratio is 5, 
    # so we gather actually 50Hz & only after 200 datapoints we proceed --> 4 seconds
    if j > duration*sampling_frequency/down_sampling_ratio:
        sig = np.asarray(sig) #sig is here a 200x8 matrix containing raw EEG of 200 datapoints, of each electrode
        feat = []
        electrode_places = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Current Time =", current_time) # indeed every 4 seconds
        for electrode in range(n_electrodes):
            # below, we start doing pre processing and other stuff --> apply 4 filter to receivce brain waves
            # after that, calculate PSD / entropy
            feat = np.asarray([comp_feat_short(sig[:, electrode], filters)]).reshape(1, -1)
            # input sign is 200x1, output is 4 filtered over that matrix, so 200x4, which each column a different 'brain wave'
            # for each column 1 PSD is calculated --> 4x1
            print(electrode_places[electrode])   
            print(feat) # print PSD for each electrode

        # after all, we empty sig and j and start collection of data again
        sig = []
        j = 0
    i+= 1


