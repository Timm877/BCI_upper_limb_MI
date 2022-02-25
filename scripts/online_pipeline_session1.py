from datetime import datetime
import argparse
import numpy as np
import pandas as pd
from pylsl import StreamInlet, resolve_stream
import matplotlib.pyplot as plt

def main():
    # init stuff
    sampling_frequency = 250
    sample_duration = 250
    n_electrodes = 8
    selected_electrodes_names = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
 
    streams = resolve_stream()
    inlet = StreamInlet(streams[0])
    aborted = False
    timestamps = []
    sig = []
    X_all = []
    y_all = []
    while not aborted:
        sample, timestamp = inlet.pull_sample()
        timestamps.append(timestamp)
        x = np.asarray([sample[i] for i in range(n_electrodes)])
        sig.append(x)

        if len(timestamps) == sample_duration:
            print(timestamps[-1]-timestamps[0]) 
            #should be 1 sec the same all the time
            # and initial tests confirm this
            timestamps = []
            df_segment = pd.DataFrame.from_dict(dict(zip(selected_electrodes_names, sig)))
        
        # inititialize time counter
        # first run capture_noise program for 60 seconds and save data in correct folder of subject (date_subject_session/trialnum.csv)
        # then here load in capture_noise program and calculate mean from each channel?
        
        # start experiment
        # pop up screen
        # start capturing data for every second
        # countdown from 10 to 0 (init signal), after 1 sec of data is collected --> add this label to each segment
        # for i in MI_tasks:
        # cross for 2 seconds --> add correct label
        # visual cue of MI_tasks.random() for 2 seconds --> add visual cue label
        # MI_tasks.pop(task)
        # 2 seconds start MI --> add ERP label
        # 6 seconds MI
        # labels: 0 --> init, 1 --> cue to look at screen, 2 --> visual cue on screen, 3 --> ERP, 40, 41, 42 --> MIs
        # after this, repeat!


if __name__ == '__main__':
    # add parser for dummy code
    main()