#!/usr/bin/env python
# -*- coding: utf-8 -*-
expName = 'Envdata'
exType = 'wet'
expInfo = {'participant': 'X03','type': exType, 'sessionNum': 'session3'}
from psychopy import gui, visual, core, data, event, logging, clock, colors, layout
import numpy as np
import pandas as pd

import psychopy.iohub as io
from psychopy.hardware import keyboard

from numpy.random import random, shuffle
import time
from datetime import date
from pathlib import Path

import os
import time
import msvcrt
import datetime

from pylsl import StreamInlet, resolve_stream


# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
result_path = Path(f'closed_loop/Expdata/Subjects/'+exType+'/'+expInfo['participant']+'/'+expInfo['sessionNum']+'/'+expName+'/')
result_path.mkdir(exist_ok=True, parents=True)

# ----------- columns of recorded eeg data ----------
#columns=['Time','EEG1', 'EEG2', 'EEG3','EEG4','EEG5','EEG6','EEG7','EEG8','AccX','AccY','AccZ','Gyro1','Gyro2','Gyro3',
#                                 'Battery','Counter','Validation']
columns=['Time','FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8','AccX','AccY','AccZ','Gyro1','Gyro2','Gyro3',
                                  'Battery','Counter','Validation']

data_dict = dict((k, []) for k in columns)

def update_data(data,res):
    i = 0
    for key in list(data.keys()):
        data[key].append(res[i])
        i = i +1
    return data


calTime = 60.0

# --------- Preparing Ready Window --------
win = visual.Window(
    size=(1440, 900), fullscr=True, screen=1, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color='black', colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')


# -----------Initializing stimuli to be shown -------
# Initialize components for Routine "60 sec calibration"
ten_sec = ten_sec = visual.ShapeStim(
    win=win, name='ten_sec',color = 'black',
    size=(0.044, 0.044), vertices='circle', # change size to 0.045,0.045 if zoomed in
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='darkgrey', fillColor=(0.3255,0.3255,0.3255),
    opacity=None, depth=-2.0, interpolate=True)


finished = False
save_update = 1
stream_timestamp = 1000
streams = resolve_stream()
inlet = StreamInlet(streams[0])
sig_tot = ''
i = 0

# Auto updating trial numbers
trial_list = []
for instance in os.scandir(result_path):
        if instance.path.endswith('.csv'):
            length = len(instance.path)
            trial_list.append(int(instance.path[length-5]))

if len(trial_list) == 0:
    session = '01'
elif len(trial_list) < 10 :
    session = len(trial_list) + 1
    session = '0' + str(session)
else :
    session = str(len(trial_list) + 1)
    
    
print(f"Conducting {expName} experiment for subject :", expInfo['participant'])
print("Trial Number :", session)

    
results_fname = expInfo['participant']+'_'+str(date.today())+'_'+expName+'_'+ expInfo['type']+'_'+session+'.csv'
print("Saving file as .. ", results_fname)

# Create a stimulus for a certain window
readyText = visual.TextStim(win, "Ready?", color=(1,1,1))
readyText.draw()
#present ready text on the screen 
win.flip()
#wait for user to press return key 
event.waitKeys(keyList=['return'])

runtime = calTime

classes = []
Fs = 250
temp = []
times = []
#start = time.time()
while not finished:

    sample, timestamp = inlet.pull_sample()
    #end = time.time()
    #elapsed_time = end - start


    if len(times) == 0:
        ten_sec.draw()
        win.flip()
        core.wait(calTime)
        classes = classes + 60*250*['X']


#         res = [timestamp] + sample
#         data_dict = update_data(data_dict,res)
                
        message = visual.TextStim(win, text="Trial Done")
        # Draw the stimulus to the window. We always draw at the back buffer of the window.
        message.draw()
        # Flip back buffer and front  buffer of the window.
        win.flip()
        # Pause 5 s, so you get a chance to see it!
        core.wait(5.0)
        # Close the window
        win.close()
#         finished = True
#         break 
        
   
    if len(times) > runtime*Fs or len(times) == runtime*Fs :
        finished = True
        break
  
    res = [timestamp] + sample
    data_dict = update_data(data_dict,res)
    times.append(timestamp)

data_dict['Class'] = classes
record_data = pd.DataFrame.from_dict(data_dict)

# Auto updating trial numbers
trial_list = []
for instance in os.scandir(result_path):
        if instance.path.endswith('.csv'):
            length = len(instance.path)
            trial_list.append(int(instance.path[length-5]))

if len(trial_list) ==0:
    session = '01'
else :
    session = len(trial_list) +1
    session = '0'+ str(session)
    #print(session)

#results_fname = expInfo['participant']+'_'+str(date.today())+'_'+expName+'_'+ expInfo['type']+'_'+session+'.csv'
#fname = Path('./Expdata/Subjects/'+expInfo['participant']+'/'+ expName + '/'+results_fname)
fname = Path('closed_loop/Expdata/Subjects/'+exType+'/'+expInfo['participant']+'/'+expInfo['sessionNum']+'/'+expName+'/'+results_fname)
record_data.to_csv(fname, index = False)
print('Trial Ended')