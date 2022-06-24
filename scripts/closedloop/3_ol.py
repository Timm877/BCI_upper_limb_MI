#!/usr/bin/env python
# -*- coding: utf-8 -*-
# adding all the required libraries
#psychopy libraries for running the visual cues
expName = 'openloop'
exType = 'wet'
expInfo = {'participant': 'X02','type': exType, 'sessionNum': 'session4'}
from psychopy import gui, visual, core, data, event, logging, clock, colors, layout
import psychopy.iohub as io
from psychopy.hardware import keyboard
#numpy and pd for data storing and manipulation
import numpy as np
import pandas as pd
from numpy.random import random, shuffle
# misc libraries to structure the cues properly and save it with date time and stuff
import time
from datetime import date
from pathlib import Path

import os
import time
import msvcrt
import datetime
# lab streaming layer library to capture the data sent by unicorn EEG headset
from pylsl import StreamInlet, resolve_stream

#change path of folders according to your needs
# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
result_path = Path(f'closed_loop/Expdata/Subjects/'+exType+'/'+expInfo['participant']+'/'+expInfo['sessionNum']+'/'+expName+'/')
result_path.mkdir(exist_ok=True, parents=True)

# ----------- columns of recorded eeg data ----------
#columns=['Time','EEG1', 'EEG2', 'EEG3','EEG4','EEG5','EEG6','EEG7','EEG8','AccX','AccY','AccZ','Gyro1','Gyro2','Gyro3',
#                                 'Battery','Counter','Validation']
columns=['Time','FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8','AccX','AccY','AccZ','Gyro1','Gyro2','Gyro3',
                                  'Battery','Counter','Validation']

data_dict = dict((k, []) for k in columns)

#updating database with captured data from EEG
def update_data(data,res):
    i = 0
    for key in list(data.keys()):
        data[key].append(res[i])
        i = i +1
    return data


# time for the trial
calTime = 10.0
restTime = 3.0
cueTime = 2.0
focusTime = 10.0
blkTime = 5.0

# main code for visual cues 
''' 
1. Initializing Cue monitor
2. Initializing the cues
3. Calling the cues with the help of time
'''
# --------- Preparing Ready Window --------
win = visual.Window(
    size=(1440, 900), fullscr=True, screen=1, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color='black', colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')

# -----------Initializing stimuli to be shown -------
# Initialize components for Routine "10 sec calibration"
ten_sec = ten_sec = visual.ShapeStim(
    win=win, name='ten_sec',color = 'black',
    size=(0.044, 0.044), vertices='circle', # change size to 0.045,0.045 if zoomed in
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='darkgrey', fillColor=(0.3255,0.3255,0.3255),
    opacity=None, depth=-2.0, interpolate=True)

# Initialize components for Routine "trial"
# image of cross being showed
restCross = visual.ImageStim(
    win=win, name='RestCross',
    image='closed_loop/VC_Cross.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
# this has been been manipulated to show random cues for the subject throughout the trial
Cue = visual.ImageStim(
    win=win, name='Cue',
    image='closed_loop/VC_Right.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
# the small dot on the screen where the subject has to focus for our trial, later to be move during closed loop trials
focus = visual.ShapeStim(
    win=win, name='focus',color = 'black',
    size=(0.044, 0.044), vertices='circle', # change size to 0.045,0.045 if zoomed in
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='darkgrey', fillColor=(0.3255,0.3255,0.3255),
    opacity=None, depth=-2.0, interpolate=True)
# blank screen for rest between cues for blinking, swallowing and other stuff
Blank = visual.ImageStim(
    win=win, name='BlankScreen',
    image='closed_loop/VC_Blank.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)

# below code is for initializing the streaming layer which will help us capture data later
finished = False
streams = resolve_stream()
inlet = StreamInlet(streams[0])
sig_tot = ''

# Auto updating trial numbers
trial_list = []
for instance in os.scandir(result_path):
        if instance.path.endswith('.csv'):
            length = len(instance.path)
            trial_list.append(int(instance.path[length-5]))

if len(trial_list) == 0:
    session = '01'
elif len(trial_list) < 9 :
    session = len(trial_list) + 1
    session = '0' + str(session)
else :
    session = str(len(trial_list) + 1)
    
    
print(f"Conducting {expName} experiment for subject :", expInfo['participant'])
print('No. of Practice Trials before :', 2)
print("Trial Number :", session)
# if int(session) < 3:
#     print('Practice Trial')
# else :
#     print('Actual Trial')
print('Actual Trial')
print('Total number of trials as of now :', int(session) + 2)
results_fname = expInfo['participant']+'_'+str(date.today())+'_'+expName+'_'+ expInfo['type']+'_'+session+'.csv'
print("Saving file as .. ", results_fname)

# -------- Beginning of trial ----------
# Create a stimulus for a certain window
readyText = visual.TextStim(win, "Ready?", color=(1,1,1))
readyText.draw()
#present ready text on the screen 
win.flip()
#wait for user to press return key 
event.waitKeys(keyList=['return'])
# if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
#     core.quit()

# image list with labels for showing randomly and storing in the database
# creating cue list
img_list = [('closed_loop/VC_Relax.jpg',0),('closed_loop/VC_Right.jpg',1),('closed_loop/VC_Left.jpg',2)]*2
trials = len(img_list)
np.random.shuffle(img_list)
# calculating run time to shut off data capturing
runtime = calTime +trials*(restTime + cueTime + focusTime + blkTime)

classes = [] # to store the showed classes in a list later to be added to database .csv file
Fs = 250 # sampling frequency of Unicorn EEG cap
temp = []
times = []
#start = time.time()
while not finished:
    # capturing sample first using inlet.pull_sample()
    sample, timestamp = inlet.pull_sample()

    # showing cues based on time, when time = 0 show ten_sec (cue initialized above)
    if len(times) == 0: #times is a multiple of sampling frequency as we show 10 secs of initial screen
        ten_sec.draw()
        win.flip()
        core.wait(calTime)
        classes = classes + 10*250*['Y'] # adding class to the class list


    elif len(times) == 250*10 : # so length of times list would be 2500 now and its time for the next cue
    
        for cue in img_list:
            
            Cue.image = cue[0] # updating cue image 
            cue_cls = cue[1] # updating cue label

            restCross.draw()
            win.flip()
            core.wait(restTime)
            #win.flip()

            Cue.draw()
            win.flip()
            core.wait(cueTime)
            #win.flip()

            focus.draw()
            win.flip()
            core.wait(focusTime)
            
            Blank.draw()
            win.flip()
            core.wait(blkTime)
            
            # updating class list based on the cues shown
            if cue_cls == 0:
                temp = 3*Fs*['Z']+4*Fs*['relax']+8*Fs*[0]+5*Fs*['rest']
                classes = classes + temp
            elif cue_cls == 1:
                temp = 3*Fs*['Z']+4*Fs*['right']+8*Fs*[1]+5*Fs*['rest']
                classes = classes + temp
            elif cue_cls == 2:
                temp = 3*Fs*['Z']+4*Fs*['left']+8*Fs*[2]+5*Fs*['rest']
                classes = classes + temp
                
        message = visual.TextStim(win, text="Trial Done")
        message.draw()
        win.flip()
        core.wait(5.0)
        win.close()
        
   # ending trial after runtime gets over (calculated beforehand)
    if len(times) > runtime*Fs or len(times) == runtime*Fs :
        finished = True
        break
    # updating data dictionary with newly transmitted samples 
    res = [timestamp] + sample
    data_dict = update_data(data_dict,res)
    times.append(timestamp)

data_dict['Class'] = classes
# making dictionary into a dataframe for saving it as csv
record_data = pd.DataFrame.from_dict(data_dict)


fname = Path('closed_loop/Expdata/Subjects/'+exType+'/'+expInfo['participant']+'/'+expInfo['sessionNum']+'/'+expName+'/'+results_fname)
record_data.to_csv(fname, index = False)
print('Trial Ended')