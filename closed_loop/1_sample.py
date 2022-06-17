from pylsl import StreamInlet, resolve_stream
from psychopy import gui, visual, core, data, event, logging, clock, colors, layout
# GUI for saving data # Store info about the experiment session
expName = 'practice'
exType = 'wet'
expInfo = {'participant': 'X03','type': exType, 'sessionNum': 'session3'}

import numpy as np
import pandas as pd

from numpy.random import random, shuffle
from datetime import date
from pathlib import Path
import os



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


# time for the trial
calTime = 10.0
restTime = 3.0
cueTime = 2.0
focusTime = 10.0
blkTime = 5.0


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
restCross = visual.ImageStim(
    win=win, name='RestCross',
    image='closed_loop/VC_Cross.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
Cue = visual.ImageStim(
    win=win, name='Cue',
    image='closed_loop/VC_Right.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
focus = visual.ShapeStim(
    win=win, name='focus',color = 'black',
    size=(0.044, 0.044), vertices='circle', # change size to 0.045,0.045 if zoomed in
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=1.0,     colorSpace='rgb',  lineColor='darkgrey', fillColor=(0.3255,0.3255,0.3255),
    opacity=None, depth=-2.0, interpolate=True)
Blank = visual.ImageStim(
    win=win, name='BlankScreen',
    image='closed_loop/VC_Blank.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)

finished = False
streams = resolve_stream()
inlet = StreamInlet(streams[0])

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
print("Trial Number :", session)

# if int(session) < 3:
#     print('Practice Trial')
# else :
#     print('Actual Trial')

print('Practice Trial')    
results_fname = expInfo['participant']+'_'+str(date.today())+'_'+expName+'_'+ expInfo['type']+'_'+session+'.csv'
print("Saving file as .. ", results_fname)

# Create a stimulus for a certain window
readyText = visual.TextStim(win, "Ready?", color=(1,1,1))
readyText.draw()
#present ready text on the screen 
win.flip()
#wait for user to press return key 
event.waitKeys(keyList=['return'])
# if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
#     core.quit()

# creating cue list
img_list = [('closed_loop/VC_Relax.jpg',0),('closed_loop/VC_Right.jpg',1),('closed_loop/VC_Left.jpg',2)]*2 # can be two
trials = len(img_list)
np.random.shuffle(img_list)
runtime = calTime +trials*(restTime + cueTime + focusTime + blkTime)

classes = []
Fs = 250
temp = []
times = []
#start = time.time()
while not finished:

    sample, timestamp = inlet.pull_sample()

    if len(times) == 0:
        ten_sec.draw()
        win.flip()
        core.wait(calTime)
        classes = classes + 10*250*['Y']


    elif len(times) == 250*10 :
    
        for cue in img_list:
         
            Cue.image = cue[0]
            cue_cls = cue[1]

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

            if cue_cls == 0:
                temp = 4*Fs*['Z']+4*Fs*['relax']+8*Fs*[0]+4*Fs*['rest']
                classes = classes + temp
            elif cue_cls == 1:
                temp = 4*Fs*['Z']+4*Fs*['right']+8*Fs*[1]+4*Fs*['rest']
                classes = classes + temp
            elif cue_cls == 2:
                temp = 4*Fs*['Z']+4*Fs*['left']+8*Fs*[2]+4*Fs*['rest']
                classes = classes + temp
                
        message = visual.TextStim(win, text="Trial Done")
        message.draw()
        win.flip()
        core.wait(5.0)
        win.close()
        
   
    if len(times) > runtime*Fs or len(times) == runtime*Fs :
        finished = True
        break
  
    res = [timestamp] + sample
    data_dict = update_data(data_dict,res)
    times.append(timestamp)

data_dict['Class'] = classes
record_data = pd.DataFrame.from_dict(data_dict)

fname = Path('closed_loop/Expdata/Subjects/'+exType+'/'+expInfo['participant']+'/'+expInfo['sessionNum']+'/'+expName+'/'+results_fname)
record_data.to_csv(fname, index = False)
print('Trial Ended')