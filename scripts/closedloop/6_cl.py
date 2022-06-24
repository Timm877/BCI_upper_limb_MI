import numpy as np
import pandas as pd
import realtime_utils as utils
import torch
import pickle
expName = 'closedloop'
exType = 'wet'
expInfo = {'participant': 'X02','type': exType, 'expName' : expName, 'sessionNum': 'session3'}
#INIT
filt_ord = 2
freq_limits = np.asarray([[1,100]]) 
freq_limits_names = ['1_100Hz']
sample_duration = 125
sampling_frequency = 250
electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
filters = utils.init_filters(freq_limits, sampling_frequency, filt_type = 'bandpass', order=filt_ord)
segments, labels, predictions = [], [], []
total_outlier = 0

# init DL model
subject = expInfo['participant']
net = utils.EEGNET()
path = r'closed_loop/final_models/models_for_closedloop/EEGNET_X02_best_finetune_session3' # TODO update path later
net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
net = net.float()
net.eval()

from psychopy import gui, visual, core, data, event, logging, clock, colors, layout
import psychopy.iohub as io
from psychopy.hardware import keyboard
#numpy and pd for data storing and manipulation
from numpy.random import  shuffle
# misc libraries to structure the cues properly and save it with date time and stuff
import time
from datetime import date
from pathlib import Path
import random
import os
import time
import msvcrt
import datetime
# lab streaming layer library to capture the data sent by unicorn EEG headset
from pylsl import StreamInlet, resolve_stream
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#change path of folders according to your needs
# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
result_path = Path(f'closed_loop/Expdata/Subjects/'+exType+'/'+expInfo['participant']+'/'+expInfo['sessionNum']+'/'+expName+'/')
result_path.mkdir(exist_ok=True, parents=True)

columns=['Time','FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8','AccX','AccY','AccZ','Gyro1','Gyro2','Gyro3',
                                  'Battery','Counter','Validation']

data_dict = dict((k, []) for k in columns)
current_seg = pd.DataFrame()
total_MI_outliers = 0
all_MI_segments, all_MI_labels, predictions = [], [], []

MI_dict = {'MI_segments' : [], 'MI_labels': [], 'predictions': []}

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
# image of cross being showed
restCross = visual.ImageStim(
    win=win, name='RestCross',
    image=r'closed_loop/VC_Cross.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
# this has been been manipulated to show random cues for the subject throughout the trial
Cue = visual.ImageStim(
    win=win, name='Cue',
    image=r'closed_loop/VC_Right.jpg', mask=None, anchor='center',
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
    image=r'closed_loop/VC_Blank.jpg', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)

# below code is for initializing the streaming layer which will help us capture data later
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
print('No. of Practice Trials before :', 2)
print("Trial Number :", session)

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

# image list with labels for showing randomly and storing in the database
# creating cue list
img_list = [(r'closed_loop/VC_Relax.jpg',0),
(r'closed_loop/VC_Right.jpg',1),
(r'closed_loop/VC_Left.jpg',2)]*2
trials = len(img_list)
np.random.shuffle(img_list)
# calculating run time to shut off data capturing
runtime = calTime +trials*(restTime + cueTime + focusTime + blkTime)

classes = [] # to store the showed classes in a list later to be added to database .csv file
Fs = 250 # sampling frequency of Unicorn EEG cap
temp = []
times = []
#start = time.time()
flush = [17*Fs, 37*Fs, 57*Fs, 77*Fs, 97*Fs, 117*Fs]

initial = 0
final = 125
prediction = -1
outliers = []

while not finished:
    
    sample, timestamp = inlet.pull_sample()
    
    if len(times) ==0 :
        start_count = sample[15]
        ten_sec.draw()
        win.flip()
        core.wait(calTime)    
        classes = classes + 10*250*['Y']
        
    elif len(times) == Fs*10:
    
        for cue in img_list:
            
            pos = (0,0)
            size = (0.045, 0.045)
            focus.setPos(pos)
            focus.setSize(size)  
            
            Cue.image = cue[0] 
            cue_cls = cue[1] 
            
            restCross.draw()
            win.flip()
            core.wait(restTime)
            
            Cue.draw()
            win.flip()
            core.wait(cueTime)
            
            focus.draw()
            win.flip()
            core.wait(2)
            
            #flushing data to do real time stuff
            while sample[15] not in [a + start_count for a in flush]:
                sample, timestamp = inlet.pull_sample()
                res = [timestamp] + sample
                data_dict = utils.update_data(data_dict,res)
                times.append(timestamp)
                
                if len(data_dict['FZ']) % 125 == 0:
                    df, initial, final = utils.segment_dict(initial, final, sample_duration, data_dict)
                    segment_filt, outlier, filters = utils.pre_processing(df, electrode_names, filters, 
                                    sample_duration, freq_limits_names, sampling_frequency)
                    current_seg = utils.concatdata(current_seg,segment_filt)
            # all columns are same now
            labels = []
            #for num in utils.Genrandom(int((focusTime-2)//0.5)):
            for num in range(0,int((focusTime-2)//0.5)):
                # inlet stuff, preprocess and predict
                buffer = {'time':[],'sample':[]}
                while len(buffer['sample'])<125:
                    sample, timestamp = inlet.pull_sample()
                    labels.append(cue_cls)
                    buffer['time'].append(timestamp)
                    buffer['sample'].append(sample)
                    
                    res = [timestamp] + sample 
                    data_dict = utils.update_data(data_dict,res)
                    times.append(timestamp)
                    
                    if len(data_dict['FZ']) % 125 == 0:
                        df, initial, final = utils.segment_dict(initial, final, sample_duration, data_dict)
                        segment_filt, out, filters = utils.pre_processing(df, electrode_names, filters, 
                                        sample_duration, freq_limits_names, sampling_frequency)
                        current_seg = utils.concatdata(current_seg,segment_filt)   
                        outliers.append(out)   
                # do prediction with current segment and update number
                if len(labels) >= 500:
                    labels_df = pd.DataFrame(labels, columns = ['label'])
                    MI_state, current_label = utils.is_MI_segment(labels_df)
                    if MI_state:
                        if sum(outliers) > 0:
                            total_MI_outliers +=1
                            print('OUTLIER')
                        else:
                            all_MI_segments.append(current_seg)
                            all_MI_labels.append(int(current_label)) 
                            prediction = utils.do_prediction(current_seg, net)
                            predictions.append(int(prediction[0]))
                            print(f"prediction: {prediction}, true label: {current_label}")
                    else:
                        print(current_label)
                    outliers = outliers[1:]
                # def movedotwhen(prediction,focus,pos,size,cue):
                #focus, pos, size = utils.movedot(num,focus,pos,size)  
                focus, pos, size = utils.movedotwhen(prediction,focus,pos,size, cue_cls) 
                focus.draw()
                win.flip()
                core.wait(0.45)
            
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
    data_dict = utils.update_data(data_dict,res)
    times.append(timestamp)
    
    if len(data_dict['FZ']) % 125 == 0:
        df, initial, final = utils.segment_dict(initial, final, sample_duration, data_dict)
        segment_filt, outlier, filters = utils.pre_processing(df, electrode_names, filters, 
                        sample_duration, freq_limits_names, sampling_frequency)
        current_seg = utils.concatdata(current_seg,segment_filt)
    
data_dict['Class'] = classes
# making dictionary into a dataframe for saving it as csv
record_data = pd.DataFrame.from_dict(data_dict)

#saving MI segments in pickle file
MI_dict = {'MI_segments' : [], 'MI_labels': [], 'predictions': []}
MI_dict['MI_segments'] = all_MI_segments
MI_dict['MI_labels'] = all_MI_labels
MI_dict['predictions'] = predictions

result_path = Path(f"closed_loop/Expdata/Subjects/{exType}/{subject}/{expInfo['sessionNum']}/{expName}")
exp_type = expInfo['type']
MI_fname = f'{subject}_{str(date.today())}_{expName}_{exp_type}_{session}_MIData.pkl'
print("Saving file MI Data file as.. ", MI_fname)
save_file = open(result_path / MI_fname, "wb")
pickle.dump(MI_dict, save_file)
save_file.close()

fname = Path('closed_loop/Expdata/Subjects/'+exType+'/'+expInfo['participant']+'/'+expInfo['sessionNum']+'/'+expName+'/'+results_fname)
record_data.to_csv(fname, index = False)
print('Trial Ended')