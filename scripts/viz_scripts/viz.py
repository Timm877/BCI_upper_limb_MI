from pathlib import Path
from numpy import size
import pandas as pd
import pickle
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
# Load packages
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns 
sns.set_style('darkgrid')



sampling_frequency = 250 
# testing here for 8 electrodes:
electrode_names =  ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
n_electrodes = len(electrode_names)
folder_path = Path(f'./data/openloop/intermediate_datafiles/preprocess/X02_RG_more')
result_path = Path(f'./results/intermediate_datafiles/openloop/X02')
result_path.mkdir(exist_ok=True, parents=True)

file_path = Path(f'./data/openloop/intermediate_datafiles/preprocess/TL_1_100Hz/X01_deep.pkl')
file_path = Path(f'data\openloop\intermediate_datafiles\preprocess\RG_WITHNOTCH\X05_riemann.pkl')
a_file = open(file_path, "rb")
data_dict = pickle.load(a_file)
X = data_dict['data']
y = data_dict['labels']

for df in X: 
    segments=0
    for segment in range(len(X[df])):
        if y[df][segment] == 1:
            current_seg = X[df][segment].T
            current_seg.columns = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8']
            current_seg.drop(['CZ', 'C4', 'PZ', 'PO7', 'OZ'], axis=1, inplace=True)
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams["legend.loc"] = 'upper right'
            current_seg.plot(subplots=True, legend=True, color='tab:blue', sharex=True, sharey=False)
            plt.xlabel('Datapoints of 2 seconds, captured with 250Hz', size=16)
            plt.ylabel('Processed EEG signal (\u03BCV)', size=16)
            plt.suptitle("Example of a 2 second segment of EEG signals for electrodes FZ, C3, and PO8",    # Set plot title
             size=25)
            plt.show()
            
#utils2.plot_dataset(current_seg, ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8'],
#              ['like', 'like', 'like','like', 'like', 'like','like', 'like'],
#              ['line','line', 'line','line','line', 'line','line','line'])
#current_seg = current_seg.T

#print(len(current_seg.columns))
#print(current_label)
# And add time comparison
'''
for instance in os.scandir(folder_path):
    if 'riemann' in instance.path and '500' in instance.path: 
        print(f'Running for {instance.path}...')
        a_file = open(instance.path, "rb")
        data_dict = pickle.load(a_file)
        X = data_dict['data']
        y = data_dict['labels']
        window_size = int(instance.path.split("ws",1)[1][:3])
        for df in X: 
            segments=0
            for segment in range(len(X[df])):
                if segments == 0:
                    data = pd.DataFrame(X[df][segment].T)
                    if y[df][segment] == 0:
                        label = pd.DataFrame({'label_relax': pd.Series(1 for i in range(500)),
                        'label_right_arm': pd.Series(0 for i in range(500)),
                        'label_left_arm': pd.Series(0 for i in range(500))})
                    elif y[df][segment] == 1:
                        label = pd.DataFrame({'label_relax': pd.Series(0 for i in range(500)),
                        'label_right_arm': pd.Series(1 for i in range(500)),
                        'label_left_arm': pd.Series(0 for i in range(500))})
                    elif y[df][segment] == 2:
                        label = pd.DataFrame({'label_relax': pd.Series(0 for i in range(500)),
                        'label_right_arm': pd.Series(0 for i in range(500)),
                        'label_left_arm': pd.Series(1 for i in range(500))})

                    data = pd.concat([data, label], axis=1)
                    data.columns = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8', 'label_relax', 
                    'label_right_arm','label_left_arm']
                elif segments % 5 == 0:
                    data2 = pd.DataFrame(X[df][segment].T)

                    if y[df][segment] == 0:
                        label = pd.DataFrame({'label_relax': pd.Series(1 for i in range(500)),
                        'label_right_arm': pd.Series(0 for i in range(500)),
                        'label_left_arm': pd.Series(0 for i in range(500))})
                    elif y[df][segment] == 1:
                        label = pd.DataFrame({'label_relax': pd.Series(0 for i in range(500)),
                        'label_right_arm': pd.Series(1 for i in range(500)),
                        'label_left_arm': pd.Series(0 for i in range(500))})
                    elif y[df][segment] == 2:
                        label = pd.DataFrame({'label_relax': pd.Series(0 for i in range(500)),
                        'label_right_arm': pd.Series(0 for i in range(500)),
                        'label_left_arm': pd.Series(1 for i in range(500))})
                    
                    #label = label.replace({0: 'relax', 1: 'right arm', 2: 'left arm'})
                    data2 = pd.concat([data2, label], axis=1)
                    data2.columns = ['FZ', 'C3', 'CZ', 'C4', 'PZ', 'PO7', 'OZ', 'PO8', 'label_relax', 
                    'label_right_arm','label_left_arm']
                    
                    data = pd.concat([data, data2], ignore_index=True)
                segments+=1
            utils.plot_dataset(data, ['CZ', 'C3', 'C4', 'label_'],
                                  ['like', 'like', 'like', 'like'],
                                  ['line','line', 'line', 'points'])
                                  '''