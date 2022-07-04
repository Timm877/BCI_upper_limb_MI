import pickle
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import  confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
total_acc = []

folder_path1 = Path(f'closed_loop\Expdata\Subjects\wet\X04\session1\closedloop')
folder_path2= Path(f'closed_loop\Expdata\Subjects\wet\X04\session2\closedloop')
folder_path3 = Path(f'closed_loop\Expdata\Subjects\wet\X04\session3\closedloop')

all_labels = []
all_preds = []
for folder_path in [folder_path1,folder_path2,folder_path3]:
    for instance in os.scandir(folder_path):
        if instance.path.endswith('.pkl'): 
            a_file = open(instance.path, "rb")
            data_dict = pickle.load(a_file)
            mi_seg = data_dict['MI_segments']
            labels = data_dict['MI_labels']
            pred = data_dict['predictions']
            for i in range(len(labels)):
                all_labels.append(labels[i])
                all_preds.append(pred[i])

            acc = sum(1 for x,y in zip(labels,pred) if x == y) / len(pred)
            print(acc)
            total_acc.append(acc)
        
print(f'total avg: {round(sum(total_acc)/len(total_acc),3)} - {round(np.std(total_acc),3)}')

cf_matrix = confusion_matrix(all_labels, all_preds)
classes = ('Relax', "Right arm", "Left arm")
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*3, index = [i for i in classes],
                columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.set(font_scale=1.8)
sn.heatmap(df_cm, annot=True, cbar=False)
plt.savefig(f'X04_cl_confmat.png')

