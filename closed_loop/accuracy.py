import pickle
import numpy as np
import os
from pathlib import Path

total_acc = []

folder_path = Path(f'closed_loop\Expdata\Subjects\wet\X01\session3\closedloop')

for instance in os.scandir(folder_path):
    if instance.path.endswith('.pkl'): 
        a_file = open(instance.path, "rb")
        data_dict = pickle.load(a_file)
        mi_seg = data_dict['MI_segments']
        labels = data_dict['MI_labels']
        pred = data_dict['predictions']

        acc = sum(1 for x,y in zip(labels,pred) if x == y) / len(pred)
        print(acc)
        total_acc.append(acc)
print(f'total avg: {round(sum(total_acc)/len(total_acc),3)} - {round(np.std(total_acc),3)}')
