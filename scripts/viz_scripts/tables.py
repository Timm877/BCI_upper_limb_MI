from tabulate import tabulate
from texttable import Texttable

import latextable

import pandas as pd

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
import numpy as np
import os


'''
df1 = pd.read_csv("results\intermediate_datafiles\TLcompare\X01_TLcompare_multipleruns/riemann_X01_trialnum_4.csv")
df2 = pd.read_csv("results\intermediate_datafiles\TLcompare\X02_TLcompare_multipleruns/riemann_X02_trialnum_4.csv")
df3 = pd.read_csv("results\intermediate_datafiles\TLcompare\X03_TLcompare_multipleruns/riemann_X03_trialnum_4.csv")
df4 = pd.read_csv("results\intermediate_datafiles\TLcompare\X04_TLcompare_multipleruns/riemann_X04_trialnum_4.csv")
df5 = pd.read_csv("results\intermediate_datafiles\TLcompare\X05_TLcompare_multipleruns/riemann_X05_trialnum_4.csv")
df6 = pd.read_csv("results\intermediate_datafiles\TLcompare\X06_TLcompare_multipleruns/riemann_X06_trialnum_4.csv")
df7 = pd.read_csv("results\intermediate_datafiles\TLcompare\X07_TLcompare_multipleruns/riemann_X07_trialnum_4.csv")
df8 = pd.read_csv("results\intermediate_datafiles\TLcompare\X08_TLcompare_multipleruns/riemann_X08_trialnum_4.csv")
df9 = pd.read_csv("results\intermediate_datafiles\TLcompare\X09_TLcompare_multipleruns/riemann_X09_trialnum_4.csv")
means = []

means.append(round(df1['final_test_accuracy'].mean(),1))
means.append(round(df2['final_test_accuracy'].mean(),1))
means.append(round(df3['final_test_accuracy'].mean(),1))
means.append(round(df4['final_test_accuracy'].mean(),1))
means.append(round(df5['final_test_accuracy'].mean(),1))
means.append(round(df6['final_test_accuracy'].mean(),1))
means.append(round(df7['final_test_accuracy'].mean(),1))
means.append(round(df8['final_test_accuracy'].mean(),1))
means.append(round(df9['final_test_accuracy'].mean(),1))
print(f"overall mean: {round(sum(means) / len(means), 3)} from {means}")


liststuff = [0.535, 0.536, 0.544, 0.554 ]
print(f"overall mean: {round(sum(liststuff) / len(liststuff), 3)} from {liststuff}")

df1 = pd.read_csv("results/finetune_results/X01/4.csv")
df2 = pd.read_csv("results/finetune_results/X02/4.csv")
df3 = pd.read_csv("results/finetune_results/X03/4.csv")
df4 = pd.read_csv("results/finetune_results/X04/4.csv")
df5 = pd.read_csv("results/finetune_results/X05/4.csv")
df6 = pd.read_csv("results/finetune_results/X06/4.csv")
df7 = pd.read_csv("results/finetune_results/X07/4.csv")
df8 = pd.read_csv("results/finetune_results/X08/4.csv")
df9 = pd.read_csv("results/finetune_results/X09/4.csv")
means = []

means.append(round(df1['test_accuracy'].mean(),1))
means.append(round(df2['test_accuracy'].mean(),1))
means.append(round(df3['test_accuracy'].mean(),1))
means.append(round(df4['test_accuracy'].mean(),1))
means.append(round(df5['test_accuracy'].mean(),1))
means.append(round(df6['test_accuracy'].mean(),1))
means.append(round(df7['test_accuracy'].mean(),1))
means.append(round(df8['test_accuracy'].mean(),1))
means.append(round(df9['test_accuracy'].mean(),1))
print(f"overall mean: {round(sum(means) / len(means), 3)} from {means}")
liststuff = [0.551  , 0.568  ,0.571   , 0.593]
print(f"overall mean: {round(sum(liststuff) / len(liststuff), 3)} from {liststuff}")
'''
from scipy.stats import wilcoxon
list = [0.478, 0.788, 0.433] 
list = [50.0,56.4,60.7,69.2,93.9,60.8,36.1,68.9,67.8,88.0,86.3,84.2]
conFT = [42.7, 33.3 , 41.0, 59.4, 90.6 , 72.6 ,49.1 ,61.1 ,51.4, 85.9, 84.6 ,89.3]  
diff = []
for i in range(len(list)):
        diff.append(list[i] - conFT[i])
w, p = wilcoxon(diff, alternative='greater')

print(p)
print(round(sum(list)/len(list),1))
print(round(np.asarray(list).std(),1))

list = [44.9, 57.3,62.0,59.4,84.7,58.2,40.0,53.6,42.6,78.6,82.5,87.2]
print(round(sum(list)/len(list),1))
print(round(np.asarray(list).std(),1))

list = [44.9, 57.3,62.0]
list = [50.0,56.4,60.7]
list = [41.9, 31.6, 37.2]
list = [42.7, 33.3 , 41.0]
print(round(sum(list)/len(list),1))
print(round(np.asarray(list).std(),1))
'''
from scipy.stats import wilcoxon

conFT = [42.7, 33.3 , 41.0, 59.4, 90.6 , 72.6 ,49.1 ,61.1 ,51.4, 85.9, 84.6 ,89.3]  
genFT = [41.9, 31.6, 37.2,62.4, 91.0,62.8,46.5,61.2, 48.4 , 86.3, 76.5 , 85.5] 

print(sum(conFT) / len(conFT))
print(np.asarray(conFT).std())
print(sum(genFT) / len(genFT))
print(np.asarray(genFT).std())
d = []
for i in range(len(conFT)):
        d.append(conFT[i] - genFT[i])
print(d)

w, p = wilcoxon(d, alternative='greater')
print(w)
print(p)
rg = [0.445, 0.717, 0.402,  0.66, 0.809, 0.543, 0.385,  0.49, 0.414]
csp = [0.45, 0.695 , 0.427, 0.716, 0.78, 0.548, 0.373, 0.525, 0.4]
dl = [0.40 , 0.799, 0.521, 0.78, 0.866, 0.597, 0.382, 0.542, 0.419 ]

d_rgdl = []
d_cspdl = []
for i in range(len(rg)):
        d_rgdl.append(dl[i] - rg[i])
        d_cspdl.append(dl[i] - csp[i])
w, p = wilcoxon(d_rgdl, alternative='greater')
print(w)
print(p)
w, p = wilcoxon(d_cspdl, alternative='greater')
print(w)
print(p)

folder_path1 = Path(f'closed_loop\Expdata\Subjects\wet\X02\session1\closedloop')
folder_path2= Path(f'closed_loop\Expdata\Subjects\wet\X02\session2\closedloop')
folder_path3 = Path(f'closed_loop\Expdata\Subjects\wet\X02\session3\closedloop')

all_labels = []
all_preds = []
total_acc = []
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

print(total_acc)
s1 = total_acc[:5]
s2 = total_acc[5:10]
s3 = total_acc[10:]
print(f'{s1}, {s2}, {s3}')

w, p = wilcoxon(s3,s1, alternative='greater')
print(p)
'''
'''
df11_pre = pd.read_csv("results/cl/x01_ses1_pre.csv")
df12_pre = pd.read_csv("results/cl/x01_ses2_pre.csv")
df13_pre = pd.read_csv("results/cl/x01_ses3_pre.csv")

df11_ft = pd.read_csv("results/cl/x01_ses1_ft.csv")
df12_ft = pd.read_csv("results/cl/x01_ses2_ft.csv")
df13_ft = pd.read_csv("results/cl/x01_ses3_ft.csv")

df21_pre = pd.read_csv("results/cl/x02_ses1_pre.csv")
df22_pre = pd.read_csv("results/cl/x02_ses2_pre.csv")
df23_pre = pd.read_csv("results/cl/x02_ses3_pre.csv")

df21_ft = pd.read_csv("results/cl/x02_ses1_ft.csv")
df22_ft = pd.read_csv("results/cl/x02_ses2_ft.csv")
df23_ft = pd.read_csv("results/cl/x02_ses3_ft.csv")

df31_pre = pd.read_csv("results/cl/x03_ses1_pre.csv")
df32_pre = pd.read_csv("results/cl/x03_ses2_pre.csv")
df33_pre = pd.read_csv("results/cl/x03_ses3_pre.csv") 

df31_ft = pd.read_csv("results/cl/x03_ses1_ft.csv")
df32_ft = pd.read_csv("results/cl/x03_ses2_ft.csv")
df33_ft = pd.read_csv("results/cl/x03_ses3_ft.csv")

df41_pre = pd.read_csv("results/cl/x04_ses1_pre.csv")
df42_pre = pd.read_csv("results/cl/x04_ses2_pre.csv")
df43_pre = pd.read_csv("results/cl/x04_ses3_pre.csv")

df41_ft = pd.read_csv("results/cl/x04_ses1_ft.csv")
df42_ft = pd.read_csv("results/cl/x04_ses2_ft.csv")
df43_ft = pd.read_csv("results/cl/x04_ses3_ft.csv")


plt.rcParams['figure.figsize'] = (12,4)
plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams['font.size'] =  16
fig, (ax1, ax2,ax3,ax4)  = plt.subplots(1,4)
t = np.arange(1,4,1,dtype=int)

means = []
stds = []
means.append(round(df11_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df11_pre['final_val_accuracy'].std(),1))
means.append(round(df12_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df12_pre['final_val_accuracy'].std(),1))
means.append(round(df13_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df13_pre['final_val_accuracy'].std(),1))




means = np.asarray(means)
std = np.asarray(stds)
width = 0.3
ax1.plot(t, means, lw=2, ls='--', label='GenFT', color= 'blue')
#ax.fill_between(t, means+stds, means-stds, facecolor='blue', alpha=0.5)
for i in range(3):
        print(f'{means[i]} - {stds[i]}')
means = []
stds = []

means.append(round(df11_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df11_ft['final_val_accuracy'].std()*100,1))
means.append(round(df12_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df12_ft['final_val_accuracy'].std()*100,1))
means.append(round(df13_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df13_ft['final_val_accuracy'].std()*100,1))
means = np.asarray(means)
std = np.asarray(stds)
#ax1.bar(t, means, lw=2, label='ConFT', color= 'yellow', width=width)
ax1.plot(t, means, lw=2, label='ConFT', color= 'blue')
#ax1.fill_between(t, means+stds, means-stds, facecolor='yellow', alpha=0.5)
for i in range(3):
        print(f'{means[i]} - {stds[i]}')

cl_means = []
cl_stds = []
folder_path1 = Path(f'closed_loop\Expdata\Subjects\wet\X01\session1\closedloop')
folder_path2 = Path(f'closed_loop\Expdata\Subjects\wet\X01\session2\closedloop')
folder_path3 = Path(f'closed_loop\Expdata\Subjects\wet\X01\session3\closedloop')
total_acc = []
for folder_path in [folder_path1, folder_path2, folder_path3]:
        for instance in os.scandir(folder_path):
                if instance.path.endswith('.pkl'): 
                        a_file = open(instance.path, "rb")
                        data_dict = pickle.load(a_file)
                        mi_seg = data_dict['MI_segments']
                        labels = data_dict['MI_labels']
                        pred = data_dict['predictions']

                        acc = sum(1 for x,y in zip(labels,pred) if x == y) / len(pred)
                        #print(acc)
                        total_acc.append(acc)

        cl_means.append(round(sum(total_acc)/len(total_acc)*100,1))
        cl_stds.append(round(np.std(total_acc)*100,1))
cl_means = np.asarray(cl_means)
cl_stds = np.asarray(cl_stds)
for i in range(3):
        print(f'{cl_means[i]} - {cl_stds[i]}')

#ax1.bar(t+0.3, cl_means, lw=2, label='ConFT', color= 'red', width=width)
ax1.plot(t, cl_means, lw=2, label='Closed loop', color= 'red')
#ax.fill_between(t, cl_means+cl_stds, cl_means-cl_stds, facecolor='black', alpha=0.5)
ax1.set_title(r'X01')
ax1.legend(loc='lower left')
ax1.set_xlabel('Session number')
ax1.set_ylabel('Validation accuracy (\%)')
ax1.set_ylim((0, 100)) 
ax1.set_xticks(t)
#ax.grid()


means = []
stds = []
means.append(round(df21_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df21_pre['final_val_accuracy'].std()*100,1))
means.append(round(df22_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df22_pre['final_val_accuracy'].std()*100,1))
means.append(round(df23_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df23_pre['final_val_accuracy'].std()*100,1))
for i in range(3):
        print(f'{means[i]} - {stds[i]}')

means = np.asarray(means)
std = np.asarray(stds)
ax2.plot(t, means, lw=2, label='GenFT', color= 'blue')
#ax.fill_between(t, means+stds, means-stds, facecolor='blue', alpha=0.5)

means = []
stds = []

means.append(round(df21_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df21_ft['final_val_accuracy'].std()*100,1))
means.append(round(df22_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df22_ft['final_val_accuracy'].std()*100,1))
means.append(round(df23_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df23_ft['final_val_accuracy'].std()*100,1))
means = np.asarray(means)
std = np.asarray(stds)
ax2.plot(t, means, lw=2, ls='--', label='ConFT', color= 'blue')
#ax.fill_between(t, means+stds, means-stds, facecolor='yellow', alpha=0.5)
for i in range(3):
        print(f'{means[i]} - {stds[i]}')

cl_means = []
cl_stds = []
folder_path1 = Path(f'closed_loop\Expdata\Subjects\wet\X02\session1\closedloop')
folder_path2 = Path(f'closed_loop\Expdata\Subjects\wet\X02\session2\closedloop')
folder_path3 = Path(f'closed_loop\Expdata\Subjects\wet\X02\session3\closedloop')
total_acc = []
for folder_path in [folder_path1, folder_path2, folder_path3]:
        for instance in os.scandir(folder_path):
                if instance.path.endswith('.pkl'): 
                        a_file = open(instance.path, "rb")
                        data_dict = pickle.load(a_file)
                        mi_seg = data_dict['MI_segments']
                        labels = data_dict['MI_labels']
                        pred = data_dict['predictions']

                        acc = sum(1 for x,y in zip(labels,pred) if x == y) / len(pred)
                        #print(acc)
                        total_acc.append(acc)

        cl_means.append(round(sum(total_acc)/len(total_acc)*100,1))
        cl_stds.append(round(np.std(total_acc)*100,1))
cl_means = np.asarray(cl_means)
cl_stds = np.asarray(cl_stds)
for i in range(3):
        print(f'{cl_means[i]} - {cl_stds[i]}')

ax2.plot(t, cl_means, lw=2, label='Closed loop', color= 'red')
#ax.fill_between(t, cl_means+cl_stds, cl_means-cl_stds, facecolor='black', alpha=0.5)
ax2.set_title(r'X02')
ax2.legend(loc='lower left')
ax2.set_xlabel('Session number')
ax2.set_ylabel('Validation accuracy (\%)')
ax2.set_ylim((0, 100)) 
ax2.set_xticks(t)
#ax.grid()


means = []
stds = []
means.append(round(df31_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df31_pre['final_val_accuracy'].std()*100,1))
means.append(round(df32_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df32_pre['final_val_accuracy'].std()*100,1))
means.append(round(df33_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df33_pre['final_val_accuracy'].std()*100,1))
for i in range(3):
        print(f'{means[i]} - {stds[i]}')

means = np.asarray(means)
std = np.asarray(stds)
ax3.plot(t, means, lw=2, ls='--', label='GenFT', color= 'blue')
#ax.fill_between(t, means+stds, means-stds, facecolor='blue', alpha=0.5)

means = []
stds = []

means.append(round(df31_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df31_ft['final_val_accuracy'].std()*100,1))
means.append(round(df32_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df32_ft['final_val_accuracy'].std()*100,1))
means.append(round(df33_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df33_ft['final_val_accuracy'].std()*100,1))
means = np.asarray(means)
std = np.asarray(stds)
ax3.plot(t, means, lw=2, label='ConFT', color= 'blue')
#ax.fill_between(t, means+stds, means-stds, facecolor='yellow', alpha=0.5)
for i in range(3):
        print(f'{means[i]} - {stds[i]}')

cl_means = []
cl_stds = []
folder_path1 = Path(f'closed_loop\Expdata\Subjects\wet\X03\session1\closedloop')
folder_path2 = Path(f'closed_loop\Expdata\Subjects\wet\X03\session2\closedloop')
folder_path3 = Path(f'closed_loop\Expdata\Subjects\wet\X03\session3\closedloop')
total_acc = []
for folder_path in [folder_path1, folder_path2, folder_path3]:
        for instance in os.scandir(folder_path):
                if instance.path.endswith('.pkl'): 
                        a_file = open(instance.path, "rb")
                        data_dict = pickle.load(a_file)
                        mi_seg = data_dict['MI_segments']
                        labels = data_dict['MI_labels']
                        pred = data_dict['predictions']

                        acc = sum(1 for x,y in zip(labels,pred) if x == y) / len(pred)
                        #print(acc)
                        total_acc.append(acc)

        cl_means.append(round(sum(total_acc)/len(total_acc)*100,1))
        cl_stds.append(round(np.std(total_acc)*100,1))
cl_means = np.asarray(cl_means)
cl_stds = np.asarray(cl_stds)
for i in range(3):
        print(f'{cl_means[i]} - {cl_stds[i]}')

ax3.plot(t, cl_means, lw=2, label='Closed loop', color= 'red')
#ax.fill_between(t, cl_means+cl_stds, cl_means-cl_stds, facecolor='black', alpha=0.5)
ax3.set_title(r'X03')
ax3.legend(loc='lower left')
ax3.set_xlabel('Session number')
ax3.set_ylabel('Validation accuracy (\%)')
ax3.set_ylim((0, 100)) 
ax3.set_xticks(t)
#ax.grid()


means = []
stds = []
means.append(round(df41_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df41_pre['final_val_accuracy'].std()*100,1))
means.append(round(df42_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df42_pre['final_val_accuracy'].std()*100,1))
means.append(round(df43_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df43_pre['final_val_accuracy'].std()*100,1))
for i in range(3):
        print(f'{means[i]} - {stds[i]}')


means = np.asarray(means)
std = np.asarray(stds)
ax4.plot(t, means, lw=2, ls='--', label='GenFT', color= 'blue')
#ax.fill_between(t, means+stds, means-stds, facecolor='blue', alpha=0.5)

means = []
stds = []

means.append(round(df41_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df41_ft['final_val_accuracy'].std()*100,1))
means.append(round(df42_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df42_ft['final_val_accuracy'].std()*100,1))
means.append(round(df43_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df43_ft['final_val_accuracy'].std()*100,1))
means = np.asarray(means)
std = np.asarray(stds)
ax4.plot(t, means, lw=2, label='ConFT', color= 'blue')
#ax.fill_between(t, means+stds, means-stds, facecolor='yellow', alpha=0.5)
for i in range(3):
        print(f'{means[i]} - {stds[i]}')

cl_means = []
cl_stds = []
folder_path1 = Path(f'closed_loop\Expdata\Subjects\wet\X04\session1\closedloop')
folder_path2 = Path(f'closed_loop\Expdata\Subjects\wet\X04\session2\closedloop')
folder_path3 = Path(f'closed_loop\Expdata\Subjects\wet\X04\session3\closedloop')
total_acc = []
for folder_path in [folder_path1, folder_path2, folder_path3]:
        for instance in os.scandir(folder_path):
                if instance.path.endswith('.pkl'): 
                        a_file = open(instance.path, "rb")
                        data_dict = pickle.load(a_file)
                        mi_seg = data_dict['MI_segments']
                        labels = data_dict['MI_labels']
                        pred = data_dict['predictions']

                        acc = sum(1 for x,y in zip(labels,pred) if x == y) / len(pred)
                        #print(acc)
                        total_acc.append(acc)

        cl_means.append(round(sum(total_acc)/len(total_acc)*100,1))
        cl_stds.append(round(np.std(total_acc)*100,1))
cl_means = np.asarray(cl_means)
cl_stds = np.asarray(cl_stds)
for i in range(3):
        print(f'{cl_means[i]} - {cl_stds[i]}')

ax4.plot(t, cl_means, lw=2, label='Closed loop', color= 'red')
#ax.fill_between(t, cl_means+cl_stds, cl_means-cl_stds, facecolor='black', alpha=0.5)
ax4.set_title(r'X04')
ax4.legend(loc='lower left')
ax4.set_xlabel('Session number')
ax4.set_ylabel('Validation accuracy (\%)')
ax4.set_ylim((0, 100)) 
ax4.set_xticks(t)
#ax.grid()







plt.suptitle('Validation accuracy for open-loop fine-tuning with GenFT or ConFT, together with average closed-loop accuracy')
fig.tight_layout()
plt.show()
#


'''

'''



means = []
stds = []
means.append(round(df21_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df21_pre['final_val_accuracy'].std(),1))
means.append(round(df22_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df22_pre['final_val_accuracy'].std(),1))
means.append(round(df23_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df23_pre['final_val_accuracy'].std(),1))
means = np.asarray(means)
std = np.asarray(stds)
ax.plot(t, means, lw=2, label='GenFT X02', color= 'black')
#ax.fill_between(t, means+stds, means-stds, facecolor='green', alpha=0.5)

means = []
stds = []
means.append(round(df21_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df21_ft['final_val_accuracy'].std(),1))
means.append(round(df22_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df22_ft['final_val_accuracy'].std(),1))
means.append(round(df23_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df23_ft['final_val_accuracy'].std(),1))
means = np.asarray(means)
std = np.asarray(stds)
ax.plot(t, means, lw=2, label='ConFT X02', color= 'black')

cl_means = []
cl_stds = []
folder_path1 = Path(f'closed_loop\Expdata\Subjects\wet\X01\session1\closedloop')
folder_path2 = Path(f'closed_loop\Expdata\Subjects\wet\X01\session2\closedloop')
folder_path3 = Path(f'closed_loop\Expdata\Subjects\wet\X01\session3\closedloop')
total_acc = []
for folder_path in [folder_path1, folder_path2, folder_path3]:
        for instance in os.scandir(folder_path):
                if instance.path.endswith('.pkl'): 
                        a_file = open(instance.path, "rb")
                        data_dict = pickle.load(a_file)
                        mi_seg = data_dict['MI_segments']
                        labels = data_dict['MI_labels']
                        pred = data_dict['predictions']

                        acc = sum(1 for x,y in zip(labels,pred) if x == y) / len(pred)
                        #print(acc)
                        total_acc.append(acc)

        cl_means.append(round(sum(total_acc)/len(total_acc),1))
        cl_stds.append(round(np.std(total_acc),1))
cl_means = np.asarray(cl_means)
cl_stds = np.asarray(cl_stds)

#ax.plot(t, cl_means, lw=2, ls=':', label='Closed loop', color= 'black')
#ax.fill_between(t, means+stds, means-stds, facecolor='darkgreen', alpha=0.5)

means = []
stds = []
means.append(round(df31_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df31_pre['final_val_accuracy'].std(),1))
means.append(round(df32_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df32_pre['final_val_accuracy'].std(),1))
means.append(round(df33_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df33_pre['final_val_accuracy'].std(),1))
means = np.asarray(means)
std = np.asarray(stds)
ax.plot(t, means, lw=2, ls='--', label='GenFT X03', color= 'red')
#ax.fill_between(t, means+stds, means-stds, facecolor='red', alpha=0.5)

means = []
stds = []
means.append(round(df31_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df31_ft['final_val_accuracy'].std(),1))
means.append(round(df32_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df32_ft['final_val_accuracy'].std(),1))
means.append(round(df33_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df33_ft['final_val_accuracy'].std(),1))
means = np.asarray(means)
std = np.asarray(stds)
ax.plot(t, means, lw=2, label='ConFT X03', color= 'red')
#ax.fill_between(t, means+stds, means-stds, facecolor='darkred', alpha=0.5)

means = []
stds = []
means.append(round(df41_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df41_pre['final_val_accuracy'].std(),1))
means.append(round(df42_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df42_pre['final_val_accuracy'].std(),1))
means.append(round(df43_pre['final_val_accuracy'].mean()*100,1))
stds.append(round(df43_pre['final_val_accuracy'].std(),1))
means = np.asarray(means)
std = np.asarray(stds)
ax.plot(t, means, lw=2, ls='--', label='GenFT X04', color= 'lightgreen')
#ax.fill_between(t, means+stds, means-stds, facecolor='lightyellow', alpha=0.5)

means = []
stds = []
means.append(round(df41_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df41_ft['final_val_accuracy'].std(),1))
means.append(round(df42_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df42_ft['final_val_accuracy'].std(),1))
means.append(round(df43_ft['final_val_accuracy'].mean()*100,1))
stds.append(round(df43_ft['final_val_accuracy'].std(),1))
means = np.asarray(means)
std = np.asarray(stds)
ax.plot(t, means, lw=2, label='ConFT X04', color= 'lightgreen')
#ax.fill_between(t, means+stds, means-stds, facecolor='yellow', alpha=0.5)

ax.set_title(r'Average validation accuracy for open-loop fine-tuning $\pm \sigma$ with GenFT or ConFT')
ax.legend(loc='lower left')
ax.set_xlabel('Session number')
ax.set_ylabel('Validation accuracy')
ax.set_ylim((0, 1)) 
ax.grid()
plt.xticks(t)
plt.show()


'''
#df0 = pd.read_csv("results/cl/x03_ses2_ft.csv")
#df1 = pd.read_csv("results/finetune_results/X02/0.csv")
#df2 = pd.read_csv("results/finetune_results/X02/0.csv")
#df3 = pd.read_csv("results/finetune_results/X02/0.csv")
#df4 = pd.read_csv("results/finetune_results/X02/0.csv")
#means = []

#print("models")
#print(f"{round(df0['test_accuracy'].mean(),1)} - {round(df0['test_accuracy'].std(),1)}")
#print(round(df0['test_accuracy'].max(),1))
#print(f"{round(df0['train/train_acc'].mean(),1)} - {round(df0['train/train_acc'].std(),1)}")
#print(f"{round(df0['final_val_accuracy'].mean(),1)} - {round(df0['final_val_accuracy'].std(),1)}")
#print(df0['Name'])



'''
print("1")
print(f"{round(df1['test_accuracy'].mean(),1)} - {round(df1['test_accuracy'].std(),1)}")
means.append(round(df1['test_accuracy'].mean(),1))

print("2")
print(f"{round(df2['test_accuracy'].mean(),1)} - {round(df2['test_accuracy'].std(),1)}")
means.append(round(df2['test_accuracy'].mean(),1))

print("3")
print(f"{round(df3['test_accuracy'].mean(),1)} - {round(df3['test_accuracy'].std(),1)}")
means.append(round(df3['test_accuracy'].mean(),1))

print("4")
print(f"{round(df4['test_accuracy'].mean(),1)} - {round(df4['test_accuracy'].std(),1)}")
means.append(round(df4['test_accuracy'].mean(),1))

print(f"overall mean: {round(sum(means) / len(means), 3)} from {means}")
'''

'''


df1 = pd.read_csv("results/intermediate_datafiles/openloop/X01_ML/csp_STD_multiclass.csv")
df2 = pd.read_csv("results/intermediate_datafiles/openloop/X02_ML/csp_STD_multiclass.csv")
df3 = pd.read_csv("results/intermediate_datafiles/openloop/X03_ML/csp_STD_multiclass.csv")
df4 = pd.read_csv("results/intermediate_datafiles/openloop/X04_ML/csp_STD_multiclass.csv")
df5 = pd.read_csv("results/intermediate_datafiles/openloop/X05_ML/csp_STD_multiclass.csv")
df6 = pd.read_csv("results/intermediate_datafiles/openloop/X06_ML/csp_STD_multiclass.csv")
df7 = pd.read_csv("results/intermediate_datafiles/openloop/X07_ML/csp_STD_multiclass.csv")
df8 = pd.read_csv("results/intermediate_datafiles/openloop/X08_ML/csp_STD_multiclass.csv")
df9 = pd.read_csv("results/intermediate_datafiles/openloop/X09_ML/csp_STD_multiclass.csv")



rows = [['Frequency band/ Subject', 'X01','X02', 'X03', 'X04', 'X05', 'X06', 'X07', 'X08', 'X09'],
        ['CSP + sLDA', df1.loc[0,'final_val_accuracy'], df2.loc[0,'final_val_accuracy'],
        df3.loc[0,'final_val_accuracy'],df4.loc[0,'final_val_accuracy'], df5.loc[2,'final_val_accuracy'],
        df6.loc[0,'final_val_accuracy'],df7.loc[2,'final_val_accuracy'], df8.loc[0,'final_val_accuracy'],
        df9.loc[1,'final_val_accuracy'],
             ],
        ['CSP + SVM',df1.loc[1,'final_val_accuracy'], df2.loc[1,'final_val_accuracy'],
        df3.loc[1,'final_val_accuracy'],df4.loc[1,'final_val_accuracy'], df5.loc[1,'final_val_accuracy'],
        df6.loc[1,'final_val_accuracy'],df7.loc[0,'final_val_accuracy'], df8.loc[2,'final_val_accuracy'],
        df9.loc[0,'final_val_accuracy'],
             ],
        ['CSP + RF', df1.loc[2,'final_val_accuracy'], df2.loc[2,'final_val_accuracy'],
        df3.loc[2,'final_val_accuracy'],df4.loc[2,'final_val_accuracy'], df5.loc[0,'final_val_accuracy'],
        df6.loc[2,'final_val_accuracy'],df7.loc[1,'final_val_accuracy'], df8.loc[1,'final_val_accuracy'],
        df9.loc[2,'final_val_accuracy'],
        ]
        ]

#,  'X04', 'X05', 'X06', 'X07', 'X08', 'X09'
table = Texttable()
table.set_cols_align(["c"] * 10)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))

import numpy as np
slda = np.array([df1.loc[0,'final_val_accuracy'], df2.loc[0,'final_val_accuracy'],
        df3.loc[0,'final_val_accuracy'],df4.loc[0,'final_val_accuracy'], df5.loc[2,'final_val_accuracy'],
        df6.loc[0,'final_val_accuracy'],df7.loc[2,'final_val_accuracy'], df8.loc[0,'final_val_accuracy'],
        df9.loc[1,'final_val_accuracy']]).mean()

svm = np.array([df1.loc[1,'final_val_accuracy'], df2.loc[1,'final_val_accuracy'],
        df3.loc[1,'final_val_accuracy'],df4.loc[1,'final_val_accuracy'], df5.loc[1,'final_val_accuracy'],
        df6.loc[1,'final_val_accuracy'],df7.loc[0,'final_val_accuracy'], df8.loc[2,'final_val_accuracy'],
        df9.loc[0,'final_val_accuracy']]).mean()
rf = np.array([df1.loc[2,'final_val_accuracy'], df2.loc[2,'final_val_accuracy'],
        df3.loc[2,'final_val_accuracy'],df4.loc[2,'final_val_accuracy'], df5.loc[0,'final_val_accuracy'],
        df6.loc[2,'final_val_accuracy'],df7.loc[1,'final_val_accuracy'], df8.loc[1,'final_val_accuracy'],
        df9.loc[2,'final_val_accuracy']]).mean()

print(slda)
print(svm)
print(rf)

from tabulate import tabulate
from texttable import Texttable

import latextable

import pandas as pd

df1_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X01_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df1_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X01_WithSPFilt2\multiclass_riemann_multiclass.csv")

df2_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X02_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df2_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X02_WithSPFilt2\multiclass_riemann_multiclass.csv")

df3_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X03_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df3_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X03_WithSPFilt2\multiclass_riemann_multiclass.csv")

df4_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X04_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df4_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X04_WithSPFilt2\multiclass_riemann_multiclass.csv")

df5_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X05_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df5_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X05_WithSPFilt2\multiclass_riemann_multiclass.csv")

df6_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X06_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df6_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X06_WithSPFilt2\multiclass_riemann_multiclass.csv")

df7_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X07_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df7_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X07_WithSPFilt2\multiclass_riemann_multiclass.csv")

df8_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X08_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df8_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X08_WithSPFilt2\multiclass_riemann_multiclass.csv")

df9_norm = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X09_WithNormalFilt2\multiclass_riemann_multiclass.csv")
df9_SP = pd.read_csv("results\intermediate_datafiles\openloop\pipeline_experiments\Filter_type\X09_WithSPFilt2\multiclass_riemann_multiclass.csv")

rows = [['Filter order / Subject', 'X01','X02', 'X03', 'X04', 'X05','X06', 'X07', 'X08', 'X09'],
        ['Butter filtfilt', df1_norm.loc[0,'final_val_accuracy'], df2_norm.loc[0,'final_val_accuracy'],
        df3_norm.loc[0,'final_val_accuracy'],df4_norm.loc[0,'final_val_accuracy'],df5_norm.loc[0,'final_val_accuracy'],
        df6_norm.loc[0,'final_val_accuracy'],df7_norm.loc[0,'final_val_accuracy'],df8_norm.loc[0,'final_val_accuracy'],
        df9_norm.loc[0,'final_val_accuracy'],
             ],
        ['State space filt',df1_SP.loc[0,'final_val_accuracy'], df2_SP.loc[0,'final_val_accuracy'],
        df3_SP.loc[0,'final_val_accuracy'],df4_SP.loc[0,'final_val_accuracy'],df5_SP.loc[0,'final_val_accuracy'],
        df6_SP.loc[0,'final_val_accuracy'],df7_SP.loc[0,'final_val_accuracy'],df8_SP.loc[0,'final_val_accuracy'],
        df9_SP.loc[0,'final_val_accuracy']
             ]
        ]

#,  'X04', 'X05', 'X06', 'X07', 'X08', 'X09'
table = Texttable()
table.set_cols_align(["c"] * 10)
table.set_deco(Texttable.HEADER | Texttable.VLINES)
table.add_rows(rows)

print('\nTabulate Latex:')
print(tabulate(rows, headers='firstrow', tablefmt='latex'))
import numpy as np
norm = np.array([df1_norm.loc[0,'final_val_accuracy'], df2_norm.loc[0,'final_val_accuracy'],
        df3_norm.loc[0,'final_val_accuracy'],df4_norm.loc[0,'final_val_accuracy'],df5_norm.loc[0,'final_val_accuracy'],
        df6_norm.loc[0,'final_val_accuracy'],df7_norm.loc[0,'final_val_accuracy'],df8_norm.loc[0,'final_val_accuracy'],
        df9_norm.loc[0,'final_val_accuracy']]).mean()

sp = np.array([df1_SP.loc[0,'final_val_accuracy'], df2_SP.loc[0,'final_val_accuracy'],
        df3_SP.loc[0,'final_val_accuracy'],df4_SP.loc[0,'final_val_accuracy'],df5_SP.loc[0,'final_val_accuracy'],
        df6_SP.loc[0,'final_val_accuracy'],df7_SP.loc[0,'final_val_accuracy'],df8_SP.loc[0,'final_val_accuracy'],
        df9_SP.loc[0,'final_val_accuracy']]).mean()
print(norm)
print(sp)
'''