import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from scipy import signal
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM


def init_filters(freq_lim, sample_freq, filt_type = 'bandpass', order=2):
    filters = []
    for f in range(freq_lim.shape[0]):
        b, a = init_filt_coef(freq_lim[f], fs=sample_freq, filtype=filt_type, order=order)
        filters.append([b, a])
    return filters

def init_filt_coef(cuttoff, fs, filtype, order):
    if filtype == 'lowpass':
        b,a = signal.butter(order, cuttoff[0]/(fs/2), filtype)
    elif filtype == 'bandpass':
        b,a = signal.butter(order, [cuttoff[0]/(fs/2), cuttoff[1]/(fs/2)], filtype)
    elif filtype == 'highpass':
        b,a = signal.butter(order, cuttoff[0]/(fs/2), filtype)
    return b, a 

def apply_filter(sig, b, a):
    sig -= sig.mean()
    sig = min_max_scale(sig)
    return signal.filtfilt(b, a, sig)

def min_max_scale(x):
    # this function alters original x which is undesirable --> change this func 
    x -= x.min()
    x = x/x.max()
    return x

def init_pipelines(pipeline_names = ['csp+lda'], n_components = 4):
    pipelines = {}

    # standard / Laura's approach
    pipelines["csp+lda"] = Pipeline(steps=[('csp', CSP(n_components=n_components)), 
                                    ('lda', LDA())])
                                    
    # Using Ledoit-Wolf lemma shrinkage, which is said to outperform standard LDA
    pipelines["csp+s_lda"] = Pipeline(steps=[('csp', CSP(n_components=n_components)), 
                                    ('slda', LDA(solver = 'lsqr', shrinkage='auto'))])

    # Minimum distance to riemanian mean (MDRM) --> directly on the manifold
    # the RMDM approach is parameter-free!!
    pipelines["mdm"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                   ('mdm', MDM(metric="riemann"))])

    # LDA in the tangent space (projection of the data in the tangent space + LDA)
    pipelines["tgsp+lda"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                        ('tg', TangentSpace(metric="riemann")),
                                        ('lda', LDA())])

    # Grid-search pipelines
    parameters = {'l1_ratio': [0.2, 0.5, 0.8],
                'C': np.logspace(-1, 1, 3)}
    elasticnet = GridSearchCV(LogisticRegression(penalty='elasticnet', solver='saga'),
                            parameters)

    pipelines["csp+en"] = Pipeline(steps=[('csp', CSP(n_components=8)),
                                        ('en', elasticnet)])
    pipelines["tgsp+en"] = Pipeline(steps=[('cov', Covariances("oas")), 
                                        ('tg', TangentSpace(metric="riemann")),
                                        ('en', elasticnet)])


    chosen_pipelines = {}
    for pipeline_name in pipeline_names:
        chosen_pipelines[pipeline_name] = pipelines[pipeline_name]

    return chosen_pipelines

def signal_segmentation(dataset, ):
    filtered_dataset = pd.DataFrame()
    for electrode in selected_electrodes_names:
        for f in range(len(filters)):
            b, a = filters[f] #TODO how to live-update these?

            end = False
            i = 0
            filter_results = []
            while not end:
                if i+sample_duration < len(dataset):
                        segment_relax = data_relax[i:i+sample_duration][electrode]
                        segment_MI= data_MI[i:i+sample_duration][electrode]
                        #plt.plot(segment)
                        #plt.show()
                        #print(segment)

                        filt_result_relax = utils.apply_filter(segment_relax,b,a)
                        filt_result_MI = utils.apply_filter(segment_relax,b,a)
                        for data_point in filt_result_relax:
                            filter_results.append(data_point)
                        i += sample_duration
                else: 
                        end = True        
            filtered_dataset[electrode + '_' + freq_limits_names[f]] = filter_results










#------------
# below not used  
def sig_vid(path):
    X = []
    Y = []
    n_session = len(os.listdir(os.path.join(path)))//2
    for i in range(1, n_session+1):
        sig_path = glob.glob(os.path.join(path, 'session_'+str(i)+'_sig*'))[0]
        log_path = glob.glob(os.path.join(path, 'session_'+str(i)+'_log*'))[0]

        log = np.loadtxt(log_path)
        sig = np.loadtxt(sig_path)

        for vid in np.unique(log[:, -1]):
            if vid != 0 : #if intro
                
                x = sig[log[:, -1]==vid]
                y = log[log[:, -1]==vid]

                X.append(np.asarray(x))
                Y.append(np.asarray(y))
    return X, Y

def gen_feat(signals, label, filters):
    Features = np.zeros((1153, 8, 4))
    Label = np.zeros((1153))
    i = 0
    for vid in range(len(signals)):
        s = signals[vid][:, 1:9] #we keep only the EEG channels 
        l = label[vid][:, -1].astype(int)
        for e in range(s.shape[-1]): #for each electrodes
            for f in range(len(filters)): 
                b, a = filters[f] 
                x = apply_filter(s[:, e], b, a)
                x = signal_segmentation(x)
                for t in range(x.shape[0]):
                    Features[i+t, e, f] = compute_entropy(x[t])
                    Label[i+t] = l[0]
        i += t
    return Features, Label

def gen_val_arousal(vid_info, film_id):
    y = []
    for f in film_id:
        y.append([ float(vid_info[f-1][2]), float(vid_info[f-1][4])])
    y = np.asarray(y)
    tmp = np.vstack(
        ((y[:, 0] > np.median(y[:, 0])).astype(int), 
        ((y[:, 1] > np.median(y[:, 1])).astype(int))))
    y = np.transpose(tmp)
    return y

def gen_smiley(emotion, dim=32, dir_path='save_dir'):
    img = np.zeros((dim, dim))
    smileys = np.load(os.path.join(dir_path, 'em_smileys.npy'), allow_pickle=True).all()

    if dim != 32:
        print("The smileys have been computed for 32x32 display.")
        assert(False)

    id_em = smileys[emotion]

    for coord in id_em:
        img[coord[0], coord[1]] = 1

    return img

def custom_mean(x, y, weight):
    return (x+weight*y)/(weight+1)

def gen_figures(path, valence, arousal, old_val, old_ars):
    #figure infors
    fig = plt.figure(figsize=(6.25, 6.25))
    r = np.linspace(-1, 1, 1000)
    fontdict = {'color':  'darkred',
        'weight': 'light',
        'size': 8.5,
        }

    plt.plot(r, -np.sqrt(1-r**2), c='black')
    plt.plot(r, np.sqrt(1 -r**2), c='black')
    plt.arrow(0, -1.15, 0, 2.25, width=0.0085, color='black')
    plt.arrow(-1.15, 0, 2.25, 0, width=0.0085, color='black')
    plt.text(1.02, 0.075, 'Valence', fontdict)
    plt.text(0.075,1.075, 'Arousal', fontdict)
    ax = plt.gca()
    ax.set_xlim([-1.175, 1.175])
    ax.set_ylim([-1.175, 1.175])
    plt.grid(visible=True)

    ax.scatter(valence-0.5, arousal-0.5)
    plt.savefig(os.path.join(path, 'current_graph'))
    old_val = valence
    old_ar  = arousal

    if valence > 0.5:
        if arousal > 0.5:
            shutil.copy(os.path.join(path, 'ha_hv.png'), os.path.join(path, 'current_smiley.png'))
        else:
            shutil.copy(os.path.join(path, 'la_hv.png'), os.path.join(path, 'current_smiley.png'))
    elif arousal > 0.5:
        shutil.copy(os.path.join(path, 'ha_lv.png'), os.path.join(path, 'current_smiley.png'))
    else: 
        shutil.copy(os.path.join(path, 'la_lv.png'), os.path.join(path, 'current_smiley.png'))
    ax.clear()
    return valence, arousal

'''
Converting numpy raw files to mne raw see https://mne.tools/stable/generated/mne.io.RawArray.html

import mne 
def convert_mne(sig, f_s=50, c_names=['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'Oz', 'PO8']):
    info = mne.create_info(ch_names=e_name, sfreq=f_s, ch_types='eeg')
    return mne.io.RawArray(sig, info)
'''
