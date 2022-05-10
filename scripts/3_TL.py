import argparse
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from meegkit.asr import ASR
import pickle
import scipy.io
from scipy import signal
from sklearn.model_selection import KFold, cross_validate
import matplotlib.pyplot as plt
import src.unicorn_utils as utils
import src.utils_TL as utils_TL

pd.options.mode.chained_assignment = None  # default='warn'

def execution(pipeline_type, subject, type):
    print(f'Initializing for Transfer learning...')
    utils_TL.run()
    print('Finished')

def main():
    for subj in FLAGS.subjects:
        print(subj)
        for type in FLAGS.type:
            print(type)
            for pline in FLAGS.pline:
                print(pline)
                execution(pline, subj, type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--pline", nargs='+', default=['deep'], help="The variant of DL model used. \
    This variable is a list containing the name of the variants. Options are:  'deep' or 'deep_inception'")
    parser.add_argument("--subjects", nargs='+', default=['X01'], help="The subject to use as validation set.\
    Model will be trained on remaining subjects.")
    parser.add_argument("--type", nargs='+', default=['multiclass'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'multiclass'.")
    FLAGS, unparsed = parser.parse_known_args()
    main()