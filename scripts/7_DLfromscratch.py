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
import src.utils_DLscratch as utils_DLscratch

pd.options.mode.chained_assignment = None  # default='warn'

def execution(type):
    print(f'Initializing for scratch DL...')
    # step 1: Run 3 times, save best: pre-train all DL models for whole population minus 1 w 3 random validation sets?
    # step 2: Run 3 times, get avg: finetune for 3 epochs on 1, 2, 3, 4, or 5 trials w random validation set, test on remaining 5
    # step 3: also train CSP and RG model for 1,2,3,4,or 5 trials --> random validation set?
    # step 4: test ML models on remaining as well
    utils_DLscratch.run()
    print('Finished')

def main():
    for type in FLAGS.type:
        print(type)
        execution(type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run offline BCI analysis experiments.")
    parser.add_argument("--type", nargs='+', default=['multiclass'], help="The variant of pipelines used. \
    This variable is a list containing the name of the variants. Options are: 'multiclass'.")
    FLAGS, unparsed = parser.parse_known_args()
    main()