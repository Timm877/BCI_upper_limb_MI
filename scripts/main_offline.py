import os
from pathlib import Path
from src.utils import apply_filters, create_pipeline
import argparse

def main():
     # init stuff
     sampling_frequency = 200 #250 for ours, 200 for Laura's
     sample_duration = 100 #0.5 seconds?
     selected_electrodes_names= ['F3', 'FZ', 'FC1','FCZ','C1','CZ','CP1','CPZ', 
     'FC5', 'FC3','C5','C3','CP5','CP3','P3','PZ','F4','FC2','FC4','FC6','C2',
     'C4','CP2','CP4','C6','CP6','P4','HR' ,'HL', 'VU', 'VD']
     n_electrodes = len(selected_electrodes_names) #27? for Laura's, 8 for ours


     folder_path = Path('./data/offline/')
     result_path = Path('./data/offline/intermediate_datafiles/1/')
     result_path.mkdir(exist_ok=True, parents=True)

     for instance in os.scandir(folder_path): # go through all data files  
          instance_path = instance.path

     # apply FB (filter bank)
     #sig_filtered = apply_filters(sig, filter parser arguments)
     # csp+lda pipeline
     #pipeline = create_pipeline(pipeline parser arguments)

     # be excited
     # :)


if __name__ == '__main__':
     parser = argparse.ArgumentParser(description="Train a neural net")
     parser.add_argument("--test", type=int, default=1000, help="just an empty parser for now")

     main()
