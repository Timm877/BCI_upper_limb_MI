# Low-cost BCI for upper-limb motor imagery detection

## Description
Brain-machine interfaces (BMIs) can be very useful for physically-impaired users.
Upper-limb motor imagery BMIs can be used to, for example, control movements of a robotic arm. 
In this project, open loop (data collection) as well as closed loop (real-time) experiments were done using a low-cost EEG device.
In the end, user could play a game of Space Invaders in real-time using upper-limb motor imagery.

## Installation instructions
```
# Create a virtual environment:
conda create --name BCI --python=3.9
conda activate BCI
# Install src to run scripts in src folder:
pip install -e .
# Install required packages:
pip install -r requirements.txt

# For running experiments with psychopy with data collection using pylsl:
conda install -c conda-forge psychopy
pip install pylsl
```

## Usage instruction
All scripts for open loop are visible in the scripts folder.
Scripts can be ran by using the command line, as example for pre-processing of subject X01 for CSP pipeline:
```
python scripts/openloop_datacollect/1_pre.py --subjects X01 --pline csp
```
Note that for above scripts, a data folder is needed, creating by running experiments or download data.
Please check if the path is correct.
Open loop scripts are found in 'openloop_datacollect'.
Closed loop scripts are found in 'realtime_exp'.