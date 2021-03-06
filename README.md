# Low-cost BCI for upper-limb motor imagery detection
In this project, a pipeline will be developed to accurately detect motor imagery of the upper limbs during various mental tasks.

## Description
Brain-machine interfaces can be very useful for physically-impaired users.
Upper limb motor imagery BMIs can be used to control movements of a robotic arm. 
In this project, open loop and closed loop experiments for controlling a dot on the screen are done.
Later, the goal is to implement the pipilines for control of a robotic arm.

## Installation instructions
```
#Create a virtual environment:
conda create --name BCI --python=3.9
conda activate BCI
# install src to run scripts in src folder:
pip install -e .
# install required packages:
pip install -r requirements.txt --user
# for running experiments with psychopy with data collection using pylsl:
conda install -c conda-forge psychopy
pip install pylsl
```

## Usage instruction
All scripts for open loop are visible in the scripts folder.
Scripts can be ran by using the command line, as example for pre-processing of subject X01 for CSP pipeline:
```
python scripts/1_pre.py --subjects X01 --pline csp
```
Note that for above scripts, a data folder is needed, creating by running experiments or download data.
Please check if the path is correct.
All scripts for doing the experiments, for open loop as well as closed loop, can be found in the cl folder.
Here, when only open loop experiments are needed to execute, just only run scripts 1 to 3.
Scripts 4 to 7 are used for closed loop.
## References