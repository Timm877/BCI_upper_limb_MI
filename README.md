# Low-cost BCI for upper-limb motor imagery detection
In this project, a pipeline will be developed to accurately detect motor imagery of the upper limbs during various mental tasks.

## Description
Brain-machine interfaces can be very useful for physically-impaired users.
Earlier, a pipeline for upper limb motor imagery has been developed using a SVM-based pipeline (see [here](https://www.sciencedirect.com/science/article/pii/S092523121401323X)), which was used to control movements of a robot arm. 
In this project, this pipeline will be re-avaluated and tried to improve.

## Installation instructions
```
conda env create -f environment.yml
```

## Usage instruction
```
python scripts/main.py
```

## References
[SVM-based Brain–Machine Interface for controlling a robot arm
through four mental tasks](https://www.sciencedirect.com/science/article/pii/S092523121401323X)