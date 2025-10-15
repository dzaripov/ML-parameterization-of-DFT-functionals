# NN parametrization of DFT functionals


## Preparing data for parametrization


1) Clone the repository
2) Create and activate virtual environment, install packages from requirements.txt, download the dataset from https://github.com/TheorChemGroup/MN_neuromorphic_dataset.git
3) Create /*data* folder in the root directory, load the *.h5* files in it
4) Run `prepare_data.py` to split the dataset into train and validation and save it in ./*checkpoints*

## Parametrization
To train and validate the models, run `calculations.py` with suitable parameters.


## Testing the non-parametrized functionals on the dataset
To test PBE and XAlpha functionals of M06-2X train and test dataset, run `functionals_test.py`



