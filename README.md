# NN parametrization of DFT functionals


## Preparing data for parametrization


1) Clone the repository
2) Create and activate virtual environment, install packages from requirements.txt, download the dataset from https://github.com/TheorChemGroup/MN_neuromorphic_dataset.git
3) Create /*data* folder in the root directory, load the *.h5* files in it
4) Run `prepare_data.py` to split the dataset into train and validation and save it in ./*checkpoints*

## Parametrization
To train and validate the models, run `predopt_train.py` with suitable parameters.

The available parameters are:
- *Name* — should be in format {XALPHA or PBE}\_{number of hidden layers}\_{number of neurons in a layer}
- *N_preopt* — number of pre-optimization epochs
- *N_train* — number of training epochs
- *Batch_size*
- *Dropout*
- *Omega*


For example, to obtain the trained NN_PBE and NN_XAlpha, run:
```
python predopt_train.py --Name PBE_8_32 --N_preopt 50 --N_train 184 --Batch_size 8 --Dropout 0.4 --Omega 0.0412
python predopt_train.py --Name XALPHA_32_32 --N_preopt 100 --N_train 853 --Batch_size 6 --Dropout 0.4 --Omega 0.206
```
The model's state is saved every 10 epochs and at the end of training in <em>mlruns</em> folder


## Analyzing the results
1) Run the script with different Ω values and save the outputs in the format <em>{Name like in parametrization}\_{Dropout value}\_{Ω  value}\_{anything else}.out or .txt</em>
2) Put the .out files in one folder and run the following script to reproduce Figures 1-3:
```
python visualize.py {name of the folder containing .out files}
```


## Testing the non-parametrized functionals on the dataset
To test PBE and XAlpha functionals of M06-2X train and test dataset, run `functionals_test.py`
