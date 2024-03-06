# NN parametrization of DFT functionals

## Train and validate NN models with PBE and XAlpha functionals on M06-2X dataset


### Reproducing the experiment:

1) Clone the repository
2) Create and activate virtual environment, install packages from requirements.txt, download the dataset from https://github.com/TheorChemGroup/MN_neuromorphic_dataset.git
3) To train and validate the models, run the following command with chosen values:
```
python predopt_train.py --Name {XALPHA or PBE}_{number of hidden layers}_{number of neurons in a layer} --N_preopt {number of pre-optimization epochs} --N_train {number of training epochs} --Batch_size {value} --Dropout {value} --Omega {value}
```
For example, to obtain the trained NN_PBE and NN_XAlpha, run:
```
python predopt_train.py --Name PBE_8_32 --N_preopt 50 --N_train 184 --Batch_size 8 --Dropout 0.4 --Omega 0.0412
python predopt_train.py --Name XALPHA_32_32 --N_preopt 100 --N_train 853 --Batch_size 6 --Dropout 0.4 --Omega 0.206
```
The model's state is saved every 10 epochs and at the end of training in <em>mlruns</em> folder

4) Run the script with different Ω values and save the outputs in the format <em>{Name like in step 3}\_{Ω  value}\_{anything else (optional)}.out</em>
5) Put the .out files in one folder and run the following script to reproduce Figures 1-3:
```
python visualize.py {name of the folder containing .out files}
```

6) To test PBE and XAlpha functionals of M06-2X train and test dataset, run:
```
python functionals_test.py
```
