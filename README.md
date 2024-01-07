# NN parametrization of DFT functionals

## Train and validate NN models with PBE and XAlpha functionals on M06-2X dataset


### Reproducing the experiment:

1) Clone the repository
2) Create and activate virtual environment, install packages from requirements.txt
3) To train and validate the models, run the following command with chosen values:
```
python predopt_train.py {XALPHA or PBE}_{number of hidden layers}_{number of neurons in a layer} {number of preoptimization epochs} {number of training epochs} {batch size} {dropout rate} {Ω value}
```
For example, to obtain the trained NN_PBE and NN_XAlpha, run:
```
python predopt_train.py PBE_8_32 50 184 8 0.4 0.0412
python predopt_train.py XALPHA_32_32 100 853 6 0.4 0.206
```
The model's state is saved every 10 epochs and at the end of training in mlruns

4) You can run the script with various Ω values and save the outputs in the format {XALPHA, PBE}_{number of hidden weight layers}_{number of neurons in a layer}_{Ω  value}_{other information if needed}.out
5) Then, put the .out files in one folder and run:
```
python visualize.py {path to the folder containing .out files}
```
To reproduce Figures S1-S14
6) To test PBE and XAlpha functionals of M06-2X train and test dataset, run:
```
python functionals_test.py
```