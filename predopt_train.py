import numpy as np
import torch
import gc
import pickle
import mlflow
from torch import nn
from tqdm.notebook import tqdm
from NN_models import MLOptimizer
from reaction_energy_calculation import calculate_reaction_energy, get_local_energies, backsplit
from prepare_data import load_chk
import sys
from utils import log_params, set_random_seed
from dataset import collate_fn, collate_fn_predopt
from predopt import DatasetPredopt, true_constants_PBE



set_random_seed(41)

data, data_train, data_test = load_chk(path='checkpoints')

name, n_predopt, n_train, batch_size, dropout, omega = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])

if "PBE" in name:
    rung = "GGA"
    dft = "PBE"
    nconstants = 24
    lr_predopt = 3e-3
    lr_train = 3e-4
elif "XALPHA" in name:
    rung = "LDA"
    dft = "XALPHA"
    nconstants = 1
    lr_predopt = 6e-3
    lr_train = 2e-3

num_layers, h_dim = map(int, name.split("_")[1:])
device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
model = MLOptimizer(num_layers=num_layers, h_dim=h_dim, nconstants=nconstants, dropout=dropout, DFT=dft).to(device)


# load dispersion corrections
with open('./dispersions/dispersions.pickle', 'rb') as handle:
    dispersions = pickle.load(handle)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        self.data[i].pop('Database', None)
        return self.data[i], self.data[i]['Energy']

    def __len__(self):
        return len(self.data.keys())


train_set = Dataset(data=data_train)
train_dataloader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               pin_memory=True,
                                               shuffle=True,
                                               collate_fn=collate_fn)

test_set = Dataset(data=data_test)
test_dataloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              pin_memory=True,
                                              shuffle=True,
                                              collate_fn=collate_fn)

criterion = nn.MSELoss()


train_predopt_set = DatasetPredopt(data=data, dft=dft)
train_predopt_dataloader = torch.utils.data.DataLoader(train_predopt_set,
                                                       batch_size=batch_size,
                                                       num_workers=4,
                                                       pin_memory=True,
                                                       shuffle=True,
                                                       collate_fn=collate_fn_predopt)

mlflow.set_experiment(name)
name+="_"+str(dropout)

optimizer = torch.optim.Adam(model.parameters(), lr=lr_predopt, betas=(0.9, 0.999), weight_decay=0.01)

from importlib import reload
import predopt

reload(predopt)
import utils

reload(utils)
import dataset

reload(dataset)

from predopt import predopt

train_loss_mse, train_loss_mae = predopt(model, criterion, optimizer, train_predopt_dataloader, device, n_epochs=n_predopt,
                                         accum_iter=1)



log_params(model, train_loss_mse, train_loss_mae, name=f"{name}_predopt", predopt=True)

torch.cuda.empty_cache()

from tqdm.notebook import tqdm
import os


mae = nn.L1Loss()


def exc_loss(reaction, pred_constants, dft="PBE", true_constants=true_constants_PBE):
    HARTREE2KCAL = 627.5095

    # turn backsplit indices into slices
    backsplit_ind = reaction["backsplit_ind"].to(torch.int32)
    indices = list(zip(torch.hstack((torch.tensor(0).to(torch.int32), backsplit_ind)), backsplit_ind))
    n_molecules = len(indices)

    # initialize loss
    loss = torch.tensor(0., requires_grad=True).to(device)

    # calculate predicted local energies
    predicted_local_energies = get_local_energies(reaction, pred_constants, device, rung=rung, dft=dft)[
        "Local_energies"]
    
    # split them into systems
    predicted_local_energies = [predicted_local_energies[start:stop] for start, stop in indices]

    # calculate local PBE energies
    true_local_energies = get_local_energies(reaction, true_constants.to(device), device, rung='GGA', dft='PBE')[
        "Local_energies"]
    
    # split them into systems
    true_local_energies = [true_local_energies[start:stop] for start, stop in indices]

    # calculate local energy loss
    for i in range(n_molecules):
        loss += 1 / len(predicted_local_energies[i]) \
                *torch.sqrt(
                    torch.sum(
                        (predicted_local_energies[i] - true_local_energies[i]) ** 2
                    )
                )

    return loss*HARTREE2KCAL/n_molecules


def train(model, criterion, optimizer, train_loader, test_loader, n_epochs=25, accum_iter=1, verbose=False, omega = 0.067):
    train_loss_mae = []
    train_loss_mse = []
    train_loss_exc = []
    test_loss_mae = []
    test_loss_mse = []
    test_loss_exc = []


    for epoch in range(n_epochs):
        torch.autograd.set_detect_anomaly(True)
        print(f'Epoch {epoch + 1}')
        # train
        model.train()
        progress_bar_train = tqdm(train_loader)
        train_mae_losses_per_epoch = []
        train_mse_losses_per_epoch = []
        train_exc_losses_per_epoch = []
        optimizer.zero_grad()
        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar_train):
            X_batch_grid, y_batch = X_batch['Grid'].to(device), y_batch.to(device)
            predictions = model(X_batch_grid)
            reaction_energy = calculate_reaction_energy(X_batch, predictions, device, rung=rung, dft=dft,
                                                        dispersions=dispersions).to(device)

            if verbose:
                print(f"{X_batch['Components']} pred {reaction_energy.item():4f} true {y_batch.item():4f}")
            #calculate local ebergy loss
            local_loss = exc_loss(X_batch, predictions, dft=dft)

            # calculate mse loss
            reaction_mse_loss = criterion(reaction_energy, y_batch)

            # calculate total loss function
            loss = (1 - omega) / 5 * torch.sqrt(reaction_mse_loss) + omega * local_loss * 100
            MSE = reaction_mse_loss.item()
            MAE = mae(reaction_energy, y_batch).item()

            # log losses
            train_mse_losses_per_epoch.append(MSE)
            train_mae_losses_per_epoch.append(MAE)
            train_exc_losses_per_epoch.append(local_loss.item())
            progress_bar_train.set_postfix(MSE=MSE, MAE=MAE)

            loss.backward()
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            del X_batch, X_batch_grid, y_batch, predictions, reaction_energy
            gc.collect()
            torch.cuda.empty_cache()

        # log losses
        train_loss_mse.append(np.mean(train_mse_losses_per_epoch))
        train_loss_mae.append(np.mean(train_mae_losses_per_epoch))
        train_loss_exc.append(np.mean(train_exc_losses_per_epoch))
        print(f'train MSE Loss = {train_loss_mse[epoch]:.8f} MAE Loss = {train_loss_mae[epoch]:.8f}')
        print(f'train Local Energy Loss = {train_loss_exc[epoch]:.8f}')

        # test
        model.eval()
        progress_bar_test = tqdm(test_loader)
        test_mae_losses_per_epoch = []
        test_mse_losses_per_epoch = []
        test_exc_losses_per_epoch = []
        with torch.no_grad():
            for X_batch, y_batch in progress_bar_test:
                # same operations as in training, but without gradient calculations
                X_batch_grid, y_batch = X_batch['Grid'].to(device), y_batch.to(device)
                predictions = model(X_batch_grid)
                local_loss = exc_loss(X_batch, predictions, dft=dft)
                reaction_energy = calculate_reaction_energy(X_batch, predictions, device, rung=rung, dft=dft,
                                                            dispersions=dispersions).to(device)
                loss = criterion(reaction_energy, y_batch)
                MSE = loss.item()
                MAE = mae(reaction_energy, y_batch).item()
                test_mse_losses_per_epoch.append(MSE)
                test_mae_losses_per_epoch.append(MAE)
                test_exc_losses_per_epoch.append(local_loss.item())
                progress_bar_test.set_postfix(MSE=MSE, MAE=MAE)
                del X_batch, X_batch_grid, y_batch, predictions, reaction_energy, loss, MAE, MSE
                gc.collect()
                torch.cuda.empty_cache()

        test_loss_mse.append(np.mean(test_mse_losses_per_epoch))
        test_loss_mae.append(np.mean(test_mae_losses_per_epoch))
        test_loss_exc.append(np.mean(test_exc_losses_per_epoch))

        print(f'test MSE Loss = {test_loss_mse[epoch]:.8f} MAE Loss = {test_loss_mae[epoch]:.8f}')
        print(f'test Local Energy Loss = {test_loss_exc[epoch]:.8f}')

        # save our model every 10 epochs
        if (epoch+1)%10 == 0:
            log_params(model, train_loss_mae, test_loss_mae, name=f"{name}_train_epoch_{epoch+1}")


    return train_loss_mae, test_loss_mae

#DO WE NEED ALL OF IT?
from importlib import reload
import PBE
import reaction_energy_calculation
import utils
import NN_models

reload(NN_models)
reload(utils)
reload(reaction_energy_calculation)
reload(PBE)

optimizer = torch.optim.RMSprop(model.parameters(), lr=lr_train)

N_EPOCHS = n_train
ACCUM_ITER = 1
train_loss_mae, test_loss_mae = train(model, criterion, optimizer,
                                      train_dataloader, test_dataloader,
                                      n_epochs=N_EPOCHS, accum_iter=ACCUM_ITER, omega=omega)

# Log the final model
log_params(model, train_loss_mae, test_loss_mae, name=f"{name}_final")
