import numpy as np
import torch
import gc
import random
import copy
import pickle
import mlflow
from torch import nn
from sklearn.metrics import mean_absolute_error
from tqdm.notebook import tqdm
from NN_models import MLOptimizer
from reaction_energy_calculation import calculate_reaction_energy, stack_reactions, get_local_energies_x, \
    get_local_energies_c, backsplit
from prepare_data import prepare, save_chk, load_chk
import matplotlib.pyplot as plt
import inspect
import sys


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def set_random_seed(seed):
    # seed everything
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(41)

data, data_train, data_test = load_chk(path='checkpoints')

with open('./dispersions/dispersions.pickle', 'rb') as handle:
    dispersions = pickle.load(handle)

from dataset import collate_fn


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
                                               batch_size=8,
                                               num_workers=4,
                                               pin_memory=True,
                                               shuffle=True,
                                               collate_fn=collate_fn)

test_set = Dataset(data=data_test)
test_dataloader = torch.utils.data.DataLoader(test_set,
                                              batch_size=8,
                                              num_workers=4,
                                              pin_memory=True,
                                              shuffle=True,
                                              collate_fn=collate_fn)

device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
criterion = nn.MSELoss()

from dataset import collate_fn_predopt
from predopt import DatasetPredopt, true_constants_PBE

train_predopt_set = DatasetPredopt(data=data, dft='PBE')
train_predopt_dataloader = torch.utils.data.DataLoader(train_predopt_set,
                                                       batch_size=8,
                                                       num_workers=4,
                                                       pin_memory=True,
                                                       shuffle=True,
                                                       collate_fn=collate_fn_predopt)

name, n_predopt, n_train = sys.argv[1:]

if name == 'PBE_8_32':
    model = MLOptimizer(num_layers=8, h_dim=32, nconstants=24, DFT='PBE').to(device)
elif name == "PBE_16_32":
    model = MLOptimizer(num_layers=16, h_dim=32, nconstants=24, DFT='PBE').to(device)
elif name == "PBE_32_32":
    model = MLOptimizer(num_layers=32, h_dim=32, nconstants=24, DFT='PBE').to(device)
elif name == "PBE_512_4":
    model = MLOptimizer(num_layers=512, h_dim=4, nconstants=24, DFT='PBE').to(device)
elif name == "PBE_4_512":
    model = MLOptimizer(num_layers=4, h_dim=512, nconstants=24, DFT='PBE').to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=0.01)

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


def log_params(model, metric1, metric2, name, predopt=False):
    with mlflow.start_run() as run:
        if predopt:
            metric1_name = "train_loss_mse"
            metric2_name = "train_loss_mae"
        else:
            metric1_name = "train_loss_mae"
            metric2_name = "test_loss_mae"
        mlflow.pytorch.log_model(model, name)
        mlflow.log_param("n_epochs", len(metric1))
        mlflow.log_metric(metric1_name, metric1[-1])
        mlflow.log_metric(metric2_name, metric2[-1])
        plt.plot(np.arange(len(metric1)), metric1, label=metric1_name)
        plt.plot(metric2, label=metric2_name)
        plt.legend()
        plt.xlabel("number of epochs")
        plt.ylabel("loss")
        plt.savefig(f"{name}.png")
        mlflow.log_artifact(f"{name}.png")
        os.remove(f"./{name}.png")


mlflow.set_experiment(name)
log_params(model, train_loss_mse, train_loss_mae, name=f"{name}_predopt", predopt=True)

torch.cuda.empty_cache()

from tqdm.notebook import tqdm
import os

log_file_path = 'log/epoch_training.log'

if os.path.isfile(log_file_path):
    os.remove(log_file_path)

mae = nn.L1Loss()


def exc_loss(reaction, pred_constants, true_constants=true_constants_PBE):
    backsplit_ind = reaction["backsplit_ind"].to(torch.int32)
    indices = list(zip(torch.hstack((torch.tensor(0).to(torch.int32), backsplit_ind)), backsplit_ind))
    n_molecules = len(indices)
    loss = torch.tensor(0., requires_grad=True).to(device)
    predicted_local_energies_x = get_local_energies_x(reaction, pred_constants, device, rung='GGA', dft='PBE')[
        "Local_energies"]
    predicted_local_energies_x = [predicted_local_energies_x[start:stop] for start, stop in indices]
    predicted_local_energies_c = get_local_energies_c(reaction, pred_constants, device, rung='GGA', dft='PBE')[
        "Local_energies"]
    predicted_local_energies_c = [predicted_local_energies_c[start:stop] for start, stop in indices]
    true_local_energies_x = get_local_energies_x(reaction, true_constants.to(device), device, rung='GGA', dft='PBE')[
        "Local_energies"]
    true_local_energies_x = [true_local_energies_x[start:stop] for start, stop in indices]
    true_local_energies_c = get_local_energies_c(reaction, true_constants.to(device), device, rung='GGA', dft='PBE')[
        "Local_energies"]
    true_local_energies_c = [true_local_energies_c[start:stop] for start, stop in indices]
    for i in range(n_molecules):
        loss += 1 / len(predicted_local_energies_x[i]) \
                * torch.sqrt(
            torch.sum((predicted_local_energies_x[i] - true_local_energies_x[i]) ** 2)
            + torch.sum((predicted_local_energies_c[i] - true_local_energies_c[i]) ** 2))
    return loss


def train(model, criterion, optimizer, train_loader, test_loader, n_epochs=25, accum_iter=1, verbose=False):
    train_loss_mae = []
    train_loss_mse = []
    test_loss_mae = []
    test_loss_mse = []
    omega = 0.067

    for epoch in range(n_epochs):
        torch.autograd.set_detect_anomaly(True)
        print(f'Epoch {epoch + 1}')
        # train
        model.train()
        progress_bar_train = tqdm(train_loader)
        train_mae_losses_per_epoch = []
        train_mse_losses_per_epoch = []
        optimizer.zero_grad()
        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar_train):
            X_batch_grid, y_batch = X_batch['Grid'].to(device), y_batch.to(device)
            predictions = model(X_batch_grid)
            reaction_energy = calculate_reaction_energy(X_batch, predictions, device, rung='GGA', dft='PBE',
                                                        dispersions=dispersions).to(device)

            if verbose:
                print(f"{X_batch['Components']} pred {reaction_energy.item():4f} true {y_batch.item():4f}")
            loss = (1 - omega) / 15 * criterion(reaction_energy, y_batch) + omega * 200 * exc_loss(X_batch, predictions)
            MSE = criterion(reaction_energy, y_batch).item()
            MAE = mae(reaction_energy, y_batch).item()
            train_mse_losses_per_epoch.append(MSE)
            train_mae_losses_per_epoch.append(MAE)
            progress_bar_train.set_postfix(MSE=MSE, MAE=MAE)

            # loss_accumulation
            loss = loss / accum_iter
            loss.backward()
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            del X_batch, X_batch_grid, y_batch, predictions, reaction_energy
            gc.collect()
            torch.cuda.empty_cache()

        train_loss_mse.append(np.mean(train_mse_losses_per_epoch))
        train_loss_mae.append(np.mean(train_mae_losses_per_epoch))

        print(f'train MSE Loss = {train_loss_mse[epoch]:.8f} MAE Loss = {train_loss_mae[epoch]:.8f}')

        # test
        model.eval()
        progress_bar_test = tqdm(test_loader)
        test_mae_losses_per_epoch = []
        test_mse_losses_per_epoch = []
        with torch.no_grad():
            for X_batch, y_batch in progress_bar_test:
                X_batch_grid, y_batch = X_batch['Grid'].to(device), y_batch.to(device)
                predictions = model(X_batch_grid)
                reaction_energy = calculate_reaction_energy(X_batch, predictions, device, rung='GGA', dft='PBE',
                                                            dispersions=dispersions).to(device)
                loss = criterion(reaction_energy, y_batch)
                MSE = loss.item()
                MAE = mae(reaction_energy, y_batch).item()
                test_mse_losses_per_epoch.append(MSE)
                test_mae_losses_per_epoch.append(MAE)
                progress_bar_test.set_postfix(MSE=MSE, MAE=MAE)
                del X_batch, X_batch_grid, y_batch, predictions, reaction_energy, loss, MAE, MSE
                gc.collect()
                torch.cuda.empty_cache()

        test_loss_mse.append(np.mean(test_mse_losses_per_epoch))
        test_loss_mae.append(np.mean(test_mae_losses_per_epoch))

        print(f'test MSE Loss = {test_loss_mse[epoch]:.8f} MAE Loss = {test_loss_mae[epoch]:.8f}')

    return train_loss_mae, test_loss_mae


from importlib import reload
import PBE
import reaction_energy_calculation
import utils
import NN_models

reload(NN_models)
reload(utils)
reload(reaction_energy_calculation)
reload(PBE)

optimizer = torch.optim.RMSprop(model.parameters(), lr=3e-5)

N_EPOCHS = n_train
ACCUM_ITER = 1
train_loss_mae, test_loss_mae = train(model, criterion, optimizer,
                                      train_dataloader, test_dataloader,
                                      n_epochs=N_EPOCHS, accum_iter=ACCUM_ITER)

log_params(model, train_loss_mae, test_loss_mae, name=f"{name}_train")
