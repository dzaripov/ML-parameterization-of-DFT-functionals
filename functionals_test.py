import pickle

import numpy as np
import torch
from torch import nn

from dataset import collate_fn
from predopt import true_constants_PBE
from predopt_train import FCHEM_VALIDATION as FCHEM_FACTORS
from predopt_train import (
    Dataset,
    exc_loss,
    extend_bases,
    loss_function,
    make_total_db_errors,
)
from prepare_data import load_chk
from reaction_energy_calculation import calculate_reaction_energy
from utils import set_random_seed

device = torch.device("cpu")

rung = "LDA"
dft = "XALPHA"

criterion = nn.MSELoss()


set_random_seed(41)

data, data_train, data_test = load_chk(path="checkpoints")


train_set = Dataset(data=data_train)
train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=6,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    collate_fn=collate_fn,
)


test_set = Dataset(data=data_test)
test_dataloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=6,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    collate_fn=collate_fn,
)
total_set = Dataset(data=data)
total_dataloader = torch.utils.data.DataLoader(
    total_set,
    batch_size=6,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    collate_fn=collate_fn,
)

mae = nn.L1Loss()
mse = nn.MSELoss()

lst = []
local_lst = []
names = {
    0: "Train",
    1: "Test",
    2: "Total",
}

with open("./dispersions/dispersions.pickle", "rb") as handle:
    dispersions = pickle.load(handle)

with torch.no_grad():
    for index, dataset in enumerate(
        [train_dataloader, test_dataloader, total_dataloader]
    ):

        local_lst = []
        pred_energies = []
        ref_energies = []
        bases = []
        errors = []
        total_database_errors = {}

        for batch_idx, (X_batch, y_batch) in enumerate(dataset):

            current_bases, bases = extend_bases(X_batch, bases)
            grid_size = len(X_batch["Grid"])
            constants = (torch.ones(grid_size) * 1.05).view(grid_size, 1)

            val = index == 1

            energies, local_energies = calculate_reaction_energy(
                X_batch,
                constants,
                device,
                rung="LDA",
                dft="XALPHA",
                dispersions=dispersions,
            )
            local_loss = exc_loss(
                X_batch, constants, local_energies, dft="XALPHA", val=val
            )
            pred_energies, ref_energies, errors, total_database_errors = (
                make_total_db_errors(
                    pred_energies,
                    energies,
                    errors,
                    ref_energies,
                    y_batch,
                    total_database_errors,
                    current_bases,
                )
            )

            local_lst.append(torch.sqrt(local_loss).item())

        fchem = loss_function(FCHEM_FACTORS, total_database_errors)

        print("Fchem =", fchem)

        print(f"XAlpha {names[index]} MAE =", mae(pred_energies, ref_energies).item())
        print(f"XAlpha {names[index]} Local Loss =", np.mean(np.array(local_lst)))


with torch.no_grad():
    for index, dataset in enumerate(
        [train_dataloader, test_dataloader, total_dataloader]
    ):
        local_lst = []
        pred_energies = []
        ref_energies = []
        bases = []
        total_database_errors = {}

        for batch_idx, (X_batch, y_batch) in enumerate(dataset):

            grid_size = len(X_batch["Grid"])
            current_bases, bases = extend_bases(X_batch, bases)
            constants = (torch.ones(grid_size * 24)).view(
                grid_size, 24
            ) * true_constants_PBE
            energies, local_energies = calculate_reaction_energy(
                X_batch,
                constants,
                device,
                rung="GGA",
                dft="PBE",
                dispersions=dispersions,
            )
            pred_energies, ref_energies, errors, total_database_errors = (
                make_total_db_errors(
                    pred_energies,
                    energies,
                    errors,
                    ref_energies,
                    y_batch,
                    total_database_errors,
                    current_bases,
                )
            )

        fchem = loss_function(FCHEM_FACTORS, total_database_errors)

        print("Fchem =", fchem)

        print(f"PBE {names[index]} MAE =", mae(pred_energies, ref_energies).item())
