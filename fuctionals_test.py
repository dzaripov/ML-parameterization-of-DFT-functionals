import random

import numpy as np
import torch
from torch import nn

from predopt import true_constants_PBE
from prepare_data import load_chk
from reaction_energy_calculation import (calculate_reaction_energy,
                                         get_local_energies)

device = torch.device("cpu")

rung = "LDA"
dft = "XALPHA"


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def exc_loss(reaction, pred_constants, dft="PBE", true_constants=true_constants_PBE):
    HARTREE2KCAL = 627.5095
    backsplit_ind = reaction["backsplit_ind"].to(torch.int32)
    indices = list(
        zip(
            torch.hstack((torch.tensor(0).to(torch.int32), backsplit_ind)),
            backsplit_ind,
        )
    )
    n_molecules = len(indices)
    loss = torch.tensor(0.0, requires_grad=True)
    predicted_local_energies = get_local_energies(
        reaction, pred_constants, device, rung=rung, dft=dft
    )["Local_energies"]
    predicted_local_energies = [
        predicted_local_energies[start:stop] for start, stop in indices
    ]
    true_local_energies = get_local_energies(
        reaction, true_constants, device, rung="GGA", dft="PBE"
    )["Local_energies"]
    true_local_energies = [true_local_energies[start:stop] for start, stop in indices]
    for i in range(n_molecules):
        loss += (
            1
            / len(predicted_local_energies[i])
            * torch.sqrt(
                torch.sum((predicted_local_energies[i] - true_local_energies[i]) ** 2)
            )
        )

    return loss * HARTREE2KCAL / n_molecules


set_random_seed(41)

data, data_train, data_test = load_chk(path="checkpoints")


from dataset import collate_fn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        self.data[i].pop("Database", None)
        return self.data[i], self.data[i]["Energy"]

    def __len__(self):
        return len(self.data.keys())


train_set = Dataset(data=data_train)
train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    collate_fn=collate_fn,
)


test_set = Dataset(data=data_test)
test_dataloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    collate_fn=collate_fn,
)


mae = nn.L1Loss()

lst = []
local_lst = []
names = {
    0: "Train",
    1: "Test",
}
with torch.no_grad():
    for index, dataset in enumerate([train_dataloader, test_dataloader]):
        lst = []
        local_lst = []
        for batch_idx, (X_batch, y_batch) in enumerate(dataset):
            grid_size = len(X_batch["Grid"])
            constants = (torch.ones(grid_size) * 1.05).view(grid_size, 1)
            local_loss = exc_loss(X_batch, constants, dft="XALPHA")
            energies = calculate_reaction_energy(
                X_batch, constants, device, rung="LDA", dft="XALPHA", dispersions=dict()
            )
            lst.append(mae(energies, y_batch).item())
            local_lst.append(local_loss.item())
        print(f"XAlpha {names[index]} MAE =", np.mean(np.array(lst)))
        print(f"XAlpha {names[index]} Local Loss =", np.mean(np.array(local_lst)))

# XAlpha Train MAE = 16.477375944478567
# XAlpha Train Local Loss = 0.11192005753170613
# XAlpha Test MAE = 17.319649955401054
# XAlpha Test Local Loss = 0.11970577926303332

with torch.no_grad():
    for index, dataset in enumerate([train_dataloader, test_dataloader]):
        lst = []
        local_lst = []
        for batch_idx, (X_batch, y_batch) in enumerate(dataset):
            grid_size = len(X_batch["Grid"])
            constants = (torch.ones(grid_size * 24)).view(
                grid_size, 24
            ) * true_constants_PBE
            constants = constants
            energies = calculate_reaction_energy(
                X_batch, constants, device, rung="GGA", dft="PBE", dispersions=dict()
            )
            lst.append(mae(energies, y_batch).item())
        print(f"PBE {names[index]} MAE =", np.mean(np.array(lst)))

# PBE Train MAE = 7.857935028800438
# PBE Test MAE = 7.648924972360524
