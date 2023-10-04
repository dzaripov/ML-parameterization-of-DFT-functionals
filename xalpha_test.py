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
from NN_models import NN_2_256, NN_4_256, NN_8_256, NN_8_64, NN_4_128, NN_8_128, MLOptimizer
from reaction_energy_calculation import calculate_reaction_energy, stack_reactions, get_local_energies_x, get_local_energies_c, backsplit
from prepare_data import prepare, save_chk, load_chk


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
                                               batch_size=1,
                                               num_workers=4,
                                               pin_memory=True,
                                               shuffle=True, 
                                               collate_fn=collate_fn)

test_set = Dataset(data=data_test)
test_dataloader = torch.utils.data.DataLoader(test_set, 
                                              batch_size=1,
                                              num_workers=4,
                                              pin_memory=True,
                                              shuffle=True, 
                                              collate_fn=collate_fn)

#device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')
device = torch.device('cpu')


mae = nn.L1Loss()

lst = []
for batch_idx, (X_batch, y_batch) in enumerate(train_dataloader):
    grid_size = len(X_batch["Grid"])
    constants = torch.tensor([torch.tensor([1.05]) for _ in range(grid_size)]).view(grid_size,1)
    energies = calculate_reaction_energy(X_batch, constants, device, rung='LDA', dft='XALPHA').to(device)
    lst.append(mae(energies, y_batch).item())
print("Train MAE =", np.mean(np.array(lst)))

lst = []
for batch_idx, (X_batch, y_batch) in enumerate(test_dataloader):
    grid_size = len(X_batch["Grid"])
    constants = torch.tensor([torch.tensor([1.05]) for _ in range(grid_size)]).view(grid_size,1)
    energies = calculate_reaction_energy(X_batch, constants, device, rung='LDA', dft='XALPHA').to(device)
    MAE = mae(energies, y_batch).item()
    lst.append(mae(energies, y_batch).item())
print("Test MAE =", np.mean(np.array(lst)))

