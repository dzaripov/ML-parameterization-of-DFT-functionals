import numpy as np
import torch
import gc
import random
from torch import nn
from sklearn.metrics import mean_absolute_error
from tqdm.notebook import tqdm
from NN_models import NN_2_256, NN_4_256, NN_8_256, NN_8_64, test # NN_8_128
from reaction_energy_calculation import calculate_reaction_energy
from prepare_data import prepare, save_chk, load_chk
import sys
import os


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

device = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

if sys.argv[1] == 'PBE_2_256':
    model = NN_2_256(nconstants=24, DFT='PBE').to(device)
elif sys.argv[1] == 'PBE_4_256':
    model = NN_4_256(nconstants=24, DFT='PBE').to(device)
elif sys.argv[1] == 'PBE_8_256':
    model = NN_8_256(nconstants=24, DFT='PBE').to(device)
elif sys.argv[1] == 'PBE_8_64':
    model = NN_8_64(nconstants=24, DFT='PBE').to(device)
elif sys.argv[1] == 'PBE_8_128':
    model = NN_8_128(nconstants=24, DFT='PBE').to(device)    
    

if os.path.exists(model_path:= f'model_chk/predopt/{sys.argv[1]}.param'):
    model.load_state_dict(torch.load(model_path, map_location=device))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))


from predopt import DatasetPredopt, predopt
train_predopt_set = DatasetPredopt(data=data, dft='PBE')
train_predopt_dataloader = torch.utils.data.DataLoader(train_predopt_set,
                                                       batch_size=None,
                                                       num_workers=1,
                                                       pin_memory=True,
                                                       shuffle=True)


predopt(model, criterion, optimizer, train_predopt_dataloader, device, n_epochs=50, accum_iter=10)