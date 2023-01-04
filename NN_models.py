from torch import nn
import torch
from copy import deepcopy



"""
Define an nn.Module class for a simple residual block with equal dimensions
"""
class ResBlock(nn.Module):

    """
    Iniialize a residual block with two FC followed by (batchnorm + relu + dropout) layers 
    """
    def __init__(self, h_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2))
        
        self.fc2 = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2))

    def forward(self, x):
        residue = deepcopy(x)
        x = self.fc1(x)
        x = self.fc2(x)
            
        return x + residue # skip connection
    
    

class MLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants, DFT=None):
        super().__init__()

        self.nconstants = nconstants
        self.num_layers = num_layers
        self.DFT = DFT
        
        modules = []
        modules.extend([nn.Linear(7, h_dim, bias=False),
                        nn.BatchNorm1d(h_dim),
                        nn.LeakyReLU(),
                        nn.Dropout(p=0.1)])
        
        for _ in range(num_layers // 2):
            modules.append(ResBlock(h_dim))
            
        modules.append(nn.Linear(h_dim, nconstants, bias=True))

        self.hidden_layers = nn.Sequential(*modules)


    def forward(self, x):
        x = self.hidden_layers(x)
        if self.DFT == 'SVWN':
            constants = []
            for b_ind, c_ind in zip((2, 3, 11, 12, 13),(4, 5, 14, 15, 16)):
                 constants.append(nn.ReLU()(x[:, c_ind]) + (x[:, b_ind]**2)/4 + 1e-5)
            x = torch.cat([x[:,0:4], torch.stack(constants[0:2], dim=1), x[:,6:14], torch.stack(constants[2:], dim=1), x[:,17:]], dim=1)
            del constants
        return x


def NN_2_256(num_layers=2, h_dim=256, nconstants=21, DFT=None):
    return MLOptimizer(num_layers, h_dim, nconstants, DFT)

    
    

class MLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants, DFT=None):
        super().__init__()

        self.nconstants = nconstants
        self.num_layers = num_layers
        self.DFT = DFT
        
        modules = []
        modules.extend([nn.Linear(7, h_dim, bias=False),
                        nn.BatchNorm1d(h_dim),
                        nn.LeakyReLU(),
                        nn.Dropout(p=0.1)])
        
        for _ in range(num_layers // 2):
            modules.append(ResBlock(h_dim))
            
        modules.append(nn.Linear(h_dim, nconstants, bias=True))

        self.hidden_layers = nn.Sequential(*modules)


    def forward(self, x):
        x = self.hidden_layers(x)
        if self.DFT == 'SVWN':
            constants = []
            for b_ind, c_ind in zip((2, 3, 11, 12, 13),(4, 5, 14, 15, 16)):
                 constants.append(nn.ReLU()(x[:, c_ind]) + (x[:, b_ind]**2)/4 + 1e-5)
            x = torch.cat([x[:,0:4], torch.stack(constants[0:2], dim=1), x[:,6:14], torch.stack(constants[2:], dim=1), x[:,17:]], dim=1)
            del constants
        return x


def NN_2_256(num_layers=2, h_dim=256, nconstants=21, DFT=None):
    return MLOptimizer(num_layers, h_dim, nconstants, DFT)


def NN_8_256(num_layers=8, h_dim=256, nconstants=21, DFT=None):
    return MLOptimizer(num_layers, h_dim, nconstants, DFT)


def NN_8_64(num_layers=8, h_dim=64, nconstants=21, DFT=None):
    return MLOptimizer(num_layers, h_dim, nconstants, DFT)

