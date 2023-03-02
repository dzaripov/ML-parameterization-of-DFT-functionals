from torch import nn
import torch


"""
Define an nn.Module class for a simple residual block with equal dimensions
"""
class ResBlock(nn.Module):

    """
    Iniialize a residual block with two FC followed by (batchnorm + relu + dropout) layers 
    """
    def __init__(self, h_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2))
        
    def forward(self, x):
        residue = x

        return self.fc(self.fc(x)) + residue # skip connection


class MLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants, DFT=None):
        super().__init__()

        self.DFT = DFT
        
        modules = []
        modules.extend([nn.Linear(7, h_dim, bias=False),
                        nn.BatchNorm1d(h_dim),
                        nn.LeakyReLU(),
                        nn.Dropout(p=0.0)])
        
        for _ in range(num_layers // 2 - 1):
            modules.append(ResBlock(h_dim))
            
        modules.append(nn.Linear(h_dim, nconstants, bias=True))

        self.hidden_layers = nn.Sequential(*modules)


    def forward(self, x):
        x = self.hidden_layers(x)
        if self.DFT == 'SVWN': # constraint for VWN3's Q_vwn function to 4*c - b**2 > 0
            constants = []
            for b_ind, c_ind in zip((2, 3, 11, 12, 13),(4, 5, 14, 15, 16)):
                 constants.append(torch.abs(x[:, c_ind]) + (x[:, b_ind]**2)/4 + 1e-5) # torch.abs or nn.ReLU()?
            x = torch.cat([x[:,0:4], torch.stack(constants[0:2], dim=1), x[:,6:14], torch.stack(constants[2:], dim=1), x[:,17:]], dim=1)
            del constants
        if self.DFT == 'PBE':
            constants = torch.abs(x[:, 0:21]) + 1e-5
            x = torch.cat([constants, x[:, 21:24]], dim=1) 
            del constants
        return x


def NN_2_256(num_layers=2, h_dim=256, nconstants=None, DFT=None):
    return MLOptimizer(num_layers, h_dim, nconstants, DFT)


def NN_4_256(num_layers=4, h_dim=256, nconstants=None, DFT=None):
    return MLOptimizer(num_layers, h_dim, nconstants, DFT)


def NN_8_256(num_layers=8, h_dim=256, nconstants=None, DFT=None):
    return MLOptimizer(num_layers, h_dim, nconstants, DFT)


def NN_8_64(num_layers=8, h_dim=64, nconstants=None, DFT=None):
    return MLOptimizer(num_layers, h_dim, nconstants, DFT)



class ResBlockTest(nn.Module):

    """
    Iniialize a residual block with two FC followed by (batchnorm + relu + dropout) layers 
    """
    def __init__(self, h_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.BatchNorm1d(h_dim),
            nn.LeakyReLU())

    def forward(self, x):
        residue = x

        return self.fc(self.fc(x)) + residue # skip connection




class MLOptimizerTest(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants, DFT=None):
        super().__init__()

        self.DFT = DFT
        
        modules = []
        modules.extend([nn.Linear(7, h_dim, bias=False),
                        nn.BatchNorm1d(h_dim),
                        nn.LeakyReLU()])
        
        for _ in range(num_layers // 2 - 1):
            modules.append(ResBlockTest(h_dim))
            
        modules.append(nn.Linear(h_dim, nconstants, bias=True))

        self.hidden_layers = nn.Sequential(*modules)
                       
    def forward(self, x):
        x = self.hidden_layers(x)
        if self.DFT == 'SVWN':
            constants = []
            for b_ind, c_ind in zip((2, 3, 11, 12, 13),(4, 5, 14, 15, 16)):
                 constants.append(torch.abs((x[:, c_ind])) + (x[:, b_ind]**2)/4 + 1e-5)
            x = torch.cat([x[:,0:4], torch.stack(constants[0:2], dim=1), x[:,6:14], torch.stack(constants[2:], dim=1), x[:,17:]], dim=1)
            del constants
        if self.DFT == 'PBE':
            constants = torch.abs(x[:, 3:24]) + 1e-5
            x = torch.cat([x[:, 0:3], constants, x[:, 24:27]], dim=1) 
            del constants
        return x


def test(num_layers=8, h_dim=256, nconstants=21, DFT=None):
    return MLOptimizerTest(num_layers, h_dim, nconstants, DFT)