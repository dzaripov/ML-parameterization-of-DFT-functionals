from torch import nn


class MLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants):
        super().__init__()

        self.nconstants = nconstants
        self.num_layers = num_layers
        modules = []
        modules.extend([nn.Linear(7, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.LeakyReLU(),
                        nn.Dropout(p=0.1)])
        for _ in range(num_layers-1):
            modules.extend([nn.Linear(h_dim, h_dim),
                            nn.BatchNorm1d(h_dim),
                            nn.LeakyReLU(),
                            nn.Dropout(p=0.2)])
        modules.append(nn.Linear(h_dim, nconstants))

        self.hidden_layers = nn.Sequential(*modules)


    def forward(self, x):
        x = self.hidden_layers(x)
        return x


def NN_2_256(num_layers=2, h_dim=256, nconstants=21):
    return MLOptimizer(num_layers, h_dim, nconstants)


def NN_8_256(num_layers=8, h_dim=256, nconstants=21):
    return MLOptimizer(num_layers, h_dim, nconstants)


def NN_8_64(num_layers=8, h_dim=64, nconstants=21):
    return MLOptimizer(num_layers, h_dim, nconstants)