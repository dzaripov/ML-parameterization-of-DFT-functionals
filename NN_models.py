import torch
from torch import nn
import random
import numpy as np

random.seed(42)

device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

true_constants_PBE = torch.Tensor(
    [
        [
            0.06672455,
            (1 - torch.log(torch.Tensor([2]))) / (np.pi**2),
            1.709921,
            7.5957,
            14.1189,
            10.357,
            3.5876,
            6.1977,
            3.6231,
            1.6382,
            3.3662,
            0.88026,
            0.49294,
            0.62517,
            0.49671,
            # 1,  1,  1,
            0.031091,
            0.015545,
            0.016887,
            0.21370,
            0.20548,
            0.11125,
            -3 / 8 * (3 / np.pi) ** (1 / 3) * 4 ** (2 / 3),
            0.8040,
            0.2195149727645171,
            0.8040,
            0.2195149727645171,
        ]
    ]
).to(device)

# Расширенная версия с 27 параметрами (оригинальные 26 + новый eta)
true_constants_PBE_extended = torch.Tensor([
    [
        0.06672455,           # 0: mbeta
        (1 - torch.log(torch.Tensor([2]))) / (np.pi**2),  # 1: mgamma
        1.709921,              # 2: fz20
        7.5957, 14.1189, 10.357,  # 3-5: params_a_beta1
        3.5876, 6.1977, 3.6231,   # 6-8: params_a_beta2  
        1.6382, 3.3662, 0.88026,  # 9-11: params_a_beta3
        0.49294, 0.62517, 0.49671, # 12-14: params_a_beta4
        0.031091, 0.015545, 0.016887,  # 15-17: params_a_a
        0.21370, 0.20548, 0.11125,     # 18-20: params_a_alpha1
        -3/8*(3/np.pi)**(1/3)*4**(2/3), # 21: LDA_X_FACTOR
        0.8040, 0.2195149727645171,     # 22-23: kappa, mu
        0.8040, 0.2195149727645171,     # 24-25: дублирование
        1.0  # 26: НОВЫЙ ПАРАМЕТР eta (начальное значение)
    ]
]).to(device)

sigmoid = torch.nn.Sigmoid()
elu = torch.nn.ELU()

"""
Define an nn.Module class for a simple residual block with equal dimensions
"""
def asymptotic_constraint_operator(f_x, f_x0, f0):
    """
    Оператор ˆθ{x0, f0}[f](x) = f(x) - f(x0) + f0.
    """
    return f_x - f_x0 + f0

def lagrange_interpolation_operator(f_x_list, x, x0_list, l_func):
    """
    Обобщенный оператор интерполяции Лагранжа ˆ из формулы (2).
    """
    n_constraints = len(f_x_list)
    result = 0.0
    for i in range(n_constraints):
        term = f_x_list[i]
        for j in range(n_constraints):
            if j != i:
                numerator = l_func(x - x0_list[j])
                denominator = l_func(x0_list[i] - x0_list[j])
                term = term * (numerator / denominator)
        result += term
    return result

def l_func_tanh_norm(x):
    return torch.tanh(torch.norm(x, dim=1, keepdim=True) ** 2)

class ResBlock(nn.Module):
    """
    Iniialize a residual block with two FC followed by (batchnorm + relu + dropout) layers
    """

    def __init__(self, h_dim, dropout):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim, bias=False),
            nn.LayerNorm(h_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        residue = x
        out = self.fc(x)
        out = self.dropout(out + residue)
        return self.activation(out)


class NoResBlock(nn.Module):
    """
    Iniialize a residual block with two FC followed by (batchnorm + relu + dropout) layers
    """

    def __init__(self, h_dim, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):

        return self.block(x)


class MLOptimizer(nn.Module):
    def __init__(self, num_layers, h_dim, nconstants, dropout, DFT=None, constants=[]):
        super().__init__()

        self.DFT = DFT
        self.constants = constants
        if self.constants:
            nconstants = len(constants)

        modules = []
        modules.extend(
            [
                nn.Linear(7, h_dim, bias=False),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ]
        )

        for _ in range(num_layers // 2 - 1):
            modules.append(ResBlock(h_dim, dropout))

        modules.append(nn.Linear(h_dim, nconstants, bias=True))

        self.hidden_layers = nn.Sequential(*modules)

    def dm21_like_sigmoid(self, x):
        """
        Custom sigmoid translates from [-inf, +inf] to [0, 2]
        """

        exp = torch.exp(-0.5 * x)
        return 2 / (1 + exp)

    def unsymm_forward(self, x):

        x = self.hidden_layers(x)
        x = 1.05 * self.dm21_like_sigmoid(x)

        return x

    def forward(self, x):
        """
        Returns:
            spin-symmetrized enhancement factor for LDA exhange energy
        """

        result = (
            self.unsymm_forward(x) + self.unsymm_forward(x[:, [1, 0, 4, 3, 2, 6, 5]])
        ) / 2

        return result


class pcPBEMLOptimizer(nn.Module):
    def __init__(
        self, num_layers, h_dim, nconstants_x=2, nconstants_c=3, dropout=0.2, DFT=None ###было nconstants_c=2
    ):
        super().__init__()

        self.DFT = DFT

        modules_x = []  # NN part for exchange
        modules_c = []  # NN part for correlation

        input_layer_c = [
            nn.Linear(7, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        ]

        input_layer_x = [
            nn.Linear(2, h_dim, bias=False),
            nn.LayerNorm(h_dim),
            nn.GELU(),
        ]

        modules_x.extend(input_layer_x)
        modules_c.extend(input_layer_c)

        if num_layers // 2 - 1 > 1:
            for _ in range(num_layers // 2 - 1):
                modules_x.append(ResBlock(h_dim, dropout))
                modules_c.append(ResBlock(h_dim, dropout))
        else:
            modules_x.append(NoResBlock(h_dim, dropout))
            modules_c.append(NoResBlock(h_dim, dropout))

        modules_x.append(nn.Linear(h_dim, nconstants_x, bias=True))
        modules_c.append(nn.Linear(h_dim, nconstants_c, bias=True))

        self.hidden_layers_x = nn.Sequential(*modules_x)
        self.hidden_layers_c = nn.Sequential(*modules_c)

    def kappa_activation(self, x):
        """
        Translates values from [-inf, +inf] to [0, 1]
        """
        return sigmoid(4 * (x + 0.5))

    def beta_activation(self, x):
        """
        Translates values from [-inf, +inf] to [0.75, 1.25] as beta is weakly dependent on density
        """
        return (sigmoid(8 * x) + 1.5) / 2

    def get_exchange_constants(self, x):

        x_x_up = self.hidden_layers_x(x[:, [2, 5]])  # Slice out density descriptors
        x_x_down = self.hidden_layers_x(x[:, [4, 6]])

        return (
            x_x_up[:, 1].view(-1, 1),
            x_x_up[:, 0].view(-1, 1),
            x_x_down[:, 1].view(-1, 1),
            x_x_down[:, 0].view(-1, 1),
        )
    
    def get_correlation_constants(self, x):
        x_c = self.hidden_layers_c(x)
        beta = x_c[:, 0].view(-1, 1)
        gamma = x_c[:, 1].view(-1, 1)
        eta_raw = x_c[:, 2].view(-1, 1)  # Сырой выход для eta
        
        # Применяем асимптотические ограничения к eta
        # Определяем точки асимптотики
        device = x.device
        
        # 1. rho -> inf, s -> 0
        x0_rho_inf = torch.tensor([1e10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0)
        
        # 2. rho -> 1, s -> inf
        x0_s_inf = torch.tensor([1.0, 1.0, 1e10, 1e10, 1e10, 0.0, 0.0], device=device).unsqueeze(0)
        
        # 3. rho -> 1, s -> 0
        x0_s_zero = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0)
        
        x0_list = [x0_rho_inf, x0_s_inf, x0_s_zero]
        f0_list = [1.0, 1.0, 1.0]  # Во всех случаях eta -> 1
        
        # Вычисляем f(x0) для каждой точки асимптотики
        f_x0_list = []
        for x0 in x0_list:
            x0_batch = x0.repeat(x.shape[0], 1)
            f_x0 = self.hidden_layers_c(x0_batch)[:, 2].view(-1, 1)
            f_x0_list.append(f_x0)
        
        # Применяем оператор ˆθ для каждого ограничения
        theta_outputs = []
        for i in range(len(x0_list)):
            theta_out = asymptotic_constraint_operator(eta_raw, f_x0_list[i], f0_list[i])
            theta_outputs.append(theta_out)
        
        # Применяем интерполяцию Лагранжа
        eta_constrained = lagrange_interpolation_operator(theta_outputs, x, x0_list, l_func_tanh_norm)
        
        return beta, gamma, eta_constrained
    # def get_correlation_constants(self, x):

    #     x_c = self.hidden_layers_c(x)

    #     return x_c[:, 0].view(-1, 1), x_c[:, 1].view(-1, 1)
    def eta_activation(self, x): ### зачем этот диапазон нужен и активация для эты - хз, нейронка явно знает лучше
        """
        Активация для eta: переводит в диапазон [0.5, 3.0]
        """
        return 0.5 + 2.5 * torch.sigmoid(x)
    @staticmethod
    def all_sigma_zero(x):
        """
        Function for parameter mu and beta constraint
        """
        return torch.hstack([x[:, :2], torch.zeros_like(x[:, 2:]).to(x.device)])

    @staticmethod
    def all_sigma_zero_beta(x):
        """
        Function for parameter beta constraint
        """
        return torch.hstack(
            [x[:, :2], torch.zeros([x.shape[0], 3]).to(x.device), x[:, 5:]]
        )

    @staticmethod
    def all_sigma_inf(x):
        """
        Function for PW91 correlation parameters constraint
        """
        return torch.hstack([x[:, :2], torch.ones([x.shape[0], 5]).to(x.device)])

    @staticmethod
    def all_rho_inf(x):
        """
        Function for parameter gamma constraint
        """
        return torch.hstack([torch.ones([x.shape[0], 2]).to(x.device), x[:, 2:]])

    @staticmethod
    def shifted_elu(x):
        return elu(x) + 1

    def forward(self, x):

        mu_up, kappa_up, mu_down, kappa_down = self.get_exchange_constants(x)

        beta, gamma, eta = self.get_correlation_constants(x) ###3 параметра а не 2

        beta = self.beta_activation((beta - self.get_correlation_constants(self.all_sigma_zero_beta(x))[0]).view(-1,1))
        gamma = self.shifted_elu((gamma - self.get_correlation_constants(self.all_rho_inf(x))[1]).view(-1,1))
        ###eta = self.eta_activation(eta_raw)  # Активируем eta Я не уверен, что без view(-1,1) будет правильно, но посмотрим
        mu_up = self.shifted_elu((mu_up - self.get_exchange_constants(self.all_sigma_zero(x))[0])).view(-1,1)
        mu_down = self.shifted_elu((mu_down - self.get_exchange_constants(self.all_sigma_zero(x))[2])).view(-1,1)
        kappa_up = self.kappa_activation(kappa_up).view(-1,1)
        kappa_down = self.kappa_activation(kappa_down).view(-1,1)

        return torch.hstack(
            [
                beta,
                gamma,
                torch.ones([x.shape[0], 20]).to(x.device),
                kappa_up,
                mu_up,
                kappa_down,
                mu_down,
                eta,  # ДОБАВЛЯЕМ НОВЫЙ ПАРАМЕТР (индекс 26)
            ]
        ) * true_constants_PBE_extended.to(x.device) # Используем расширенную версию
