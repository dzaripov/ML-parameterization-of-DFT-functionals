import torch
import mlflow
import os

from .PBE import F_PBE
from .SVWN3 import F_XALPHA
from .NN_models import NN_XALPHA_model, NN_PBE_model, true_constants_PBE


dir_path = os.path.dirname(os.path.realpath(__file__))
relative_path_to_model_state_dict = {
    'NN_PBE': '0/a00f863b96054f5299789b556e2b4ade/artifacts/NN_PBE',
    'NN_XALPHA': '0/a00f863b96054f5299789b556e2b4ade/artifacts/NN_XALPHA'
}
nn_model = {
    'NN_PBE': NN_PBE_model,
    'NN_XALPHA': NN_XALPHA_model,
}
MEAN_TRAIN_FEATURES = torch.tensor([-6.33887041, -6.68211794, -8.682011,   -7.72950894, -8.96326543, -6.54985721, -6.87318032])
STD_TRAIN_FEATURES = torch.tensor([5.00045545, 5.19493982, 6.30814192, 6.72813273, 6.34910604, 5.24995811, 5.41875425])

# Only for tests
def torch_grad(outputs, inputs):
    grads = torch.autograd.grad(outputs, inputs,
                                 create_graph=True,
                                 only_inputs=True,
                                 )
    return grads

class NN_FUNCTIONAL:

    def __init__(self, name):
        path_to_mlruns = '/'.join(dir_path.split('/')[:-3])+'/mlruns/'
        path_to_model_state_dict = path_to_mlruns + relative_path_to_model_state_dict[name]
        model = nn_model[name]()
        state_dict = mlflow.pytorch.load_state_dict(path_to_model_state_dict)
        model.load_state_dict(state_dict)
        model.eval()
        self.name = name
        self.model = model

    def create_features_from_rhos(self, features, device):
        rho_only_a, grad_a_x, grad_a_y, grad_a_z, _, tau_a = torch.unsqueeze(
            features['rho_a'], dim=1)
        rho_only_b, grad_b_x, grad_b_y, grad_b_z, _, tau_b = torch.unsqueeze(
            features['rho_b'], dim=1)
        
        eps = 1e-27
        rho_only_a = torch.where(rho_only_a>eps, rho_only_a, 0.)
        rho_only_b = torch.where(rho_only_b>eps, rho_only_b, 0.)

        norm_grad_a = (grad_a_x**2 + grad_a_y**2 + grad_a_z**2)
        norm_grad_b = (grad_b_x**2 + grad_b_y**2 + grad_b_z**2)

        grad_x = grad_a_x + grad_b_x
        grad_y = grad_a_y + grad_b_y
        grad_z = grad_a_z + grad_b_z
        norm_grad = (grad_x**2 + grad_y**2 + grad_z**2)

        keys = ['rho_a', 'rho_b', 'norm_grad_a', 'norm_grad', 'norm_grad_b', 'tau_a', 'tau_b']
        values = [rho_only_a, rho_only_b, norm_grad_a, norm_grad, norm_grad_b, tau_a, tau_b]
        
        feature_dict = dict()
        for key, value in zip(keys, values):
            feature_dict[key] = value.to(device)
            feature_dict[key].requires_grad = True

        feature_dict['norm_grad_ab'] = (feature_dict['norm_grad']-feature_dict['norm_grad_a']-feature_dict['norm_grad_b'])/2

        return feature_dict




    def __call__(self, features, device):

        torch.autograd.set_detect_anomaly(True)

        # Transfer model to device
        self.model = self.model.to(device)


        # Get features for nn and functional
        feature_dict = self.create_features_from_rhos(features, device)

        keys = ['rho_a', 'rho_b', 'norm_grad_a', 'norm_grad', 'norm_grad_b', 'tau_a', 'tau_b', 'norm_grad_ab']

        # Concatenate features to get input for NN
        nn_inputs = torch.cat([feature_dict[key] for key in keys[:7]], dim=0).T

        # Logarithmize and add small constant
        eps = 10e-8
        nn_features = torch.log(nn_inputs+eps)

        # Scale, substract mean and divide by std of train set
        nn_features_scaled = (nn_features-MEAN_TRAIN_FEATURES.to(device))/STD_TRAIN_FEATURES.to(device)

        # Get the NN output
        constants = self.model(nn_features_scaled)

        # Get densities for functional input
        functional_densities = torch.cat([feature_dict[key] for key in keys[:2]], dim=0).T 

        # Get gradients for functional input
        functional_gradients = torch.cat([feature_dict[key] for key in [keys[2], keys[7], keys[4]]], dim=0).T

        if self.name == 'NN_PBE':
            vxc = F_PBE(functional_densities, functional_gradients, constants, device)
        elif self.name == 'NN_XALPHA':
            vxc = F_XALPHA(functional_densities, constants)
        else:
            raise NameError(f'Invalid functional name: {self.name}')

        local_xc = vxc*(feature_dict['rho_a']+feature_dict['rho_b'])


        return local_xc, vxc, feature_dict

