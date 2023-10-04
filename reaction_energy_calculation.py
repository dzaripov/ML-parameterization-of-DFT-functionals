import numpy as np
import torch
from SVWN3 import f_svwn3, F_XALPHA
from PBE import F_PBE, F_PBE_X, F_PBE_C
from collections import defaultdict
from itertools import chain
from operator import methodcaller
# from importlib import reload
# import PBE
# reload(PBE)
import pickle



def stack_reactions(reactions):
    reaction_indices = [0]
    stop = 0
    grid_size = 0
    reaction = defaultdict(list)
    dict_items = map(methodcaller('items'), reactions)
    for k, v in chain.from_iterable(dict_items):
        if k == "Components":
            reaction_indices.append(stop+len(v))
            stop += len(v)
        if k in ("Components", "Coefficients", "Database"):
            reaction[k] = np.hstack([np.array(reaction[k]), v]) if len(
                reaction[k]) else v
        elif k in ("Grid", "Densities", "Gradients"):
            reaction[k] = torch.vstack([reaction[k], v]) if len(
                reaction[k]) else v
        elif k == 'backsplit_ind':
            reaction[k] = torch.hstack([reaction[k], v +
                                        reaction[k][-1]]) if len(reaction[k]) else v
        else:
            if type(reaction[k])!=torch.Tensor:
                reaction[k]=torch.Tensor(reaction[k])
            reaction[k] = torch.hstack([reaction[k], v]) if reaction[k].dim!=0 else v
        grid_size = len(reaction["Grid"])
    reaction["reaction_indices"] = reaction_indices
    del dict_items
    return dict(reaction)


def get_local_energies(reaction, constants, device, rung='GGA', dft='PBE'):
    calc_reaction_data = {}
    densities = reaction['Densities'].to(device)
    if rung == 'LDA':
        if dft == 'SVWN3':
            local_energies = f_svwn3(densities, constants)
        if dft == 'pw':
            local_energies = pw_test(densities, constants)
        if dft == 'XALPHA':
            local_energies = F_XALPHA(densities, constants)
    elif rung == 'GGA':
        gradients = (reaction['Gradients']).to(device)
        if dft == 'PBE':
            local_energies = F_PBE(densities, gradients, constants, device)
        elif dft == 'PBE_new':
            local_energies = F_PBE_new(densities, gradients, constants, device)
    
    calc_reaction_data['Local_energies'] = local_energies
    calc_reaction_data['Densities'] = densities
    calc_reaction_data['Weights'] = reaction['Weights'].to(device)
    del local_energies, densities
    return calc_reaction_data


def get_local_energies_x(reaction, constants, device, rung='GGA', dft='PBE'):
    calc_reaction_data = {}
    densities = reaction['Densities'].to(device)
    gradients = (reaction['Gradients']).to(device)
    
    local_energies = F_PBE_X(densities, gradients, constants, device)
    
    calc_reaction_data['Local_energies'] = local_energies
    calc_reaction_data['Densities'] = densities
    calc_reaction_data['Weights'] = reaction['Weights'].to(device)
    del local_energies, densities
    return calc_reaction_data


def get_local_energies_c(reaction, constants, device, rung='GGA', dft='PBE'):
    calc_reaction_data = {}
    densities = reaction['Densities'].to(device)
    gradients = (reaction['Gradients']).to(device)
    
    local_energies = F_PBE_C(densities, gradients, constants, device)
    
    calc_reaction_data['Local_energies'] = local_energies
    calc_reaction_data['Densities'] = densities
    calc_reaction_data['Weights'] = reaction['Weights'].to(device)
    del local_energies, densities
    return calc_reaction_data


def backsplit(reaction, calc_reaction_data):
    backsplit_ind = reaction['backsplit_ind'].type(torch.int)
    splitted_data = dict()
    stop = 0
    
    for i, component in enumerate(np.frombuffer(reaction['Components'], dtype='<U20')):
        splitted_data[component] = dict()
        start = stop
        stop = backsplit_ind[i]
        for elem in ('Local_energies', 'Weights', 'Densities'):
            splitted_data[component][elem] = calc_reaction_data[elem][start:stop]
    del backsplit_ind, start, stop
    return splitted_data


def integration(reaction, splitted_calc_reaction_data, dispersions=dict()):
    
    molecule_energies = dict()
    for i, component in enumerate(np.frombuffer(reaction['Components'], dtype='<U20')):
        molecule_energies[component+str(i)] = torch.sum(splitted_calc_reaction_data[component]['Local_energies'] \
                                              * (splitted_calc_reaction_data[component]['Densities'][:,0] \
                                              + splitted_calc_reaction_data[component]['Densities'][:,1]) \
                                              * (splitted_calc_reaction_data[component]['Weights'])) \
                                              + reaction['HF_energies'][i] \
                                              + torch.Tensor(dispersions.get(component,0))
    del splitted_calc_reaction_data
    return molecule_energies


def get_energy_reaction(reaction, molecule_energies):
    slices = reaction.get("reaction_indices", [0, len(reaction["Components"])])
    hartree2kcal = 627.5095
    reaction_energy_kcal = []
    for i in range(len(slices)-1):
        s = 0
        for coef, ener in list(zip(reaction['Coefficients'], molecule_energies.values()))[slices[i]:slices[i+1]]:
            s += coef * ener
        reaction_energy_kcal.append(s * hartree2kcal)
    del ener, coef, s, slices
#    if type(reaction_energy_kcal)==list:
#        reaction_energy_kcal=torch.tensor(reaction_energy_kcal)
#    reaction_energy_kcal.requires_grad = True
    return torch.stack(reaction_energy_kcal) #tensor of size len(reactions)


def calculate_reaction_energy(reaction, constants, device, rung, dft, dispersions=dict()):
    local_energies = get_local_energies(reaction, constants, device, rung, dft)
    if local_energies['Local_energies'].isnan().any():
        print(local_energies['Local_energies'].isnan().sum())
        torch.save(local_energies['Local_energies'], 'local_energies.pt')
        raise Error()
    splitted_calc_reaction_data = backsplit(reaction, local_energies)
    molecule_energies = integration(reaction, splitted_calc_reaction_data, dispersions)
    reaction_energy_kcal = get_energy_reaction(reaction, molecule_energies)
    del molecule_energies, splitted_calc_reaction_data, local_energies
    return reaction_energy_kcal


def test_energy_PBE(test_grid, constants):
    
    local_energies = F_PBE(test_grid['Densities'], test_grid['Gradients'], constants)
    local_scaled_energies =   local_energies \
                            * (test_grid['Densities'][:,0] \
                            + test_grid['Densities'][:,1]) \
                            * (test_grid['Weights'])
    return local_energies, local_scaled_energies


