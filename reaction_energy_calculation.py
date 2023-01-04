import numpy as np
import torch
from SVWN3 import f_svwn3


def get_local_energies(reaction, constants, device):
    calc_reaction_data = {}
    densities = reaction['Densities'].to(device)
    local_energies = f_svwn3(densities, constants)
    calc_reaction_data['Local_energies'] = local_energies
    # print(f"NaN {local_energies.isnan().sum()}")
    del local_energies, densities
    return calc_reaction_data


def add_calc_reaction_data(reaction, calc_reaction_data, device):
    calc_reaction_data['Weights'] = reaction['Weights'].to(device)
    calc_reaction_data['Densities'] = reaction['Densities'].to(device)
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


def integration(reaction, splitted_calc_reaction_data):
    molecule_energies = dict()
    for i, component in enumerate(np.frombuffer(reaction['Components'], dtype='<U20')):
        molecule_energies[component] = torch.sum(splitted_calc_reaction_data[component]['Local_energies'] \
                                              * (splitted_calc_reaction_data[component]['Densities'][:,0] \
                                              + splitted_calc_reaction_data[component]['Densities'][:,1]) \
                                              * (splitted_calc_reaction_data[component]['Weights'])) \
                                              + reaction['HF_energies'][i]
    del splitted_calc_reaction_data
    return molecule_energies


def get_energy_reaction(reaction, molecule_energies):
    hartree2kcal = 627.5095
    s = 0
    for coef, ener in zip(reaction['Coefficients'], molecule_energies.values()):
        s += coef*ener
    reaction_energy_kcal = s * hartree2kcal
    del ener, coef, s
    return reaction_energy_kcal


def calculate_reaction_energy(reaction, constants, device):
    local_energies = get_local_energies(reaction, constants, device)
    # print(local_energies)
    if local_energies['Local_energies'].isnan().any():
        print(local_energies['Local_energies'].isnan().sum())
        torch.save(local_energies['Local_energies'], 'local_energies.pt')
        raise Error()
    calc_reaction_data = add_calc_reaction_data(reaction, local_energies, device)
    splitted_calc_reaction_data = backsplit(reaction, calc_reaction_data)
    # print(splitted_calc_reaction_data)
    molecule_energies = integration(reaction, splitted_calc_reaction_data)
    # print(molecule_energies)
    reaction_energy_kcal = get_energy_reaction(reaction, molecule_energies)

    
    del molecule_energies, calc_reaction_data, splitted_calc_reaction_data, local_energies
    return reaction_energy_kcal


