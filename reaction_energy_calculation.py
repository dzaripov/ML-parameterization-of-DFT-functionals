import numpy as np
import torch
from SVWN3 import f_svwn3
from PBE import F_PBE, pw_test
from importlib import reload
from PBE_new import F_PBE_new
import PBE_new
reload(PBE_new)
import PBE
reload(PBE)
import pickle


def get_local_energies(reaction, constants, device, rung='GGA', dft='PBE'):
    calc_reaction_data = {}
    densities = reaction['Densities'].to(device)
    # eps = 1e-29
    # condition_rho_not_0 = torch.logical_not((densities[:, 0] < eps) & (densities[:, 0] < eps)) # energy of both alpha and beta density equal zero will be zero
    # densities = densities[condition_rho_not_0]
    # constants = constants[condition_rho_not_0]
    
    if rung == 'LDA':
        if dft == 'SVWN3':
            local_energies = f_svwn3(densities, constants)
        if dft == 'pw':
            local_energies = pw_test(densities, constants)
    elif rung == 'GGA':
        gradients = (reaction['Gradients']).to(device)
        if dft == 'PBE':
            local_energies = F_PBE(densities, gradients, constants)
        elif dft == 'PBE_new':
            local_energies = F_PBE_new(densities, gradients, constants)
    
    calc_reaction_data['Local_energies'] = local_energies
    calc_reaction_data['Densities'] = densities
    calc_reaction_data['Weights'] = reaction['Weights'].to(device)
    del local_energies, densities
    # with open('checkpoints/calc_reaction_data.pickle', 'wb') as f:
    #     pickle.dump(calc_reaction_data, f)
    # with open('checkpoints/reaction.pickle', 'wb') as f:
    #     pickle.dump(reaction, f)
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
    
    
#     import pickle
#     with open(f'log/reaction_H_H2.pickle', 'wb') as f:
#         pickle.dump(reaction, f)
#     with open(f'log/splitted_calc_reaction_data_H_H2.pickle', 'wb') as f:
#         pickle.dump(splitted_calc_reaction_data, f)
#     print('Saved reaction and splitted calc reaction data for H and H2')
    
    
    
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
        # print(ener)
        s += coef * ener
    reaction_energy_kcal = s * hartree2kcal
    del ener, coef, s
    return reaction_energy_kcal


def calculate_reaction_energy(reaction, constants, device, rung, dft):
    local_energies = get_local_energies(reaction, constants, device, rung, dft)
    # print(local_energies)
    if local_energies['Local_energies'].isnan().any():
        print(local_energies['Local_energies'].isnan().sum())
        torch.save(local_energies['Local_energies'], 'local_energies.pt')
        raise Error()
    splitted_calc_reaction_data = backsplit(reaction, local_energies)
    # print(splitted_calc_reaction_data)
    molecule_energies = integration(reaction, splitted_calc_reaction_data)
    # print(molecule_energies)
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