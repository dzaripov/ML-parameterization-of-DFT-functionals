import copy
import csv

import h5py
import numpy as np
import torch

from utils import stack_reactions


def ref(x, y, path):
    """
    returns reference energies for points of a reaction grid from Reference_data.csv
    """
    if path == None:
        pathfile = "Reference_data.csv"
    else:
        pathfile = f"{path}/Reference_data.csv"

    hartree2kcal = 627.5095
    with open(pathfile, newline="", encoding="cp1251") as csvfile:
        ref_file = csv.reader(csvfile, delimiter=",")
        k = 1
        if y == 391:
            k = hartree2kcal
        ref = []
        for n, i in enumerate(ref_file):
            if x <= n + 1 <= y:
                ref.append((i[0], float(i[2]) * k))

        return ref


def load_ref_energies(path):
    """Returns {db_name: [equation, energy]}"""
    ref_e = {  # Get the reference energies
        "MGAE109": ref(8, 116, path),
        "IP13": ref(155, 167, path),
        "EA13": ref(180, 192, path),
        "PA8": ref(195, 202, path),
        "DBH76": ref(251, 288, path) + ref(291, 328, path),
        "NCCE31": ref(331, 361, path),
        "ABDE4": ref(206, 209, path),
        # "AE17":ref(375, 391),
        "pTC13": ref(232, 234, path) + ref(237, 241, path) + ref(244, 248, path),
    }
    return ref_e


def load_component_names(path):
    """
    Returns {db_name: {id: {'Components': [...], 'Coefficients: [...]'
                                }
                            }
                        }
     which is a dictionary with Components and Coefficients data about all reactions
    """
    if path == None:
        pathfile = "total_dataframe_sorted_final.csv"
    else:
        pathfile = f"{path}/total_dataframe_sorted_final.csv"

    with open(pathfile, newline="", encoding="cp1251") as csvfile:
        ref_file = csv.reader(csvfile, delimiter=",")
        ref = dict()
        current_database = None

        for n, line in enumerate(ref_file):
            line = np.array(line)
            if n == 0:
                components = np.array(line)
            else:
                reaction_id = int(line[0])
                reaction_database = line[1]
                reaction_component_num = np.nonzero(list(map(float, line[2:])))[0] + 2
                if reaction_database in ref:
                    ref[reaction_database][reaction_id] = {
                        "Components": components[reaction_component_num],
                        "Coefficients": line[reaction_component_num],
                    }
                else:
                    ref[reaction_database] = {
                        reaction_id: {
                            "Components": components[reaction_component_num],
                            "Coefficients": line[reaction_component_num],
                        }
                    }
        return ref


def get_compounds_coefs_energy(reactions, energies):
    """Returns {id:
                    {'Components': [...], 'Coefficients: [...]', 'Energy: float', Database: str
                                }
                            }
    which is a dictionaty from load_component_names with Energy information added
    """
    data_final = dict()
    i = 0
    databases = energies.keys()
    for database in databases:
        data = reactions[database]
        for reaction in data:
            data_final[i] = {
                "Database": database,
                "Components": reactions[database][reaction][
                    "Components"
                ],  # .astype(object),
                "Coefficients": torch.Tensor(
                    reactions[database][reaction]["Coefficients"].astype(np.float32)
                ),
                "Energy": torch.Tensor(np.array(energies[database][reaction][1])),
            }
            i += 1

    return data_final


def get_h5_names(reaction):
    """reaction must be from the function get_compounds_coefs_energy"""
    database_match = {
        "MGAE109": "mgae109",
        "IP13": "ip13",
        "EA13": "ea13",
        "PA8": "pa8",
        "DBH76": "ntbh38",
        "NCCE31": "ncce31",
        "ABDE4": "abde4",
        "AE17": "ae17",
        "pTC13": "ptc13",
    }
    names = []
    for elem in reaction["Components"]:
        database = database_match[reaction["Database"]]
        names.append(f"{elem}.h5")
    return names


def add_reaction_info_from_h5(reaction, path):
    """
    reaction must be from get_compounds_coefs_energy
    returns merged descriptos array X, integration weights,
    a and b densities and indexes for backsplitting
    Values are filtered based on density vanishing
    (rho[0] !~ 0 & rho[1] !~ 0)

    Adds the following information to the reaction dict using h5 files from the dataset:
    Grid : np.array with grid descriptors
    Weights : list with integration weights of grid points
    Densities : np.array with alpha and beta densities data for grid points
    HF_energies : list of Total HF energy (T+V) which needs to be added to E_xc
    backsplit_ind: list of indexes where we concatenate molecules' grids
    """
    eps = 1e-27
    X = np.array([])
    backsplit_ind = []
    HF_energies = np.array([])
    for component_filename in get_h5_names(reaction):
        with h5py.File(f"{path}/{component_filename}", "r") as f:
            HF_energies = np.append(HF_energies, f["ener"][:][0])
            X_raw = np.array(f["grid"][:])
            if len(X) == 0:
                X = X_raw[:, 3:-1]
            else:
                X = np.vstack((X, X_raw[:, 3:-1]))
            X = X[
                np.logical_or((X[:, 1] > eps), (X[:, 2] > eps))
            ]  # energy of both alpha and beta density equal zero will be zero
            backsplit_ind.append(len(X))  # add indexes of molecules start/end
    weights = X[:, 0]  # get the integral weights
    densities = X[:, 1:3]  # get the densities
    sigmas = X[:, 3:6]  # get the contracted gradients

    X = X[:, 1:]  # get the grid descriptors

    # sigma_a_b to norm_grad=sigma_a + sigma_b + 2*sigma_a_b to get positive descriptor for log-transformation
    X = np.copy(X)
    X[:, 3] = X[:, 2] + X[:, 4] + 2 * X[:, 3]

    # log grid data
    eps = 10e-8
    X = np.log(X + eps)

    backsplit_ind = np.array(backsplit_ind)

    labels = [
        "Grid",
        "Weights",
        "Densities",
        "Gradients",
        "HF_energies",
        "backsplit_ind",
    ]
    values = [X, weights, densities, sigmas, HF_energies, backsplit_ind]
    for label, value in zip(labels, values):
        reaction[label] = torch.Tensor(value)

    return reaction


def make_reactions_dict(path=None):
    """
    Path : absolute or relative path to the molecules grid / reaction data
    Returns a dict like {reaction_id: {*reaction info}} with all info available listed below:
    ['Database', 'Components', 'Coefficients', 'Energy', 'Grid', 'Weights', 'Densities', 'HF_energies', 'backsplit_ind']
    """
    data = get_compounds_coefs_energy(
        load_component_names(path), load_ref_energies(path)
    )
    for i in data.keys():
        data[i] = add_reaction_info_from_h5(data[i], path)

    return data


def collate_fn(data):
    """
    Custom collate function for torch train and test dataloader
    """
    data = copy.deepcopy(data)
    reactions = []
    energies = []
    for reaction, energy in data:
        energies.append(energy)
        reaction.pop("Energy", None)
        reactions.append(reaction)
        torch_tensor_energy = torch.tensor(energies)
        reactions_stacked = stack_reactions(reactions)
    del energies, reactions, data
    return reactions_stacked, torch_tensor_energy


def collate_fn_predopt(data):
    """
    Custom collate function for torch predopt dataloader
    """
    data = copy.deepcopy(data)
    reactions = []
    for reaction, constant in data:
        reactions.append(reaction)
        reactions_stacked = stack_reactions(reactions)
    del reactions, data
    return reactions_stacked, constant
