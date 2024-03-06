import copy
import pickle
import random

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from dataset import make_reactions_dict


def rename_keys(data):
    # Turns reaction_data dict keys names into numbers.
    l = len(data)
    keys = data.keys()
    data_new = {}
    for i, key in zip(range(l), keys):
        data_new[i] = data[key]
    return data_new


def train_split(data, test_size, shuffle=False, random_state=42):
    # Returns train and test reaction dictionaries.
    if shuffle:
        keys = list(data.keys())
        random.shuffle(keys, random_state=random_state)
        for i in keys:
            data[keys[i]] = data[i]

    train, test = dict(), dict()
    border = round(len(data.keys()) * (1 - test_size))
    for i in range(len(data.keys())):
        if i <= border:
            train[i] = data[i]
        else:
            test[i] = data[i]
    return rename_keys(train), rename_keys(test)


def prepare(path="data", test_size=0.2, random_state=42):
    # Make a single dictionary from the whole dataset.
    data = make_reactions_dict(path=path)

    # Train-test split.
    data_train, data_test = train_split(copy.deepcopy(data), test_size, shuffle=True, random_state=random_state)

    # Stdscaler fit.
    lst = []
    for i in range(len(data_train)):
        lst.append(data_train[i]["Grid"])

    train_grid_data = torch.cat(lst)
    stdscaler = StandardScaler()
    stdscaler.fit(np.array(train_grid_data))

    # Check mean and var for later SCF calculations
    print("mean:", stdscaler.mean_)
    print("std:", np.sqrt(stdscaler.var_))
    # Stdscaler transform.
    for data_t in (data_train, data_test):
        for i in range(len(data_t)):
            data_t[i]["Grid"] = torch.Tensor(stdscaler.transform(data_t[i]["Grid"]))

    return data, data_train, data_test


def save_chk(data, data_train, data_test, path="checkpoints"):
    # Save all processed data into pickle.
    with open(f"{path}/data.pickle", "wb") as f:
        pickle.dump(data, f)
    with open(f"{path}/data_train.pickle", "wb") as f:
        pickle.dump(data_train, f)
    with open(f"{path}/data_test.pickle", "wb") as f:
        pickle.dump(data_test, f)


def load_chk(path="checkpoints"):
    # Load processed data from pickle.
    with open(f"{path}/data.pickle", "rb") as f:
        data = pickle.load(f)
    with open(f"{path}/data_train.pickle", "rb") as f:
        data_train = pickle.load(f)
    with open(f"{path}/data_test.pickle", "rb") as f:
        data_test = pickle.load(f)
    return data, data_train, data_test


if __name__ == '__main__':
    data, data_train, data_test = prepare(path='data', test_size=0.2)
    save_chk(data, data_train, data_test, path='checkpoints')