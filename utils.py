import torch
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import inspect
import os
import random
from collections import defaultdict
from itertools import chain
from operator import methodcaller


def catch_nan(**kwargs):
    nan_detected = False
    inf_detected = False
    for k, v in kwargs.items():
        if v.isnan().any():
            print(f'{k} is NaN')
            nan_detected = True
        if v.isinf().any():
            print(f'{k} is inf')
            inf_detected = True            
        
    if nan_detected != False:
        for k, v in kwargs.items():
            torch.set_printoptions(precision=25)
            torch.save(v, f'log/{k}.pt')
        raise ValueError('NaN detected')
    if inf_detected != False:
        for k, v in kwargs.items():
            torch.set_printoptions(precision=25)
            torch.save(v, f'log/{k}.pt')
        raise ValueError('infinity detected')


def save_tensors(**kwargs):
    for k, v in kwargs.items():
        torch.set_printoptions(precision=25)
        torch.save(v, f'log/{k}.pt')


def log_params(model, metric1, metric2, name, predopt=False):
    with mlflow.start_run() as run:
        if predopt:
            metric1_name = "train_loss_mse"
            metric2_name = "train_loss_mae"
        else:
            metric1_name = "train_loss_mae"
            metric2_name = "test_loss_mae"
        mlflow.pytorch.log_model(model, name)
        mlflow.log_param("n_epochs", len(metric1))
        mlflow.log_metric(metric1_name, metric1[-1])
        mlflow.log_metric(metric2_name, metric2[-1])
        plt.plot(np.arange(1,len(metric1)+1), metric1, label=metric1_name)
        plt.plot(np.arange(1,len(metric1)+1), metric2, label=metric2_name)
        plt.legend()
        plt.xlabel("number of epochs")
        plt.ylabel("loss")
        plt.grid()
        plt.savefig(f"{name}.png")
        mlflow.log_artifact(f"{name}.png")
        os.remove(f"./{name}.png")
        plt.close()


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def set_random_seed(seed):
    # seed everything
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def stack_reactions(reactions):
    reaction_indices = [0]
    stop = 0
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
    reaction["reaction_indices"] = reaction_indices
    del dict_items
    return dict(reaction)