import inspect
import os
import random
from collections import defaultdict
from itertools import chain
from operator import methodcaller

import matplotlib.pyplot as plt
import numpy as np
import torch


def catch_nan(**kwargs):
    nan_detected = False
    inf_detected = False
    for k, v in kwargs.items():
        if v.isnan().any():
            print(f"{k} is NaN")
            nan_detected = True
        if v.isinf().any():
            print(f"{k} is inf")
            inf_detected = True

    if nan_detected != False:
        for k, v in kwargs.items():
            torch.set_printoptions(precision=25)
            torch.save(v, f"log/{k}.pt")
        raise ValueError("NaN detected")
    if inf_detected != False:
        for k, v in kwargs.items():
            torch.set_printoptions(precision=25)
            torch.save(v, f"log/{k}.pt")
        raise ValueError("infinity detected")


def save_tensors(**kwargs):
    for k, v in kwargs.items():
        torch.set_printoptions(precision=25)
        torch.save(v, f"log/{k}.pt")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]


def set_random_seed(seed):
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
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
    dict_items = map(methodcaller("items"), reactions)
    for k, v in chain.from_iterable(dict_items):
        if k == "Components":
            reaction_indices.append(stop + len(v))
            stop += len(v)
        if k in ("Components", "Coefficients", "Database"):
            reaction[k] = (
                np.hstack([np.array(reaction[k]), v]) if len(reaction[k]) else v
            )
        elif k in ("Grid", "Densities", "Gradients"):
            reaction[k] = torch.vstack([reaction[k], v]) if len(reaction[k]) else v
        elif k == "backsplit_ind":
            reaction[k] = (
                torch.hstack([reaction[k], v + reaction[k][-1]])
                if len(reaction[k])
                else v
            )
        else:
            if type(reaction[k]) != torch.Tensor:
                reaction[k] = torch.Tensor(reaction[k])
            reaction[k] = torch.hstack([reaction[k], v]) if reaction[k].dim != 0 else v
    reaction["reaction_indices"] = reaction_indices
    del dict_items
    return dict(reaction)


def configure_optimizers(model, learning_rate):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (
        torch.nn.LayerNorm,
        torch.nn.PReLU,
        torch.nn.BatchNorm1d,
    )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert (
        len(param_dict.keys() - union_params) == 0
    ), "parameters %s were not separated into either decay/no_decay set!" % (
        str(param_dict.keys() - union_params),
    )
    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": 0.01,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]
    #    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)

    optimizer = torch.optim.RAdam(
        optim_groups, lr=learning_rate, decoupled_weight_decay=True
    )
    return optimizer
