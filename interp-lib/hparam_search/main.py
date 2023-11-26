import torch
import numpy as np
from copy import deepcopy

def grid_from_config(config):
    for k, v in config.items():
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            config[k] = v.tolist()

    res= [{}]
    for param, param_values in config.items():
        new_res = []
        for d in res:
            for v in param_values:
                copied_d = deepcopy(d)
                copied_d[param] = v
                new_res.append(copied_d)
        res = new_res
    return res

