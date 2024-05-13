import torch.nn as nn
import numpy as np
import torch

def mlp(dim, hidden_dim, output_dim, layers, activation='relu', normalization=None):
    activation = {
        'relu': nn.ReLU
    }[activation]

    if isinstance(dim, list):
        dim = np.prod(dim)

    if normalization == "layernorm":
        seq = [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), activation()]
        for _ in range(layers):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), activation()]
        seq += [nn.Linear(hidden_dim, output_dim)]
    elif normalization == "batchnorm":
        seq = [nn.Linear(dim, hidden_dim), nn.BatchNorm1d(hidden_dim), activation()]
        for _ in range(layers):
            seq += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), activation()]
        seq += [nn.Linear(hidden_dim, output_dim)]
    else:
        seq = [nn.Linear(dim, hidden_dim), activation()]
        for _ in range(layers):
            seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
        seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)