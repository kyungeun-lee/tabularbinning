import torch.nn as nn
from libs.models import mlp
import torch, torchvision

class BilinearCritic(nn.Module):
    def __init__(self, dim, normalization=None, **extra_kwargs):
        super(BilinearCritic, self).__init__()
        if normalization == "layernorm":
            seq = [nn.LayerNorm(dim), nn.Linear(dim, dim)]
            self.model = nn.Sequential(*seq)
        elif normalization == "batchnorm":
            seq = [nn.BatchNorm1d(dim), nn.Linear(dim, dim)]
            self.model = nn.Sequential(*seq)
        else:
            if type(dim) == list:
                dim = dim[0]
            self.model = nn.Linear(dim, dim)
    def forward(self, x, y):
        return torch.matmul(x, self.model(y).t())

class SeparableCritic(nn.Module):
    """Separable critic. where the output value is g(x) h(y). """

    def __init__(self, dim, hidden_dim, embed_dim, layers, activation="relu", normalization="None", **extra_kwargs):
        super(SeparableCritic, self).__init__()
        self._g = mlp(dim, hidden_dim, embed_dim, layers, activation, normalization)
        self._h = mlp(dim, hidden_dim, embed_dim, layers, activation, normalization)

    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores

class JointCritic(nn.Module):
    """Concat critic, where we concat the inputs and use one MLP to output the value."""

    def __init__(self, dim, hidden_dim, layers, activation="relu", normalization="None", **extra_kwargs):
        super(JointCritic, self).__init__()
        # output is scalar score
        self._f = mlp(dim * 2, hidden_dim, 1, layers, activation, normalization)

    def forward(self, x, y):
        batch_size = x.size(0)
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()

class InnerCritic(torch.nn.Module):
    def __init__(self):
        super(InnerCritic, self).__init__()
        self.fc = torch.nn.Identity()
    def forward(self, x, y):
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        return torch.matmul(x, y.t())


def set_critic(critic_type, dim, hidden_dim=100, embed_dim=100, layers=1,
               activation="relu",
               normalization="None", dnn_type="mlp", device="cuda"):

    if critic_type == "joint":
        critic = JointCritic(dim, hidden_dim=hidden_dim, layers=layers,
                             critic_type=dnn_type, activation=activation, normalization=normalization)
    elif critic_type == "separable":
        critic = SeparableCritic(dim, hidden_dim=hidden_dim, layers=layers, embed_dim=embed_dim,
                                 critic_type=dnn_type, activation=activation, normalization=normalization)
    elif critic_type == "bilinear":
        critic = BilinearCritic(dim, normalization=normalization)
    elif critic_type == "inner":
        critic = InnerCritic()
    else:
        raise ValueError
    return critic.to(device)

def log_prob_gaussian(x):
    return torch.sum(torch.distributions.Normal(0., 1.).log_prob(x), -1)