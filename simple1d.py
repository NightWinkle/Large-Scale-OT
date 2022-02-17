from typing import Union
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
from torch.nn import Parameter
from torch.optim import Adam, SGD

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "10"

def l2_distance(x: torch.Tensor, y: torch.Tensor) \
        -> torch.Tensor:
    """Compute the Gram matrix holding all ||.||_2 distances."""
    xTy = 2 * x.matmul(y.transpose(0, 1))
    x2 = torch.sum(x ** 2, dim=1)[:, None]
    y2 = torch.sum(y ** 2, dim=1)[None, :]
    K = x2 + y2 - xTy
    return K

class OTPlan(nn.Module):
    def __init__(self, *,
                 source_type: str = 'discrete',
                 target_type: str = 'discrete',
                 source_dim: Union[int, None] = None,
                 target_dim: Union[int, None] = None,
                 source_length: Union[int, None] = None,
                 target_length: Union[int, None] = None,
                 alpha: float = 0.1,
                 regularization: str = 'entropy'
                 ):
        super().__init__()
        self.source_type = source_type

        if source_type == 'discrete':
            assert isinstance(source_length, int)
            self.u = DiscretePotential(source_length)
        elif source_type == 'continuous':
            assert isinstance(source_dim, int)
            self.u = ContinuousPotential(source_dim)
        self.target_type = target_type
        if target_type == 'discrete':
            assert isinstance(target_length, int)
            self.v = DiscretePotential(target_length)
        elif target_type == 'continuous':
            assert isinstance(target_dim, int)
            self.v = ContinuousPotential(target_dim)
        self.alpha = alpha

        assert regularization in ['entropy', 'l2'], ValueError
        self.regularization = regularization
        self.reset_parameters()

    def reset_parameters(self):
        self.u.reset_parameters()
        self.v.reset_parameters()

    def _get_uv(self, x, y, xidx=None, yidx=None):
        if self.source_type == 'discrete':
            u = self.u(xidx)
        else:
            u = self.u(x)
        if self.target_type == 'discrete':
            v = self.v(yidx)
        else:
            v = self.v(y)
        return u, v

    def loss(self, x, y, xidx=None, yidx=None):
        K = torch.sqrt(l2_distance(x, y))
        u, v = self._get_uv(x, y, xidx, yidx)

        if regularization == 'entropy':
            reg = - alpha * torch.exp((u[:, None] + v[None, :] - K) / alpha)
        else:
            reg = - torch.clamp((u[:, None] + v[None, :] - K),
                                min=0) ** 2 / 4 / alpha
        return - torch.mean(u[:, None] + v[None, :] + reg)

    def forward(self, x, y, xidx=None, yidx=None):
        K = torch.sqrt(l2_distance(x, y))
        u, v = self._get_uv(x, y, xidx, yidx)
        if self.regularization == 'entropy':
            return torch.exp((u[:, None] + v[None, :] - K) / self.alpha)
        else:
            return torch.clamp((u[:, None] + v[None, :] - K),
                               min=0) / (2 * self.alpha)

class DiscretePotential(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.u = Parameter(torch.empty(length))
        self.reset_parameters()

    def reset_parameters(self):
        self.u.data.zero_()

    def forward(self, idx):
        return self.u[idx]


class ContinuousPotential(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Sequential(nn.Linear(dim, 128),
                               nn.ReLU(),
                               nn.Linear(128, 256),
                               nn.ReLU(),
                               nn.Linear(256, 128),
                               nn.ReLU(),
                               nn.Linear(128, 1)
                               )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.u._modules.values():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, x):
        return self.u(x)[:, 0]


n_plan_iter = 10000
n_map_iter = 10000
alpha = .0025
batch_size = 100
regularization = 'l2'
n_target_samples = 1000
lr = 1e-3


def run(setting='discrete_discrete'):
    if setting == 'discrete_discrete':
        x = Normal(torch.eye(1) * mu_source, torch.eye(1) * sigma_source)

        y = Normal(torch.eye(1) * mu_target, torch.eye(1) * sigma_target)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        wy = torch.from_numpy(wy).float()
        wx = torch.from_numpy(wx).float()

        x = MultivariateNormal(torch.zeros(2), torch.eye(2) / 4)
        x = x.sample((n_target_samples, ))
        wx = np.full(len(x), 1 / len(x))
        wx = torch.from_numpy(wx).float()

        ot_plan = OTPlan(source_type='discrete', target_type='discrete',
                         target_length=len(y), source_length=len(x))
    elif setting == 'continuous_discrete':
        x = MultivariateNormal(torch.zeros(2), torch.eye(2) / 4)
        y, wy = make_circle(radius=4, n_samples=n_target_samples)

        y = torch.from_numpy(y).float()
        wy = torch.from_numpy(wy).float()

        ot_plan = OTPlan(source_type='continuous', target_type='discrete',
                         target_length=len(y), source_dim=2)
    else:
        raise ValueError

    mapping = Mapping(ot_plan, dim=2)
    optimizer = Adam(ot_plan.parameters(), amsgrad=True, lr=lr)
    # optimizer = SGD(ot_plan.parameters(), lr=lr)


    plan_objectives = []
    map_objectives = []

    print('Learning OT plan')

    for i in range(n_plan_iter):
        optimizer.zero_grad()

        if setting == 'discrete_discrete':
            this_yidx = torch.multinomial(wy, batch_size)
            this_y = y[this_yidx]
            this_xidx = torch.multinomial(wx, batch_size)
            this_x = x[this_xidx]
        else:
            this_x = x.sample((batch_size,))
            this_yidx = torch.multinomial(wy, batch_size)
            this_y = y[this_yidx]
            this_xidx = None
        loss = ot_plan.loss(this_x, this_y, yidx=this_yidx, xidx=this_xidx)
        loss.backward()
        optimizer.step()
        plan_objectives.append(-loss.item())
        if i % 100 == 0:
            print(f'Iter {i}, loss {-loss.item():.3f}')

    optimizer = Adam(mapping.parameters(), amsgrad=True, lr=lr)
    # optimizer = SGD(mapping.parameters(), lr=1e-5)


    print('Learning barycentric mapping')
    for i in range(n_map_iter):
        optimizer.zero_grad()

        if setting == 'discrete_discrete':
            this_yidx = torch.multinomial(wy, batch_size)
            this_y = y[this_yidx]
            this_xidx = torch.multinomial(wx, batch_size)
            this_x = x[this_xidx]
        else:
            this_x = x.sample((batch_size,))
            this_yidx = torch.multinomial(wy, batch_size)
            this_y = y[this_yidx]
            this_xidx = None

        loss = mapping.loss(this_x, this_y, yidx=this_yidx, xidx=this_xidx)
        loss.backward()
        optimizer.step()
        map_objectives.append(loss.item())
        if i % 100 == 0:
            print(f'Iter {i}, loss {loss.item():.3f}')

    if setting == 'continuous_discrete':
        x = x.sample((len(y),))
    with torch.no_grad():
        mapped = mapping(x)
    x = x.numpy()
    y = y.numpy()
    mapped = mapped.numpy()

    return x, y, mapped, plan_objectives, map_objectives

@hydra.main(config_path="simple1dconf", config_name="config")
def main(cfg : DictConfig) -> None:
    wandb.init(project="largescaleot_seguy", entity="nightwinkle", config=OmegaConf.to_container(cfg))

if __name__=="__main__":
    main()