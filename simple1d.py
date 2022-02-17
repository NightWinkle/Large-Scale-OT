from typing import Union
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "10"

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
from torch.nn import Parameter
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb


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

        if self.regularization == 'entropy':
            reg = - self.alpha * torch.exp((u[:, None] + v[None, :] - K) / self.alpha)
        else:
            reg = - torch.clamp((u[:, None] + v[None, :] - K),
                                min=0) ** 2 / 4 / self.alpha
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

def build_optimizer(params, config):
    if config['optimizer'] == "adam":
        opt = Adam(params, config['lr'], amsgrad=True)
    elif config['optimizer'] == "sgd":
        opt = SGD(params, lr=config['lr'])
    else:
        opt = Adam(params, config['lr'], amsgrad=True)
    return opt

@hydra.main(config_path="simple1dconf", config_name="config")
def app(cfg : DictConfig) -> None:
    wandb.init(project="largescaleot_seguy", entity="nightwinkle", config=OmegaConf.to_container(cfg))

    config = wandb.config

    x = Normal(torch.eye(1) * config['source']['mean'], torch.eye(1) * config['source']['std'])
    source_length = None
    if config['source']['setting'] == 'discrete':
        x = x.sample((config['source']['n_samples'], 1))
        source_length = config['source']['n_samples']
        wx = np.full((config['source']['n_samples'],), 1/config['source']['n_samples'])

    y = Normal(torch.eye(1) * config['target']['mean'], torch.eye(1) * config['target']['std'])
    target_length = config['target']['n_samples']
    if config['target']['setting'] == 'discrete':
        x = y.sample((config['target']['n_samples'], 1))
        target_length = config['target']['n_samples']
        wy = np.full((config['target']['n_samples'],), 1/config['target']['n_samples'])

    ot_plan = OTPlan(source_type=config['source']['setting'], target_type=config['target']['setting'],
                     source_length=source_length, target_length=target_length, 
                     source_dim=1, target_dim=1, regularization=config['regularization'])

    optimizer = build_optimizer(ot_plan.parameters(), config)

    for step in range(config.steps):
        optimizer.zero_grad()

        if config['source']['setting'] == 'discrete':
            this_xidx = torch.multinomial(wx, config.batch_size)
            this_x = x[this_xidx]
        else:
            this_xidx = None
            this_x = x.sample((config.batch_size, 1))
        if config['target']['setting'] == 'discrete':
            this_yidx = torch.multinomial(wy, config.batch_size)
            this_y = y[this_xidx]
        else:
            this_yidx = None
            this_y = y.sample((config.batch_size, 1))

        loss = ot_plan.loss(this_x, this_y, yidx=this_yidx, xidx=this_xidx)
        loss.backward()
        optimizer.step()

        wandb.log({"objective": -loss.item()})
        if step % config['log_steps'] == 0:
            if config['source']['setting'] == 'discrete':
                u_fig = plt.figure()
                u_fig.plot(x, ot_plan.u)
            else:
                u_support = torch.linspace(config['source']['support'][0], config['source']['support'][1], config['n_support'])
                u_val = ot_plan.u(u_support)
                u_fig = plt.figure()
                u_fig.plot(u_support, u_val)

            if config['target']['setting'] == 'discrete':
                v_fig = plt.figure()
                v_fig.plot(y, ot_plan.v)
            else:
                v_support = torch.linspace(config['target']['support'][0], config['target']['support'][1], config['n_support'])
                v_val = ot_plan.v(v_support)
                v_fig = plt.figure()
                v_fig.plot(u_support, v_val)
            wandb.log({"u_potential": u_fig, "v_potential": v_fig})

if __name__=="__main__":
    app()