import os

from typing import Union


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "10"

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb

from geomloss import SamplesLoss
from omegaconf import DictConfig, OmegaConf
from scipy import interpolate
from torch.distributions import MultivariateNormal, Normal
from torch.nn import Parameter
from torch.optim import SGD, Adam


def l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the Gram matrix holding all ||.||_2 distances."""
    xTy = 2 * x.matmul(y.transpose(0, 1))
    x2 = torch.sum(x**2, dim=1)[:, None]
    y2 = torch.sum(y**2, dim=1)[None, :]
    K = x2 + y2 - xTy
    return K


class OTPlan(nn.Module):
    def __init__(
        self,
        *,
        source_type: str = "discrete",
        target_type: str = "discrete",
        source_dim: Union[int, None] = None,
        target_dim: Union[int, None] = None,
        source_length: Union[int, None] = None,
        target_length: Union[int, None] = None,
        alpha: float = 0.1,
        regularization: str = "entropy"
    ):
        super().__init__()
        self.source_type = source_type

        if source_type == "discrete":
            assert isinstance(source_length, int)
            self.u = DiscretePotential(source_length)
        elif source_type == "continuous":
            assert isinstance(source_dim, int)
            self.u = ContinuousPotential(source_dim)
        self.target_type = target_type
        if target_type == "discrete":
            assert isinstance(target_length, int)
            self.v = DiscretePotential(target_length)
        elif target_type == "continuous":
            assert isinstance(target_dim, int)
            self.v = ContinuousPotential(target_dim)
        self.alpha = alpha

        assert regularization in ["entropy", "l2"], ValueError
        self.regularization = regularization
        self.reset_parameters()

    def reset_parameters(self):
        self.u.reset_parameters()
        self.v.reset_parameters()

    def _get_uv(self, x, y, xidx=None, yidx=None):
        if self.source_type == "discrete":
            u = self.u(xidx)
        else:
            u = self.u(x)
        if self.target_type == "discrete":
            v = self.v(yidx)
        else:
            v = self.v(y)
        return u, v

    def loss(self, x, y, xidx=None, yidx=None):
        K = torch.sqrt(l2_distance(x, y))
        u, v = self._get_uv(x, y, xidx, yidx)

        if self.regularization == "entropy":
            reg = -self.alpha * torch.exp(
                (u[:, None] + v[None, :] - K) / self.alpha
            )
        else:
            reg = (
                -torch.clamp((u[:, None] + v[None, :] - K), min=0) ** 2
                / 4
                / self.alpha
            )
        return -torch.mean(u[:, None] + v[None, :] + reg)

    def forward(self, x, y, xidx=None, yidx=None):
        K = torch.sqrt(l2_distance(x, y))
        u, v = self._get_uv(x, y, xidx, yidx)
        if self.regularization == "entropy":
            return torch.exp((u[:, None] + v[None, :] - K) / self.alpha)
        else:
            return torch.clamp((u[:, None] + v[None, :] - K), min=0) / (
                2 * self.alpha
            )


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
        self.u = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.u._modules.values():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, x):
        return self.u(x)[:, 0]


def build_optimizer(params, config):
    if config["optimizer"] == "adam":
        opt = Adam(params, config["lr"], amsgrad=True)
    elif config["optimizer"] == "sgd":
        opt = SGD(params, lr=config["lr"])
    else:
        opt = Adam(params, config["lr"], amsgrad=True)
    return opt


@hydra.main(config_path="simple1dconf", config_name="config")
def app(cfg: DictConfig) -> None:
    run = wandb.init(
        project="largescaleot_seguy",
        entity="nightwinkle",
        config=OmegaConf.to_container(cfg),
        reinit=True,
    )

    config = wandb.config

    x = Normal(config["source"]["mean"], config["source"]["std"])
    u_support = torch.linspace(
        config["source"]["support"][0],
        config["source"]["support"][1],
        config["n_support"],
    ).unsqueeze(-1)
    mu_log = x.log_prob(u_support)
    source_length = None
    if config["source"]["setting"] == "discrete":
        x = x.sample((config["source"]["n_samples"], 1))
        source_length = config["source"]["n_samples"]
        wx = torch.full(
            (config["source"]["n_samples"],), 1 / config["source"]["n_samples"]
        )

    y = Normal(config["target"]["mean"], config["target"]["std"])
    v_support = torch.linspace(
        config["target"]["support"][0],
        config["target"]["support"][1],
        config["n_support"],
    ).unsqueeze(-1)
    nu_log = y.log_prob(v_support)
    target_length = config["target"]["n_samples"]
    if config["target"]["setting"] == "discrete":
        y = y.sample((config["target"]["n_samples"], 1))
        target_length = config["target"]["n_samples"]
        wy = torch.full(
            (config["target"]["n_samples"],), 1 / config["target"]["n_samples"]
        )
    ot_plan = OTPlan(
        source_type=config["source"]["setting"],
        target_type=config["target"]["setting"],
        source_length=source_length,
        target_length=target_length,
        source_dim=1,
        target_dim=1,
        regularization=config["regularization"],
    ).cuda()
    sinkhorn = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=config["alpha"],
        debias=False,
        potentials=True,
    )
    u_exact, v_exact = sinkhorn(
        torch.exp(mu_log).squeeze().cuda(),
        u_support.cuda(),
        torch.exp(nu_log).squeeze().cuda(),
        v_support.cuda(),
    )

    optimizer = build_optimizer(ot_plan.parameters(), config)

    for step in range(config.steps):
        optimizer.zero_grad()
        if config["source"]["setting"] == "discrete":
            this_xidx = torch.multinomial(wx, config.batch_size).cuda()
            this_x = x[this_xidx].cuda()
        else:
            this_xidx = None
            this_x = x.sample((config.batch_size, 1)).cuda()
        if config["target"]["setting"] == "discrete":
            this_yidx = torch.multinomial(wy, config.batch_size).cuda()
            this_y = y[this_xidx].cuda()
        else:
            this_yidx = None
            this_y = y.sample((config.batch_size, 1)).cuda()

        loss = ot_plan.loss(this_x, this_y, yidx=this_yidx, xidx=this_xidx)
        loss.backward()
        optimizer.step()

        wandb.log({"objective": -loss.item()})
        if step % config["log_steps"] == 0:
            if config["source"]["setting"] == "discrete":
                uinterp = interpolate.interp1d(
                    x.detach().cpu().numpy().reshape(-1),
                    ot_plan.u.u.detach().cpu().numpy().reshape(-1),
                    fill_value="extrapolate",
                )
                u_x = u_support.detach().cpu().numpy()
                u_val = uinterp(u_x.reshape(-1))
            else:
                u_val = ot_plan.u(u_support.cuda()).detach().cpu().numpy()
                u_x = u_support
            u_fig = plt.figure()
            plt.plot(u_x, u_val, label="NN Approximation")
            plt.plot(
                u_support.squeeze().cpu().numpy(),
                u_exact.squeeze().cpu().numpy(),
                label="Sinkhorn solution",
            )
            plt.legend()
            if config["target"]["setting"] == "discrete":
                vinterp = interpolate.interp1d(
                    y.detach().numpy().reshape(-1),
                    ot_plan.v.u.detach().cpu().numpy().reshape(-1),
                    fill_value="extrapolate",
                )
                v_x = v_support.detach().cpu().numpy()
                v_val = vinterp(v_x.reshape(-1))
            else:
                v_val = ot_plan.v(v_support.cuda()).detach().cpu().numpy()
                v_x = v_support
            v_fig = plt.figure()
            plt.plot(v_x, v_val, label="NN Approximation")
            plt.plot(
                v_support.squeeze().cpu().numpy(),
                v_exact.squeeze().cpu().numpy(),
                label="Sinkhorn solution",
            )
            plt.legend()

            cost = l2_distance(u_support, v_support)
            transport_plan = np.exp(
                u_val[:, None] / config["alpha"]
                + mu_log[:, None].squeeze().numpy()
                + v_val[None, :] / config["alpha"]
                + nu_log[None, :].squeeze().numpy()
                - cost.numpy() / config["alpha"]
            )

            transport_plan_exact = np.exp(
                u_exact[:, None].squeeze().cpu().numpy() / config["alpha"]
                + mu_log[:, None].squeeze().numpy()
                + v_exact[None, :].squeeze().cpu().numpy() / config["alpha"]
                + nu_log[None, :].squeeze().numpy()
                - cost.numpy() / config["alpha"]
            )

            plans, (ax_plan, ax_plan_exact) = plt.subplots(1, 2)
            ax_plan.imshow(transport_plan)
            ax_plan.set_title("Approximate plan")
            ax_plan_exact.imshow(transport_plan_exact)
            ax_plan_exact.set_title("Exact plan - GeomLoss")

            wandb.log(
                {
                    "u_potential": u_fig,
                    "v_potential": v_fig,
                    "transport_plans": plans,
                }
            )
    run.finish()


if __name__ == "__main__":
    app()
