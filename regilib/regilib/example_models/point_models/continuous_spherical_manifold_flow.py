#!/usr/bin/env python3

import os
from enum import Enum
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchdyn.nn.node_layers as tdnl
from regilib.core.distributions import MultivariateNormal
from regilib.core.dynamics.dynamical_state import DynamicalState
from regilib.core.dynamics.dynamics import RegularisedDynamics, StateDynamics
from regilib.core.invertible_modules.charts import PadProj
from regilib.core.invertible_modules.continuous_ambient_flow import \
    ContinuousAmbientFlow
from regilib.core.invertible_modules.continuous_manifold_flow import \
    ContinuousManifoldFlow
from regilib.core.invertible_modules.normalising_flow import NormalisingFlow

from .point_distribution_model import PointDistributionModel


class State(Enum):
    """State that the manifold is in."""

    MANIFOLD_LEARNING = 1
    DENSITY_LEARNING = 2
    INFERENCE = 3

class FunctionDynamicsF(nn.Module):
    """ Models the dynamics of the injection to the manifold. """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # expected format: N x (C * L)
        # +1 for time
        self.fc1 = nn.Linear(in_channels + 1, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, out_channels)

    def forward(self, t, x):
        _x = x.clone()

        x = torch.cat([x,t], -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x

class FunctionDynamicsH(nn.Module):
    """ Models the dynamics of the ambient flow on the manifold. """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # expected format: N x (C * L)
        # +1 for time
        self.fc1 = nn.Linear(in_channels + 1, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, out_channels)

    def forward(self, t, x):
        _x = x.clone()

        x = torch.cat([x,t], -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x


class ContinuousSphericalManifoldFlow(NormalisingFlow, PointDistributionModel, pl.LightningModule):
    def __init__(self, base_distribution=None, chart=PadProj(), fdynF=None, fdynH=None):
        if base_distribution is None:
            base_distribution = MultivariateNormal(torch.zeros(2), torch.eye(2))
        if fdynH is None: fdynH = FunctionDynamicsH(2, 2)
        if fdynF is None: fdynF = FunctionDynamicsF(3, 3)

        super().__init__(base_distribution=base_distribution)

        # state=[l, e, n | state]
        self.aug1 = tdnl.Augmenter(augment_dims=3)
        self.af1 = ContinuousAmbientFlow(
            dynamics=RegularisedDynamics(fdyn=fdynH)
        )
        self.mf1 = ContinuousManifoldFlow(
            chart=chart, dynamics=StateDynamics(fdyn=fdynF),
            default_n_steps=5
        )

    def forward(self, x, af_estimate=True, mf_skip=True):
        ds = x.clone() if isinstance(x, DynamicalState) else DynamicalState(state=x)

        # p(z)
        ds = super().forward(ds)

        # u=h(z)
        ds = self.af1.dynamics.update_ds(ds, self.aug1(ds['state']))
        ds = self.af1.forward(ds, estimate_trace=af_estimate)

        # x=g(u)
        ds = self.mf1.forward(ds, skip_jacobian_det=mf_skip)
        return ds

    def inverse(self, x, af_estimate=True, mf_skip=True):
        ds = x.clone() if isinstance(x, DynamicalState) else DynamicalState(state=x)

        # u=g⁻¹(x)
        ds = self.mf1.inverse(ds, skip_jacobian_det=mf_skip)

        # z=h⁻¹(u)
        ds = self.af1.dynamics.update_ds(ds, self.aug1(ds['state']))
        ds = self.af1.inverse(ds, estimate_trace=af_estimate)

        # p(z)
        ds = super().inverse(ds)
        return ds

    #############
    # INTERFACE #
    #############
    def encode(self, X):
        ds = self.inverse(X)
        return ds['state'].cpu().detach()

    def decode(self, z):
        ds = self.forward(z)
        return ds['state'].cpu().detach()

    def log_likelihood(self, X):
        _, log_prob = self.inverse(X)
        return log_prob.cpu().detach()

    def sample_posterior(self, n_samples):
        ds = super().sample_posterior(n_samples)
        return ds['state'].detach().cpu()

    def fit(self, dataloader, log=True, root_dir=None, m_epochs=700,
            d_epochs=2000, gpus=[1], optimizer=None, **kwargs):
        self._optimizer = optimizer
        if not log:
            kwargs['checkpoint_callback'], kwargs['logger'] = False, False
            root_dir = None

        # MANIFOLD PHASE
        self.state = State.MANIFOLD_LEARNING
        self.mf1.freeze(False); self.af1.freeze(True)
        mpdir = os.path.join(root_dir, 'mp/') if root_dir is not None else None
        trainer = pl.Trainer(
            max_epochs=m_epochs, gpus=gpus, default_root_dir=mpdir, **kwargs)
        trainer.fit(self, dataloader);

        # DENSITY PHASE
        self.state = State.DENSITY_LEARNING
        self.mf1.freeze(True); self.af1.freeze(False)
        dpdir = os.path.join(root_dir, 'dp/') if root_dir is not None else None
        trainer = pl.Trainer(
            max_epochs=d_epochs, gpus=gpus, default_root_dir=dpdir, **kwargs)
        trainer.fit(self, dataloader);

        # INFERENCE PHASE
        self.state = State.INFERENCE
        self.eval()

    ############
    # TRAINING #
    ############
    def noise_enhance_data(self, x, b=0.1):
        nu = torch.randn(x.shape[0], 3, device=x.device) * b
        return x + nu

    def training_step(self, batch, batch_idx):
        x = batch.pos

        if self.state is State.MANIFOLD_LEARNING:
            loss = self.train_step_g(x)
        if self.state is State.DENSITY_LEARNING:
            loss = self.train_step_h(x)

        self.log('train_loss', loss)

        return {'loss': loss}


    def train_step_g(self, x):
        # add noise to data
        x_prime = self.noise_enhance_data(x, b=0.01)
        ds = DynamicalState(state=x_prime)

        # project point to manifold
        ds_u = self.mf1.inverse(ds, skip_jacobian_det=True)

        # reconstruct point
        ds_x = self.mf1.forward(ds_u.clone(), skip_jacobian_det=True)

        # boundary_error = (b1+b2+b3+b4).pow(2).sum(-1)

        mse = (x - ds_x['state']).pow(2).sum(-1)

        loss = mse.sum() / (x.shape[0]*x.shape[1])

        return loss

    def train_step_h(self, x):
        lambda_e, lambda_n = 0.01, 0.01

        # logp(z_t1) = logp(z_t0) - \int_0^1 - Tr ∂f/∂z(t)
        ds_z = self.inverse(x, af_estimate=True, mf_skip=True)

        # maximise the density a.k.a. likelihood => minimise negative log likelihood
        loss = (
            # log p(z(t₁)) = log p(z(t₀)) - \int_0^1 - Tr ∂f/∂z(t)
            -ds_z.log_prob + lambda_e * ds_z.e[:, 0] + lambda_n * ds_z.n[:, 0]
        ).sum() / (x.shape[0]*x.shape[1])

        return loss

    def __str__(self):
        return 'manifold_flow_pdm'


    def configure_optimizers(self):
        if self._optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=2e-3, weight_decay=1e-4)
            return {
                'optimizer': optimizer,
                'lr_scheduler':
                torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                        min_lr=1.e-10,
                                                        factor=0.9,
                                                        verbose=True,
                                                        patience=1000),
                'monitor': 'train_loss'
            }
        else:
            return self._optimizer
