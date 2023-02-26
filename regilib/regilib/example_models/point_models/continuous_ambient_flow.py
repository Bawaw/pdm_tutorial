#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchdyn.nn.node_layers as tdnl
from regilib.core.distributions import MultivariateNormal
from regilib.core.dynamics.dynamical_state import DynamicalState
from regilib.core.dynamics.dynamics import RegularisedDynamics
from regilib.core.invertible_modules.bijective import ContinuousAmbientFlow
from regilib.core.invertible_modules.normalising_flow import NormalisingFlow
from torch_geometric.data import Data

from .point_distribution_model import PointDistributionModel


class ContinuousAmbientFlowPDM(
        NormalisingFlow, PointDistributionModel, pl.LightningModule):
    """PointDistribution modelled using continuous ambient flows."""

    class FunctionDynamics(nn.Module):
        """ Models the dynamics of the injection to the manifold. """

        def __init__(self, in_channels, out_channels, n_conditions=0):
            super().__init__()
            self._in_channels = in_channels
            self._out_channels = out_channels
            self._n_conditions = n_conditions

            # expected format: N x (C * L)
            # +1 for time
            self.fc1 = nn.Linear(in_channels + n_conditions + 1, 128)
            self.fc2 = nn.Linear(128, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc4 = nn.Linear(256, 128)
            self.fc5 = nn.Linear(128, out_channels)

        @property
        def in_channels(self):
            return self._in_channels

        @property
        def out_channels(self):
            return self._out_channels

        @property
        def n_conditions(self):
            return self._n_conditions

        def forward(self, ds):
            if hasattr(ds, 'condition'): x = torch.cat([
                    ds.state, ds.condition, ds.t], -1)
            else: x = torch.cat([ds.state, ds.t], -1)

            #x = torch.cat([x,t], -1)
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            x = self.fc5(x)
            return x


    def __init__(self, fdyn=None, base_distribution=None):
        if base_distribution is None:
            base_distribution = MultivariateNormal(torch.zeros(3), torch.eye(3))
        if fdyn is None:
            fdyn=ContinuousAmbientFlowPDM.FunctionDynamics(3, 3)

        super().__init__(base_distribution=base_distribution)

        # state=[l, e, n | state]
        self.aug1 = tdnl.Augmenter(augment_dims=3)
        self.af1 = ContinuousAmbientFlow(dynamics=RegularisedDynamics(fdyn=fdyn))

    def __str__(self):
        return 'ambient_flow_pdm'

    @property
    def n_conditions(self):
        return self.af1.dynamics.n_conditions

    # Region NormalisingFlow
    def forward(self, ds:DynamicalState, af_estimate=True):
        ds = super().forward(ds)
        ds = self.af1.dynamics.update_ds(ds, self.aug1(ds['state']))
        ds = self.af1.forward(ds, estimate_trace=af_estimate)
        return ds

    def inverse(self, ds, af_estimate=True):
        ds = self.af1.dynamics.update_ds(ds, self.aug1(ds['state']))
        ds = self.af1.inverse(ds, estimate_trace=af_estimate)
        ds = super().inverse(ds)
        return ds

    # Region PointDistribution
    def fit(
            self, dataloader, log=True, epochs=750, gpus=[1], root_dir=None, **kwargs):
        if not log:
            kwargs['checkpoint_callback'], kwargs['logger'] = False, False
            root_dir = None

        trainer = pl.Trainer(max_epochs=epochs, gpus=gpus, default_root_dir=root_dir, **kwargs)
        trainer.fit(self, dataloader);

    def encode(self, x:Data, condition=None) -> torch.Tensor:
        ds = DynamicalState(state=x.pos)
        if condition is not None: ds['condition'] = condition

        ds = self.inverse(ds)
        return ds['state'].cpu().detach()

    def decode(self, z:torch.Tensor, condition=None) -> Data:
        ds = DynamicalState(state=z)
        if condition is not None: ds['condition'] = condition

        ds = self.forward(ds)
        return Data(pos=ds['state'].cpu().detach())

    def log_likelihood(self, x:Data, condition=None) -> torch.Tensor:
        ds = DynamicalState(state=x.pos)
        if condition is not None: ds['condition'] = condition

        ds = self.inverse(ds, af_estimate=False)
        return ds.log_prob.cpu().detach()

    def sample_posterior(self, n_samples:int, condition=None) -> Data:
        ds = super().sample_posterior(n_samples, condition)
        return Data(pos=ds['state'].cpu().detach())

    # Region Training
    def training_step(self, batch, batch_idx):
        x = batch.pos

        lambda_e, lambda_n = 0.01, 0.01

        # logp(z_t1) = logp(z_t0) - \int_0^1 - Tr ∂f/∂z(t)
        ds_z = self.inverse(DynamicalState(state=x), af_estimate=True)

        # maximise the density a.k.a. likelihood => minimise negative log likelihood
        loss = (
            # log p(z(t₁)) = log p(z(t₀)) - \int_0^1 - Tr ∂f/∂z(t)
            -ds_z.log_prob + lambda_e * ds_z.e[:, 0] + lambda_n * ds_z.n[:, 0]
        ).sum() / (x.shape[0]*x.shape[1])

        self.log('train_loss', loss)

        return {'loss': loss}

    def configure_optimizers(self):
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

