#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
import torch.nn as nn
from regilib.core.distributions import MultivariateNormal
from regilib.core.dynamics.dynamics import RegularisedDynamics, StateDynamics
from regilib.core.invertible_modules.bijective import \
    ContinuousAmbientFlow
from regilib.core.invertible_modules.bijective import \
    ContinuousManifoldFlow
from regilib.core.invertible_modules.normalising_flow import NormalisingFlow

from regilib.example_models.point_models.point_distribution_model import PointDistributionModel
from regilib.example_models.point_models.continuous_ambient_flow import ContinuousAmbientFlowPDM

from torch_geometric.transforms import SamplePoints
from regilib.core.dynamics.dynamical_state import DynamicalState
import torchdyn.nn.node_layers as tdnl
from torch_geometric.data import Data
from .point_flow_model import PointFlowModel

class ContinuousAmbientFlowPCASM(PointFlowModel, pl.LightningModule):
    def __init__(self, shape_distribution, conditional_point_distribution=None):
        super().__init__()

        self._shape_distribution = shape_distribution
        if conditional_point_distribution is None:
            self._point_distribution = ContinuousAmbientFlowPDM(
                fdyn = ContinuousAmbientFlowPDM.FunctionDynamics(
                    3, 3, shape_distribution.n_components))

    def __str__(self):
        return 'PCA_CAF_point_flow_model'

    # Region nn.Model
    def forward(self, x, af_estimate=True, n_point_samples=2**10):
        if hasattr(x, 'batch'):
            n_shapes = x.batch.max() + 1
        else: n_shapes = 1


        breakpoint()
        # check that the meshes are equal sized
        assert x.batch.unique(return_counts=True)[1].all()

        # conditional variables
        x_tensor = x.pos.view(n_shapes, -1)
        shape_z = self.shape_distribution.inverse(x_tensor.view(n_shapes, -1))
        conditions = shape_z.unsqueeze(
            1).repeat(1, n_point_samples, 1).view(-1, self.n_conditions)

        # random initial point distribution
        sampler = SamplePoints(num=n_point_samples)
        pd_x = torch.stack([sampler(d).pos for d in x.to_data_list()])

        # encode the conditional pointcloud
        ds = DynamicalState(state=pd_x.view(-1, 3), condition=conditions)
        ds = self.conditional_point_distribution.inverse(ds, af_estimate=True)
        return ds

    # Region PointFlowModel
    @property
    def n_conditions(self):
        return self.shape_distribution.n_components

    @property
    def conditional_point_distribution(self):
        return self._point_distribution

    @property
    def shape_distribution(self):
        return self._shape_distribution

    def encode_shape(self, x:Data) -> torch.Tensor:
        return self.shape_distribution.encode(x)

    def decode_shape(self, z: torch.Tensor):
        return self.shape_distribution.decode(z)

    def log_prob_shape(self, x:Data) -> torch.Tensor:
        return self.shape_distribution.log_likelihood(x)

    def encode_point(self, x, condition):
        return self.conditional_point_distribution.encode(x, condition)

    def decode_point(self, z):
        return self.conditional_point_distribution.decode(z, condition)

    def log_prob_point(self, x:torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.conditional_point_distribution.log_likelihood(x, condition)

    def log_likelihood(self, x:Data):
        n_shape_samples = x.batch.max() + 1
        n_point_samples = int(x.pos.shape[0] / (x.batch.max() + 1))

        # SHAPE
        # B x Z
        shape_zs = self.shape_distribution.encode(x)
        log_prob_shape = self.shape_distribution.encode(x)

        # B x N x Z
        conditions = shape_zs.unsqueeze(1).repeat(
            1, n_point_samples, 1).view(-1, self.n_conditions)
        log_prob_shape = shape_zs.unsqueeze(1).repeat(
            1, n_point_samples, 1).view(-1, self.n_conditions)

        # POINT
        log_prob_point = self.log_prob_point(x.pos, conditions)

        # p(x,s) = p(x|s)p(s)
        joint_log_prob = log_prob_shape + log_prob_point

        return joint_log_prob

    def sample_posterior(self, n_shape_samples, n_point_samples):
        condition = self.shape_distribution.sample_prior(n_shape_samples)
        condition = condition.unsqueeze(1).repeat(1, n_point_samples, 1)
        data = self.sample_conditional_posterior(condition, n_point_samples)

        # TODO: add batch support
        #data.batch=

        return data

    def sample_conditional_posterior(self, condition, n_point_samples):
        # TODO: add batch support
        data = self.conditional_point_distribution.sample_posterior(
            n_point_samples, condition.view(-1, self.n_conditions))

        return data

    # region training
    def fit(self, dataloader, log=True, epochs=750, gpus=[1], root_dir=None, **kwargs):
        if not log:
            kwargs['checkpoint_callback'], kwargs['logger'] = False, False
            root_dir = None

        trainer = pl.Trainer(max_epochs=epochs, gpus=gpus, default_root_dir=root_dir, **kwargs)
        trainer.fit(self, dataloader);

    def training_step(self, batch, batch_idx):
        x = batch

        lambda_e, lambda_n = 0.01, 0.01

        # logp(z_t1) = logp(z_t0) - \int_0^1 - Tr ∂f/∂z(t)
        ds_z = self.forward(x, af_estimate=True)

        # maximise the density a.k.a. likelihood => minimise negative log likelihood
        loss = (
            # log p(z(t₁)) = log p(z(t₀)) - \int_0^1 - Tr ∂f/∂z(t)
            -ds_z.log_prob + lambda_e * ds_z.e[:, 0] + lambda_n * ds_z.n[:, 0]
        ).sum() / (ds_z.state.shape[0]*ds_z.state.shape[1])

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
