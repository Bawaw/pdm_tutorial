#!/usr/bin/env python3

import torch
import torch.nn as nn

from torch_geometric.data import Data
from .shape_distribution_model import ShapeDistributionModel
from regilib.core.invertible_modules import InvertibleModule

class PCAWrapper(ShapeDistributionModel, InvertibleModule):
    def __init__(self, pca_model):
        super().__init__()
        self.pca = pca_model
        self.register_buffer('mean', torch.from_numpy(self.pca.mean_).float())
        self.register_buffer('components', torch.from_numpy(self.pca.components_).float())
        self.register_buffer('explained_variance', torch.from_numpy(self.pca.explained_variance_).float())


    # InvertibleModule region
    def forward(self, X):
        return X @ (torch.sqrt(self.explained_variance[:, None]) * self.components) + self.mean

    def inverse(self, X):
        X = X - self.mean
        X_transformed = X @ self.components.T
        X_transformed /= torch.sqrt(self.explained_variance)
        return X_transformed

    # ShapeDistributionModel region
    @property
    def n_components(self):
        return self.components.shape[0]

    def fit(self, dataloader):
        raise NotImplementedError(
            "This is a wrapper for a pre-trained pca model.")

    def encode(self, x: Data) -> torch.Tensor:
        if hasattr(x, 'batch'):
            n_shapes = x.batch.max() + 1
        else: n_shapes = 1

        tensorised_x = x.pos.view(n_shapes, -1)
        return self.inverse(tensorised_x).detach().cpu()

    def decode(self, z: torch.Tensor) -> Data:
        return Data(pos=self.forward(z).detach().cpu().view(-1, 3))

    def log_likelihood(self, x: Data) -> torch.Tensor:
        if hasattr(x, 'batch'):
            n_shapes = x.batch.max() + 1
        else: n_shapes = 1

        tensorised_x = x.pos.view(n_shapes, -1)
        return torch.from_numpy(self.pca.score_samples(tensorised_x))

    def sample_prior(self, n_samples: int):
        return torch.randn(n_samples, 2)

    def sample_posterior(self, n_samples: int) -> Data:
        return self.forward(self.sample_prior(n_samples))
