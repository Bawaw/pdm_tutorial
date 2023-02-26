#!/usr/bin/env python3

import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Union

class ShapeDistributionModel(ABC):
    """Base class for shape distribution models."""

    @abstractmethod
    def n_components(self):
        """Number of dimensions used in the encoding. """

        pass

    @abstractmethod
    def fit(self, dataloader:DataLoader):
        """Fit the model to the data in dataloader.

        :param dataloader: the dataset as a DataLoader object.
        :returns: None
        """
        pass

    @abstractmethod
    def encode(self, x:Data) -> torch.Tensor:
        """Deform the shapes in the normalising direction.

        :param X: Current state of the shapes
        :returns: encoded state of the shapes.

        """
        pass

    @abstractmethod
    def decode(self, z:torch.Tensor) -> Data:
        """Deform the shapes in the generalising direction.

        :param x: encoded state of the shapes
        :returns: decoded state of the shapes.

        """

        pass

    @abstractmethod
    def log_likelihood(self, x:Data) -> torch.Tensor:
        pass

    @abstractmethod
    def sample_prior(self, n_samples: int):
        pass

    @abstractmethod
    def sample_posterior(self, n_samples: int) -> Data:
        pass
