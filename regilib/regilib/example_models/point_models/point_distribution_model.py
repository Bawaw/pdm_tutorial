#!/usr/bin/env python3
import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Union
from torch_geometric.data import Data


class PointDistributionModel(ABC):
    """Base class for point distribution models, this point-distribution
    model can be conditional.

    """

    @abstractmethod
    def fit(self, dataloader: DataLoader, **kwargs):
        """Fit the model to the data in dataloader.

        :param dataloader: the dataset as a DataLoader object.
        :returns: None

        """
        pass

    @abstractmethod
    def encode(
            self, x:Data,
            condition:Union[torch.Tensor, None]=None) -> torch.Tensor:

        """Deform the point distribution in the normalising direction.

        :param x: Current state of the point-distribution.
        :param condition: variables to condition on
        :returns: new state of the point-distribution.

        """
        pass

    @abstractmethod
    def decode(
            self, z:torch.Tensor,
            condition:Union[torch.Tensor, None]=None) -> torch.Tensor:

        """Deform the point distribution in the generative direction.

        :param z: Current state of the point-distribution.
        :param condition: variables to condition on
        :returns: new state of the point-distribution.

        """
        pass

    @abstractmethod
    def log_likelihood(
            self, x:Data,
            condition:Union[torch.Tensor, None]=None) -> torch.Tensor:

        """Return the log-likelihood of each sample

        :param x: Point-distribution
        :param condition: variables to condition on
        :returns: log-likelihood

        """
        pass

    @abstractmethod
    def sample_posterior(
            self, n_samples:int,
            condition:Union[torch.Tensor, None]=None) -> Data:

        """Sample the prior distribution and deform it in the
        generative direction, generating a point-distribution.

        :param n_samples: number of generative samples
        :param condition: variables to condition on
        :returns: Point-distribution

        """
        pass
