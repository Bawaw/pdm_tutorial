#!/usr/bin/env python3
import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Union
from torch_geometric.data import Data

class PointFlowModel(ABC):
    """Base class for hybrid model that contains shape distribution and
    conditional point-distribution.
    """

    @property
    def n_conditions(self):
        pass

    @property
    def conditional_point_distribution(self):
        pass

    @property
    def shape_distribution(self):
        pass

    @abstractmethod
    def fit(self, dataloader:DataLoader):
        """Fit the model to the data in dataloader.

        :param dataloader: the dataset as a DataLoader object.
        :returns: None
        """
        pass

    @abstractmethod
    def encode_shape(self, x:Data) -> torch.Tensor:
        """Deform the shapes in the normalising direction.

        :param x: Current state of the shapes
        :returns: encoded state of the shapes.
        """
        pass

    @abstractmethod
    def decode_shape(self, z:torch.Tensor) -> Data:
        """Deform the shapes in the generalising direction.

        :param z: encoded state of the shapes
        :returns: decoded state of the shapes.

        """

        pass

    @abstractmethod
    def log_prob_shape(self, x:Data) -> torch.Tensor:
        """Return density function value for input shape.

        :param x: shapes
        :returns: density
        """


        pass

    @abstractmethod
    def encode_point(self, x:Data, condition:torch.Tensor) -> torch.Tensor:

        """Deform the point distribution in the normalising direction.

        :param X: Current state of the points
        :param condition: variables to condition on
        :returns: new state of the points

        """

        pass

    @abstractmethod
    def decode_point(self, z, condition:Union[
            torch.Tensor, None]=None) -> Data:

        """Deform the point distribution in the generative direction.

        :param z: Current state of the points.
        :param condition: variables to condition on
        :returns: new state of the points.

        """

        pass

    @abstractmethod
    def log_prob_point(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Return density function value for input shape.

        :param x: shapes
        :returns: density
        """

        pass


    @abstractmethod
    def sample_posterior(
            self, n_shape_samples:int, n_point_samples:int) -> torch.Tensor:

        """Sample the prior distribution and deform it in the
        generative direction, generating a point-cloud.

        :param n_samples: number of generative samples
        :param condition: variables to condition on
        :returns: Point-cloud

        """
        pass

    @abstractmethod
    def log_likelihood(self, X:Data) -> torch.Tensor:

        """Return the log-likelihood of each shape sample

        :param X: batched shape
        :param condition: variables to condition on
        :returns: log-likelihood

        """
        pass
