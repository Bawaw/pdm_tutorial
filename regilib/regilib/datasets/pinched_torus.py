#!/usr/bin/env python3

import math

import torch
import torch.distributions as tdist
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.distributions.transformed_distribution import \
    TransformedDistribution
from torch.distributions.transforms import AffineTransform
from torch_geometric.data import (Data, DataLoader, InMemoryDataset,
                                  download_url)


class PinchedTorus(InMemoryDataset, LightningDataModule):
    """ Skulls generated using a SSM on the Paris dataset skulls. """

    def __init__(
            self, n_samples=500, seed=10,
            transform=None, pre_transform=None):
        """
        Parameters
        ----------
        root : str
            Root folder
        n_samples : int, optional
            Number of generative samples
        n_components : int, optional
            Number of components used to generate data
        seed : int, optional
            Seed used to generate data
        """
        super().__init__(None, transform, pre_transform)

        self.dist = TransformedDistribution(
            tdist.beta.Beta(2, 5), AffineTransform(0, 2*math.pi))

        # draw u, v coords according to distribution
        torch.manual_seed(seed)
        v, u = self.dist.sample((2, n_samples))

        # generate new data
        X = self.gen_data(u, v)
        data_list = [Data(pos=X)]
        self.data, self.slices = self.collate(data_list)


    def gen_data(self, u, v):
        R, r = 0.8, 0.5
        x = (R + r*torch.cos(v))*torch.cos(u)
        y = (R + r*torch.cos(v))*torch.sin(u)
        z = r*torch.sin(v)*torch.sin(u/2)

        return torch.stack([x,y,z], -1)

    @property
    def data_shape(self):
        return torch.tensor(self.X.shape[1:])

    def train_dataloader(self, batch_size = 10, shuffle=True, num_workers=8):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
