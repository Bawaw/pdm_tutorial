#!/usr/bin/env python3

import math

import pyvista as pv
from pyvista import examples as pv_examples
import torch
import torch.distributions as tdist
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.distributions.transformed_distribution import \
    TransformedDistribution
from torch.distributions.transforms import AffineTransform
from torch_geometric.data import (Data, DataLoader, InMemoryDataset,
                                  download_url)


class PvExampleAdapter(InMemoryDataset, LightningDataModule):
    """ Skulls generated using a SSM on the Paris dataset skulls. """

    def __init__(
            self,
            pv_download_function=pv_examples.download_nefertiti, seed=10,
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

        mesh = pv_download_function().triangulate()

        faces = torch.stack(torch.from_numpy(mesh.faces).split(4))[:, 1:].T
        pos = torch.from_numpy(mesh.points)

        data_list = [Data(pos=pos, face=faces)]
        self.data, self.slices = self.collate(data_list)

    @property
    def data_shape(self):
        return torch.tensor(self.X.shape[1:])

    def train_dataloader(self, batch_size = 10, shuffle=True, num_workers=8):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
