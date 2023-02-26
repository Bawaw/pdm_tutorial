#!/usr/bin/env python3
import glob
import os

import numpy as np
from joblib import dump, load
from sklearn.decomposition import PCA

import pyvista as pv
import torch
from pytorch_lightning.core.datamodule import LightningDataModule
from regilib.example_models.registration_models.affine_registration import \
    AffineRegistration
from regilib.io.ply import read_ply
from torch_geometric.data import (Data, DataLoader, InMemoryDataset,
                                  download_url)

from regilib.vis.plotter import Plotter

class SyntheticSkulls(InMemoryDataset, LightningDataModule):
    """ Skulls generated using a SSM on the Paris dataset skulls. """

    def __init__(
            self, root, n_samples=500, n_components=2, seed=10,
            transform=None, pre_transform=None, center_skulls=True, affine_align=True):
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

        self.n_samples, self.seed = n_samples, seed
        self.n_components = n_components
        self.affine_align = affine_align
        self.affine_transform = AffineRegistration()
        self.model = PCA(n_components, random_state=seed, whiten=True)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def pca_model(self):
        path = os.path.join(self.processed_dir, 'pca_model.joblib')
        return load(path)

    @property
    def raw_file_names(self):
        file_list = glob.glob(self.raw_dir, '*/bone_template.ply')
        file_list.sort()
        return file_list

    @property
    def _raw_in_correspondence_data(self):
        ply_files = glob.glob(os.path.join(self.raw_dir, "*/bone_template.ply"))

        mesh_data = [read_ply(ply_file) for ply_file in ply_files]

        return mesh_data

    def _generate_synthetic_data(self, template, data):
        np.random.seed(self.seed)
        z_samples = np.random.randn(self.n_samples, self.n_components)

        data_samples = self.model.inverse_transform(z_samples)

        generated_mesh_data = []
        for data_sample in data_samples:
            t = template.clone()
            t.pos = torch.from_numpy(
                data_sample).view(-1, 3).float() # convert to N×3 coordinates
            generated_mesh_data.append(t)

        return generated_mesh_data

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # 1) get the in correspondence data
        data_list = self._raw_in_correspondence_data

        # 2) preprocess the data
        data_list = [d for d in data_list if d is not None]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 3) get random template
        template = data_list[0].clone()

        # 4) affine align scans to template
        if self.affine_align:
            data_list = [self.affine_transform(template, data) for data in data_list]

        # 5) fit the pca model
        data_list = torch.stack([d.pos for d in data_list if d])
        data_list = data_list.view(data_list.shape[0], -1)
        self.model.fit(data_list)
        dump(self.model, os.path.join(self.processed_dir, 'pca_model.joblib'))

        # 6) generate synthetic data using the pca model
        data_list = self._generate_synthetic_data(template, data_list)
        assert len(data_list) == self.n_samples

        # 7) Store generated data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def train_dataloader(self, batch_size=32, shuffle=True, num_workers=8):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    from regilib.vis.plotter import Plotter
    from torch_geometric.transforms import NormalizeScale

    dataset = SyntheticSkulls('./synthetic_skulls/data/SYNTHETIC_PARIS_SKULLS/', transform = NormalizeScale())
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for data in data_loader:
        plotter = Plotter()
        plotter.add_generic(data)
        plotter.show_bounds()
        plotter.camera_position = [(0,-5,0),(0,0,0),(0,0,1)]
        plotter.show()
