#!/usr/bin/env python3

import math

import torch
import torch.distributions as tdist
import torch.utils.data as tdata
from regilib.core.dynamics.dynamical_state import DynamicalState
from regilib.core.invertible_modules.charts import SwissRollCoordProj
from regilib.core.invertible_modules.bijective import AffineTransform

class JointDistribution:
    """Joint beta distribution of two independent variables ψ, γ."""
    def __init__(self, alpha=1, beta=3, low=-1., high=1.):

        self.dist_psi = tdist.beta.Beta(alpha, beta)
        self.dist_gamma = tdist.uniform.Uniform(low, high)

    def rsample(self, n_samples):
        return torch.stack([
            self.dist_psi.rsample(n_samples),
            self.dist_gamma.rsample(n_samples)
        ], -1)

    def log_prob(self, x):
        return self.dist_psi.log_prob(x[:, 0]) + self.dist_gamma.log_prob(x[:, 1])


class SwissRoll():
    def __init__(self, n_samples=100**2, seed=11, normalise=True):

        self.base_distribution = JointDistribution()
        self._h = AffineTransform(0, 4*math.pi)
        self._g = SwissRollCoordProj()

        # sample dataset
        (self._X, self._log_prob, self._index_colors
         ) = self.sample_points_randomly(n_samples, seed)

        self._shape = self._X.shape

        # standardise
        self._mean = self._X.mean(0)
        self._std = (1 / (self._X - self._mean).abs().max()) * 0.999999

        if normalise: self._X = self.normalise_scale(self._X)

        n_training_examples = math.floor(0.9*self._shape[0])
        self._X_train = self._X[:n_training_examples]
        self._X_val = self._X[n_training_examples:]

    @property
    def X(self):
        return torch.Tensor(self._X).view(self._X.shape[0], -1)

    @property
    def X_train(self):
        return torch.Tensor(self._X_train).view(self._X_train.shape[0], -1)

    @property
    def X_val(self):
        return torch.Tensor(self._X_val).view(self._X_val.shape[0], -1)

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    @property
    def shape(self):
        return torch.tensor(self._shape)

    @property
    def log_prob(self):
        return self._log_prob

    @property
    def index_colors(self):
        return self._index_colors

    @property
    def raw_data(self):
        return self.unnormalise_scale(self._X.clone())

    @property
    def z_extremes(self):
        u = torch.tensor([0.0, 0.5,  0.5,  1.0])
        v = torch.tensor([0.5, 0.5, -0.5, -0.5])
        return torch.stack([u, v], -1)

    @property
    def y_extremes(self):
        ds_z = DynamicalState(
            state=self.z_extremes)
        ds_u = self._h(ds_z)
        ds_y = self._g(ds_u)

        return self.normalise_scale(ds_y.state)

    def color_map(self, z):
        r, g, b = z[:,0], 0.5*torch.ones(z.shape[0]), (z[:,1] + 1)/2
        return torch.stack([r,g,b], -1)

    def gen_data_from_initial_tensor(self, z, return_intermediate_steps=False):
        # 1) ds_z ~ p(z)
        ds_z = DynamicalState(state=z.requires_grad_(True))
        ds_z['log_prob'] = self.base_distribution.log_prob(ds_z.state)

        # 2) u = h(z)
        ds_u = self._h(ds_z.clone())

        # 3) y = g(u)
        ds_y = self._g(ds_u.clone())

        X = ds_y['state'].detach()
        log_prob = ds_y['log_prob'].detach()
        index_colors = self.color_map(ds_z.state).detach()

        if not return_intermediate_steps: return X, log_prob, index_colors
        else: return (X, log_prob, index_colors), (ds_z, ds_u, ds_y)

    def sample_points_randomly(
            self, n_samples, seed, return_intermediate_steps=False):

        if seed is not None: torch.manual_seed(seed)

        z = self.base_distribution.rsample([n_samples])
        return self.gen_data_from_initial_tensor(z, return_intermediate_steps)

    def sample_points_uniformly(
            self, n_samples, seed, return_intermediate_steps=False):
        n_points = math.floor(math.sqrt(n_samples))

        if seed is not None: torch.manual_seed(seed)
        z = torch.stack(torch.meshgrid(
            torch.linspace(0.01, 0.99, n_points),
            torch.linspace(-0.99, 0.99, n_points)
        ), -1).view(-1, 2)

        return self.gen_data_from_initial_tensor(z, return_intermediate_steps)

    def normalise_scale(self, X):
        return (X-self.mean) * self.std

    def unnormalise_scale(self, x):
        return (x/self.std) + self.mean

    def X_loader(self, batch_size=10, num_workers=8, shuffle=True):
        X_dataset = tdata.TensorDataset(self._X)

        return tdata.DataLoader(
            X_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers)

    def train_loader(self, batch_size=10, num_workers=8):
        X_train_dataset = tdata.TensorDataset(self._X_train)

        return tdata.DataLoader(
            X_train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers)

    def validation_loader(
            self, batch_size=10, num_workers=8):
        X_val_dataset = tdata.TensorDataset(self._X_val)
        return tdata.DataLoader(
            X_val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers)

# if __name__ == "__main__":
#     import pyvista as pv
#     ptd = SwissRoll(n_samples=100**2)
#     state, log_prob = ptd.sample_points_uniformly(100**2, 11)

#     pv.set_plot_theme("document")
#     plotter = pv.Plotter()
#     plotter.add_mesh(
#         pv.StructuredGrid(*state.view(100, 100, 3).permute(2, 0, 1).numpy()),
#         scalars=log_prob, style='surface', pbr=True, metallic=1.,
#         scalar_bar_args={'title':'Log probability'}
#     )

#     plotter.add_light(pv.Light(
#         position=(-65, 0, -65), show_actor=True, positional=True,
#         cone_angle=20, intensity=2.))
#     plotter.add_light(pv.Light(
#         position=(0, 0, -65), show_actor=True, positional=True,
#         cone_angle=20, intensity=1.))
#     plotter.camera_position = [(-65, 0, 65), (0, 0, 0), (0, 1, 0)]
#     plotter.show(window_size=[800,800])

    # plotter = pv.Plotter()
    # plotter.add_mesh(
    #     pv.StructuredGrid(*state.view(100, 100, 3).T.numpy()),
    #     scalars=log_prob,
    # )

    # plotter.add_mesh(
    #     pv.polydata(ptd.raw_data.numpy()),
    #     render_points_as_spheres=true, point_size=10,
    #     diffuse=0.99, specular=0.8, ambient=0.3, smooth_shading=true,
    #     scalars=ptd.log_prob,
    #     style='points'
    # )

    # plotter.show_grid()
    # plotter.camera_position = [(-65, 0, 65), (0, 0, 0), (0, 1, 0)]
    # _=plotter.show(window_size=[800,800])

    # _, (ds_z, _, ds_y) = ptd.sample_points_uniformly(100**2, seed=11, return_intermediate_steps=True)

    # plotter = pv.Plotter()
    # plotter.add_mesh(
    #     pv.StructuredGrid(*ds_y.state.view(100, 100, 3).permute(2, 0, 1).detach().numpy()),
    #     scalars=ptd.color_map(ds_z.state.detach()).numpy(), rgb=True
    # )

    # plotter.show_grid()
    # plotter.camera_position = [(-65, 0, 65), (0, 0, 0), (0, 1, 0)]
    # _=plotter.show(window_size=[800,800])
