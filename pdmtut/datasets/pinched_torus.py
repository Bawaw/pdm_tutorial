#!/usr/bin/env python3

import math

import torch
import torch.distributions as tdist
import torch.utils.data as tdata
from regilib.core.dynamics.dynamical_state import DynamicalState
from regilib.core.invertible_modules.charts import PinchedTorusCoordProj
from regilib.core.invertible_modules.bijective import AffineTransform

class DBetaDistribution:
    """Joint beta distribution of two independent variables."""
    def __init__(self, alpha, beta, d=2):
        self.d = d
        self.dist = tdist.beta.Beta(alpha, beta)

    def rsample(self, n_samples):
        return self.dist.rsample(n_samples + [self.d])

    def log_prob(self, x):
        return self.dist.log_prob(x).sum(1)


class PinchedTorus():
    def __init__(self, n_samples=100, seed=11, normalise=True):

        self.base_distribution = DBetaDistribution(2, 5)
        self._h = AffineTransform(0, 2*math.pi)
        self._g = PinchedTorusCoordProj()

        # sample dataset
        self._X, self._log_prob = self.sample_points_randomly(
            n_samples, seed)
        self._shape = self._X.shape

        # standardise
        self._mean = self._X.mean(0)
        self._std = (1 / (self._X - self._mean).abs().max()) * 0.999999

        if normalise: self._X = self.normalise_scale(self._X)



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
    def raw_data(self):
        return self.unnormalise_scale(self._X.clone())

    @property
    def uv_extremes(self):
        u = torch.tensor([1/6*math.pi, 7/6*math.pi, 5/6*math.pi, 11/6*math.pi])
        v = torch.tensor([1/3*math.pi, 1/3*math.pi, 2/3*math.pi, 2/3*math.pi])
        return torch.stack([u, v], -1)

    @property
    def cartesian_extremes(self):
        uv = self.get_uv_extremes()
        extreme_points = self._g(uv)
        return self.normalise(extreme_points)

    def sample_points_randomly(
            self, n_samples, seed, return_intermediate_steps=False):

        if seed is not None: torch.manual_seed(seed)

        # 1) ds_z ~ p(z)
        ds_z = DynamicalState(state=self.base_distribution.rsample(
            [n_samples]).requires_grad_(True))
        ds_z['log_prob'] = self.base_distribution.log_prob(ds_z.state)

        # 2) u = h(z)
        ds_u = self._h(ds_z.clone())

        # 3) y = g(u)
        ds_y = self._g(ds_u.clone())

        X, log_prob = ds_y[
            'state'].detach(), ds_y['log_prob'].detach()

        if not return_intermediate_steps: return X, log_prob
        else: return (X, log_prob), (ds_z, ds_u, ds_y)

    def sample_points_uniformly(
            self, n_samples, seed, return_intermediate_steps=False):
        n_points = math.floor(math.sqrt(n_samples))

        # 1) ds_z ~ p(z)
        ds_z = DynamicalState(state=torch.stack(torch.meshgrid(
            torch.linspace(0, 1, n_points),
            torch.linspace(0, 1, n_points)
        ),-1).view(n_samples, -1).requires_grad_(True))
        ds_z['log_prob'] = self.base_distribution.log_prob(ds_z.state)

        # 2) u = h(z)
        ds_u = self._h(ds_z.clone())

        # 3) y = g(u)
        ds_y = self._g(ds_u.clone())

        if not return_intermediate_steps:
            return ds_y['state'].detach(), ds_y['log_prob'].detach()
        else:
            return (ds_y['state'].detach(), ds_y['log_prob'].detach(
            )), (ds_z, ds_u, ds_y)

    def normalise_scale(self, X):
        return (X-self.mean) * self.std

    def unnormalise_scale(self, x):
        return (x/self.std) + self.mean

    def train_loader(self, batch_size=10, num_workers=8):
        X_train = torch.Tensor(self._X).view(self._X.shape[0], -1)
        train = tdata.TensorDataset(X_train)

        return tdata.DataLoader(
            train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers)

# ptd = PinchedTorusDataset()
# import pyvista as pv
# state, log_prob = ptd.sample_points_uniformly(100**2, 11)
# plotter = pv.Plotter()
# plotter.add_mesh(pv.StructuredGrid(*state.T.numpy()))
# #plotter.add_mesh(pv.StructuredGrid(*state.T.numpy()))
# plotter.camera_position = [(5.25,0,0),(0,0,0),(0,0,1)]
# plotter.show(window_size = [1000,1000])

# state, log_prob = ptd.sample_points_uniformly(100**2, 11)

# breakpoint()
# import pyvista as pv
# X = state[:,0].view(100, 100).numpy()
# Y = state[:,1].view(100, 100).numpy()
# Z = state[:,2].view(100, 100).numpy()
# pv.StructuredGrid(X,Y,Z).plot()
