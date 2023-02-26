#!/usr/bin/env python3

import torch

from dynamics import Dynamics
from pykeops.torch import LazyTensor

class HamiltonianDynamics(Dynamics):
    """Just state in dynamics"""

    def __init__(self, kernel=None, **kwargs):
        super().__init__(**kwargs)

        if kernel is None:
            self.kernel = lambda x, y: (
                -((x - y)**2).sum(-1)/(self.sigma**2)).exp()
        else: self.kernel = kernel

    @property
    def input_size(self):
        return self._fdyn.in_channels

    @property
    def output_size(self):
        return self._fdyn.out_channels

    def gramm_matrix(self, x, y):
        """ Kernel matrix K, evaluated for each pair x, y"""

        rows = LazyTensor(x[:, :, None, :])
        cols = LazyTensor(y[:, None, :, :])

        return self.kernel(rows, cols)

    def velocity(self, xs, cs, ms):
        """Kernel product K(x,c)m"""

        # TODO: add batch support
        xs, cs, ms = xs[None], cs[None], ms[None]

        Vs = []
        for i in range(xs.shape[0]):
            x, c, m = xs[i:i+1], cs[i:i+1], ms[i:i+1]

            K = self.gramm_matrix(x, c)
            Km = (K @ m)
            Vs.append(Km)

        return torch.cat(Vs)

    def hamiltonian_eqs(self, x, m):
        """ Compute hamilton's equations of motion

        \frac{dx}{dt} = \fac{\partial H}{ \partial m}
        \frac{dm}{dt} = -\fac{\partial H}{ \partial x}
        :param x:
        :param m:
        :returns:

        """
        Km = self.velocity(x, x, m)
        H = 0.5*torch.sum(m * Km, (-2, -1))
        dx_t, dm_t = torch.autograd.grad(H.sum(), (m, x), create_graph=True)
        return torch.stack([dx_t, -dm_t])

    # def forward_(self, ds):
    #     # the neural network will handle all the dynamics here
    #     z_dot = self._fdyn(ds)

    #     # +0*x ensures that all is connected to autograd graph
    #     return z_dot

    # def parse_tensor_to_ds(self, x, t=None, condition=None):
    #     """Bundle dynamical state in state object."""

    #     ds = DynamicalState(state=x[:, -self.input_size:])

    #     if condition is not None:
    #         ds['condition'] = condition

    #     if t is not None:
    #         ds['t'] = t.repeat(x.shape[0], 1).type(x.dtype)
    #     return ds

    # def parse_ds_to_tensor(self, ds):
    #     """Convert dynamical state object to tensor."""

    #     # if hasattr(ds, 'conditions'):
    #     #     return torch.cat([ds.state, ds.conditions], -1)

    #     return ds.state

    # def update_ds(self, ds, x):
    #     new_ds = self.parse_tensor_to_ds(x)

    #     for key in new_ds.keys:
    #         ds[key] = new_ds[key]

    #     return ds
