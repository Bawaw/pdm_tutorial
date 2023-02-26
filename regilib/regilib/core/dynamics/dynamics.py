#!/usr/bin/env python3

import torch
import torch.nn as nn
from regilib.core.distributions import MultivariateNormal
from regilib.core.dynamics.dynamical_state import DynamicalState
from regilib.core.trace_estimators import (autograd_trace,
                                           hutchinson_trace_estimator)
from copy import copy

class Dynamics(nn.Module):
    def __init__(self, include_time=True):
        super().__init__()
        self._include_time = include_time
        self.params={}

    def set_params(self, **kwargs):
        self.params = kwargs

    def forward(self, t, x):
        params = copy(self.params)
        cond = params.pop('condition', None)
        if isinstance(x, torch.Tensor):
            ds = self.parse_tensor_to_ds(x, t=t, condition=cond)
        else:
            raise ValueError(
                "{}, not valid for Dynamics.forward call".format(type(x)))

        return self.forward_(ds, **params)


class StateDynamics(Dynamics):
    """Just state in dynamics"""

    def __init__(self, fdyn, **kwargs):
        super().__init__(**kwargs)
        self._fdyn = fdyn

    @property
    def input_size(self):
        return self._fdyn.in_channels

    @property
    def output_size(self):
        return self._fdyn.out_channels

    def parse_tensor_to_ds(self, x, t=None, condition=None):
        """Bundle dynamical state in state object."""

        ds = DynamicalState(state=x[:, -self.input_size:])

        if condition is not None:
            ds['condition'] = condition

        if t is not None:
            ds['t'] = t.repeat(x.shape[0], 1).type(x.dtype)
        return ds

    def parse_ds_to_tensor(self, ds):
        """Convert dynamical state object to tensor."""

        # if hasattr(ds, 'conditions'):
        #     return torch.cat([ds.state, ds.conditions], -1)

        return ds.state

    def update_ds(self, ds, x):
        new_ds = self.parse_tensor_to_ds(x)

        for key in new_ds.keys:
            ds[key] = new_ds[key]

        return ds

    def forward_(self, ds):
        # the neural network will handle all the dynamics here
        z_dot = self._fdyn(ds)

        # +0*x ensures that all is connected to autograd graph
        return z_dot


class TraceDynamics(StateDynamics):
    """Includes Jacobian trace in dynamics"""

    def parse_tensor_to_ds(self, x, t=None, condition=None):
        """Bundle dynamical state in state object with
        trace, state and time

        """

        ds = super().parse_tensor_to_ds(x, t, condition)
        ds['l'] = x[:, :1]
        self.estimate_trace = True
        return ds

    def parse_ds_to_tensor(self, ds):
        """Convert dynamical state object to tensor."""
        try:
            return torch.cat([ds.l, super().parse_ds_to_tensor(ds)], -1)
        except:
            print(
                "Failed to parse {} in {}, did you use the correct DynamicalState?".format(
                    ds, type(self)))
            raise

    def compute_trace(self, z_dot, z, estimate_trace, include_e_dzdx=False):
        # compute/approximate trace

        if estimate_trace:
            e_dzdx, divergence = hutchinson_trace_estimator(
                z_dot, z, noise=self.noise)
        else:
            e_dzdx, divergence = autograd_trace(z_dot, z, include_e_dzdx)

        if include_e_dzdx: return -divergence[:, None], e_dzdx
        else: return -divergence[:, None]

    def forward_(self, ds, estimate_trace=False):
        ds['state'] = ds['state'].requires_grad_(True)

        # enable grad for trace computation, this might not be the case when using adjoint
        with torch.set_grad_enabled(True):
            z_dot = self._fdyn(ds)

            # compute/approximate trace
            neg_l_dot = self.compute_trace(z_dot, ds['state'], estimate_trace)

        return torch.cat([neg_l_dot, z_dot], -1)


class RegularisedDynamics(TraceDynamics):
    """Includes frobenius and energy in dynamics"""

    def parse_tensor_to_ds(self, x, t=None, condition=None):
        """Split input in trace, kinetic energy, frobenius jac, state and time."""

        ds = super().parse_tensor_to_ds(x, t, condition)
        ds['e'] = x[:, 1:2]
        ds['n'] = x[:, 2:3]
        return ds

    def parse_ds_to_tensor(self, ds):
        """Convert dynamical state object to tensor."""
        try:
            ds_tensor = super().parse_ds_to_tensor(ds)
            ds_tensor = torch.cat([ds_tensor[:, :1], ds.e, ds.n, ds_tensor[:, 1:]], -1)
            return ds_tensor
        except:
            print(
                "Failed to parse {} in {}, did you use the correct DynamicalState?".format(
                    ds, type(self)))
            raise

    def forward_(self, ds, estimate_trace=False):
        ds['state'] = ds['state'].requires_grad_(True)

        # enable grad for trace computation, this might not be the case when using adjoint
        with torch.set_grad_enabled(True):
            z_dot = self._fdyn(ds)

            # compute/approximate trace
            neg_l_dot, e_dzdx = self.compute_trace(z_dot, ds['state'], estimate_trace, include_e_dzdx=True)

            # kinetic energy
            e_dot = torch.sum(z_dot**2, 1)[:,None]
            # frobenius jacobian
            n_dot = torch.sum(e_dzdx**2, 1)[:, None]

        return torch.cat([neg_l_dot, e_dot, n_dot, z_dot], -1)
