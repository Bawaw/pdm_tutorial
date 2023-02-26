#!/usr/bin/env python3

import torch
from regilib.core.dynamics.dynamical_state import DynamicalState
from regilib.core.invertible_modules import InvertibleModule
from torchdyn.numerics.odeint import odeint, str_to_solver


class PointCloudDeformationLayer(InvertibleModule):
    DEFAULT_T0 = 0
    DEFAULT_T1 = 1

    def __init__(self, default_n_steps, solver='rk4'):
        super().__init__()
        self.default_n_steps = default_n_steps
        self.solver = solver if type(solver) != str else str_to_solver(solver)

    def integrate(self, ds, v_t, t0=None, t1=None, steps=None):
        t_span = torch.linspace(t0, t1, steps)

        Xs_t, Xs = odeint(
            v_t, ds.state, t_span, solver=self.solver
        )

        return Xs

    def forward(self, ds_z, v_t, t0=None, t1=None, steps=None,
                include_trajectory=False):
        ds_x = ds_z.clone() if isinstance(
            ds_z, DynamicalState) else DynamicalState(state=ds_z)
        t0 = PointCloudDeformationLayer.DEFAULT_T0 if t0 is None else t0
        t1 = PointCloudDeformationLayer.DEFAULT_T1 if t1 is None else t1
        steps = self.default_n_steps if steps is None else steps

        # t: 0 -> 1
        xt = self.integrate(ds_x, v_t, t0, t1, steps)
        ds_x.state = xt[-1]
        if include_trajectory: ds_x.append_or_create('trajectory', xt)

        return ds_x

    def inverse(self, ds_x, v_t, t0=None, t1=None, steps=None,
                include_trajectory=False):
        ds_z = ds_x.clone() if isinstance(
            ds_x, DynamicalState) else DynamicalState(state=ds_x)
        t0 = PointCloudDeformationLayer.DEFAULT_T1 if t0 is None else t0
        t1 = PointCloudDeformationLayer.DEFAULT_T0 if t1 is None else t1
        steps = self.default_n_steps if steps is None else steps

        # t: 1 -> 0
        zt = self.integrate(ds_x, v_t, t0, t1, steps)
        ds_z.state = zt[-1]
        if include_trajectory: ds_z.append_or_create('trajectory', zt)

        return ds_z
