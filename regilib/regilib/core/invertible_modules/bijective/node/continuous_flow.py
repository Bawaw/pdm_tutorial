#!/usr/bin/env python3
from typing import Callable, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from regilib.core.distributions import MultivariateNormal
from regilib.core.dynamics.dynamical_state import DynamicalState
from torch import Tensor
from torchdyn.core.problems import ODEProblem
from torchdyn.core.utils import standardize_vf_call_signature
from torchdyn.numerics import odeint

from regilib.core.invertible_modules import InvertibleModule

class NeuralODE(ODEProblem):
    def __init__(self, vector_field:Union[Callable, nn.Module], solver:Union[str, nn.Module]='tsit5', order:int=1,
                atol:float=1e-3, rtol:float=1e-3, sensitivity='autograd', solver_adjoint:Union[str, nn.Module, None] = None,
                atol_adjoint:float=1e-4, rtol_adjoint:float=1e-4, interpolator:Union[str, Callable, None]=None, \
                integral_loss:Union[Callable, None]=None, seminorm:bool=False, return_t_eval:bool=True):
        super().__init__(vector_field=standardize_vf_call_signature(vector_field, order, defunc_wrap=True), order=order, sensitivity=sensitivity,
                         solver=solver, atol=atol, rtol=rtol, solver_adjoint=solver_adjoint, atol_adjoint=atol_adjoint, rtol_adjoint=rtol_adjoint,
                         seminorm=seminorm, interpolator=interpolator, integral_loss=integral_loss)

        self.u, self.controlled, self.t_span = None, False, None # data-control conditioning
        self.return_t_eval = return_t_eval
        if integral_loss is not None: self.vf.integral_loss = integral_loss
        self.vf.sensitivity = sensitivity
        self._noise_dist = MultivariateNormal(
            torch.zeros(self.vf.vf.input_size), torch.eye(self.vf.vf.input_size))

    def _prep_integration(self, x:Tensor, t_span:Tensor, estimate_trace) -> Tensor:
        "Performs generic checks before integration. Assigns data control inputs and augments state for CNFs"

        # loss dimension detection routine; for CNF div propagation and integral losses w/ autograd
        excess_dims = 0
        if (not self.integral_loss is None) and self.sensitivity == 'autograd':
            excess_dims += 1

        # handle aux. operations required for some jacobian trace CNF estimators e.g Hutchinson's
        # as well as datasets-control set to DataControl module
        for _, module in self.vf.named_modules():
            if estimate_trace:
                module.noise = self._noise_dist.sample((x.shape[0],))
                excess_dims += 1

            # data-control set routine. Is performed once at the beginning of odeint since the control is fixed to IC
            if hasattr(module, 'u'):
                self.controlled = True
                module.u = x[:, excess_dims:].detach()
        return x, t_span


    def forward(self, ds:Tensor, t_span:Tensor, estimate_trace):
        x, t_span = self._prep_integration(ds, t_span, estimate_trace)
        t_eval, sol =  super().forward(x, t_span)
        if self.return_t_eval: return t_eval, sol
        else: return sol

    def trajectory(self, x:torch.Tensor, t_span:Tensor, estimate_trace):
        x, t_span = self._prep_integration(x, t_span, estimate_trace)
        _, sol = odeint(self.vf, x, t_span, solver=self.solver, atol=self.atol, rtol=self.rtol)
        return sol

class ContinuousFlow(InvertibleModule):
    DEFAULT_T0 = 0
    DEFAULT_T1 = 1

    def __init__(
            self, dynamics, default_n_steps=5, solver='dopri5',
            sensitivity='adjoint', **kwargs):
        super().__init__()
        self.default_n_steps, self.dynamics = default_n_steps, dynamics

        self.nde = NeuralODE(
            vector_field=self.dynamics, solver=solver, sensitivity=sensitivity, **kwargs
        )

    def integrate(self, ds, t0, t1, steps, include_trajectory=False,
                  include_time_steps=False,
                  estimate_trace: Union[bool, None] = None, **kwargs):
        ds_tensor = self.dynamics.parse_ds_to_tensor(ds)

        # TODO: is there not a better way to pass this?
        if hasattr(ds, 'condition'): kwargs['condition'] = ds['condition']

        # if estimate trace parameter is given, save it to the parameters
        if estimate_trace is not None:
            kwargs['estimate_trace'] = estimate_trace
        self.dynamics.set_params(**kwargs)
        time_steps, x = self.nde(ds_tensor, torch.linspace(t0, t1, steps), estimate_trace)
        ds = self.dynamics.update_ds(ds, x[-1])

        if include_trajectory: ds.append_or_create('trajectory', x)
        if include_time_steps: ds.append_or_create('time_steps', time_steps)

        return ds

    def forward(self, ds_z, t0=None, t1=None, steps=None, **kwargs):
        t0 = ContinuousFlow.DEFAULT_T1 if t0 is None else t0
        t1 = ContinuousFlow.DEFAULT_T0 if t1 is None else t1
        steps = self.default_n_steps if steps is None else steps

        # t: 1 -> 0
        return self.integrate(ds_z, t0=t0, t1=t1, steps=steps, **kwargs)

    def inverse(self, ds_x, t0=None, t1=None, steps=None, **kwargs):
        t0 = ContinuousFlow.DEFAULT_T0 if t0 is None else t0
        t1 = ContinuousFlow.DEFAULT_T1 if t1 is None else t1
        steps = self.default_n_steps if steps is None else steps

        # t: 0 -> 1
        return self.integrate(ds_x, t0=t0, t1=t1, steps=steps, **kwargs)
