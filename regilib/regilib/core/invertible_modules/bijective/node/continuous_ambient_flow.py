#!/usr/bin/env python3

from .continuous_flow import ContinuousFlow
from regilib.core.dynamics.dynamical_state import DynamicalState

class ContinuousAmbientFlow(ContinuousFlow):
    def forward(self, ds_z : DynamicalState, **kwargs):
        """Generative Flow"""

        # z(t₀) = \int_1^0 f(z(t), t)
        # in torchdyn this is set to
        # * \int_{-1}^0 - f(z(t), t) # t_span = -t_span; f = -f
        # * \int_{-1}^0 - f(z(-t), -t) # t = -t
        # * \int_{1}^0 - f(z(t), t) # t = -t
        # see torchdyn.numerics.odeint
        ds_x = super().forward(ds_z, **kwargs)

        # log det J_h(z) = \int_1^0 - Tr(∂f/∂z(t))
        # in torchdyn this is set to log det J_h(z) = + \int_0^1 Tr(Jf)
        # see torchdyn.numerics.odeint
        if hasattr(ds_z, 'l'):
            ds_x.add_or_create('log_prob', ds_z['l'][:,0])
            ds_x.remove_key('l')

        return ds_x

    def inverse(self, ds_x : DynamicalState, **kwargs):
        """Normalising flow"""

        # z(t₁) = \int_0^1 f(z(t), t)
        ds_z = super().inverse(ds_x, **kwargs)

        # log det J_h^{-1}(z) = + \int_1^0 -Tr(∂f/∂z(t))
        # <=> - \int_0^1 - Tr(∂f/∂z(t))
        if hasattr(ds_z, 'l'):
            ds_z.add_or_create('log_prob', -ds_z['l'][:,0])
            ds_z.remove_key('l')

        return ds_z
