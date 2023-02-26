#!/usr/bin/env python3

from regilib.core.numerics.derivatives import batch_jacobian
from regilib.core.invertible_modules.invertible_module import InvertibleModule

class AffineTransform(InvertibleModule):

    def __init__(self, loc, scale):
        super().__init__()
        self.loc, self.scale = loc, scale

    def forward(self, ds, **kwargs):
        state_in = ds['state']
        state_out = self.loc + self.scale * state_in

        ds['state'] = state_out

        # p(x) -= |log det Jf(z)| TODO: compute analytically
        if hasattr(ds, 'log_prob'):
            ds['log_prob'] -= batch_jacobian(state_out, state_in).slogdet()[1]

        return ds

    def inverse(self, ds, **kwargs):
        state_in = ds['state']
        state_out = (state_in - self.loc) / self.scale

        ds['state'] = state_out
        # p(x) += |log det Jf(z)| TODO: compute analytically

        if hasattr(ds, 'log_prob'):
            ds['log_prob'] += batch_jacobian(state_out, state_in).slogdet()[1]
        return ds
