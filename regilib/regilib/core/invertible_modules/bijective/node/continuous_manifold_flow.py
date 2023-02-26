#!/usr/bin/env python3

import torch
from regilib.core.numerics.derivatives import batch_jacobian
from .continuous_ambient_flow import ContinuousAmbientFlow
from .continuous_flow import ContinuousFlow

class ContinuousManifoldFlow(ContinuousAmbientFlow):
    def __init__(self, chart, **kwargs):
        super().__init__(**kwargs)
        self.chart = chart

    def forward(self, ds_z, skip_jacobian_det=False, **kwargs):
        """Generative Flow"""

        state_in = ds_z.state

        # log p(z) + log det J_h(z)
        ds_u = self.chart.forward(ds_z)
        ds_x = super().forward(ds_u, **kwargs)

        state_out = ds_x.state

        if not skip_jacobian_det:
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            Jf = batch_jacobian(state_out, state_in)
            # -1/2 log det [J_g(u)ᵀJ_g(u)]
            ds_x.add_or_create(
                'log_prob', -0.5 * torch.bmm(
                    torch.transpose(Jf, -2, -1), Jf).slogdet()[1]
            )

        return ds_x

    def inverse(self, ds_x, skip_jacobian_det=False, **kwargs):
        """Normalising flow"""

        state_in = ds_x.state

        ds_u = super().inverse(ds_x, **kwargs)
        ds_z = self.chart.inverse(ds_u)

        state_out = ds_z.state

        if not skip_jacobian_det:
            Jf_inv = batch_jacobian(state_out, state_in)

            # A=Jg ⇒ Jg⁻¹ = A⁻¹ (inverse function theorem)
            # (A⁻¹)(A⁻¹)ᵀ ⇔ (A⁻¹)(Aᵀ)⁻¹ ⇔ (AᵀA)⁻¹ ⇔ (JgᵀJg)⁻¹
            # ⇒ 1/2 log det [J_g⁻¹(x)J_g⁻¹(x)ᵀ]
            ds_z.add_or_create(
                'log_prob', 0.5 * torch.bmm(
                    Jf_inv, torch.transpose(Jf_inv, -2, -1)).slogdet()[1]
            )

        return ds_z
