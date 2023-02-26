#!/usr/bin/env python3

import torch.nn as nn
from regilib.core.invertible_modules import InvertibleModule
from regilib.core.dynamics.dynamical_state import DynamicalState

class VAE(InvertibleModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, ds_z, ds_context=None, sample=False):
        ds_x = ds_z.clone()

        if ds_context is None:
            ds_x.state = self.decoder(ds_z.state, context=None)
        else:
            assert False
            #TODO: check this
            # log_px, ds_x.state = self.decoder(ds_z.state, context=ds_context.state)
            # log_qz, z_rec = self.encoder(ds_x.state, sample=sample)

            # # p(x) = p(z) + log p(x|z) - log q(z|x)
            # ds_x.add_or_create('log_prob', log_px - log_qz)

        return ds_x

    def inverse(self, ds_x, sample=False):
        ds_z = ds_x.clone()
        log_qz, ds_z.state = self.encoder(ds_x.state, sample=sample)
        log_px, x_reconstr = self.decoder(ds_z.state, context=ds_x.state)

        # p(x) = p(z) + log p(x|z) - log q(z|x)
        ds_z.add_or_create('log_prob', log_px - log_qz)

        return ds_z
