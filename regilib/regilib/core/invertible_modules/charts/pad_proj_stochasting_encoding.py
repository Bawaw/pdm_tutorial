#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from regilib.core.invertible_modules.invertible_module import InvertibleModule

class PadProjStochasticEncoder(InvertibleModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = PadEnc.Encoder(3, 2)

    class Encoder(nn.Module):
        """ Models the encoding process as a stochastic function. """

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels

            # expected format: N x (C * L)
            self.fc1 = nn.Linear(in_channels, 128)
            self.fc2 = nn.Linear(128, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc4 = nn.Linear(256, 128)
            self.fc5 = nn.Linear(128, out_channels)

        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            x = self.fc5(x)
            return x

    def forward(self, ds, **kwargs):
        # deterministic pad: g: U → M ⊂ X
        ds['state'] = F.pad(ds['state'], (0,1))
        return ds

    def inverse(self, ds, **kwargs):
        # stoachastic project: g⁻¹: X -> U
        ds['state'] = self.encoder(ds['state'])
        return ds
