#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from regilib.core.invertible_modules.invertible_module import InvertibleModule

class PadProj(InvertibleModule):
    def forward(self, ds, **kwargs):
        # deterministic pad: g: U → M ⊂ X
        #ds['state'] = F.pad(ds['state'], (0,1))

        ds['state'] = torch.cat([ds['state'][:,:1], torch.zeros([ds.state.shape[0], 1], device=ds.state.device), ds['state'][:, 1:]], -1)

        return ds

    def inverse(self, ds, **kwargs):
        # deterministic project: g⁻¹: X -> U
        #ds['state'] = ds['state'][:, :-1]

        ds['state'] = ds['state'][:, [0,2]]
        return ds
