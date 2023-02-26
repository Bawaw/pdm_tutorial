#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from regilib.core.invertible_modules.invertible_module import InvertibleModule


class SphericalStereographicProjection(InvertibleModule):
    def forward(self, ds, **kwargs):
        phi, Theta = ds['state'].T

        # unit-sphere -> polar
        r = torch.sin(phi) / (1 - torch.cos(phi))

        # polar -> xy
        ds['state'] = torch.stack([
            r*torch.cos(Theta), r*torch.sin(Theta)], -1)

        # deterministic pad: g: U → M ⊂ X
        ds['state'] = F.pad(ds['state'], (0, 1))
        return ds

    def inverse(self, ds, **kwargs):
        """ φ the zenith angle, 0 ≤ φ ≤ π, and θ the azimuth, -π ≤ θ ≤ π

        :param ds:
        :returns:

        """

        # deterministic project: g⁻¹: X -> U
        ds['state'] = ds['state'][:, :-1]

        # xy -> polar coordinates
        r = torch.norm(ds['state'], p=2, dim=1)
        Theta = torch.atan2(ds['state'][:, 1], ds['state'][:, 0])

        # polar -> unit-sphere
        phi = 2*torch.arctan(1/r)
        ds['state'] = torch.stack([phi, Theta], -1)

        return ds
