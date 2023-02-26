#!/usr/bin/env python3

import torch
from regilib.core.invertible_modules.invertible_module import InvertibleModule

class SphericalCoordProj(InvertibleModule):
    def forward(self, ds, **kwargs):
        theta, phi = ds['state'].T

        ds['state'] = torch.stack([
            torch.cos(phi)*torch.sin(theta),
            torch.sin(phi)*torch.sin(theta),
            torch.cos(theta)
        ], -1)

        return ds

    def inverse(self, ds, **kwargs):
        """ φ the zenith angle, 0 ≤ φ ≤ π, and θ the azimuth, -π ≤ θ ≤ π

        :param ds:
        :returns:

        """

        # unit-normalise x,y,z coordinates
        x, y, z = ds['state'].div(ds['state'].norm(dim=1, keepdim=True)).T

        ds['state'] = torch.stack([
            torch.arccos(z), torch.atan2(y, x)], -1)

        return ds
