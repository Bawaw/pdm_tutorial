#!/usr/bin/env python3

import torch
from regilib.core.numerics.derivatives import batch_jacobian
from regilib.core.invertible_modules.invertible_module import InvertibleModule

class SwissRollCoordProj(InvertibleModule):

    def forward(self, ds, **kwargs):
        state_in = ds['state']
        psi, gamma = state_in.T
        x, y, z = psi*torch.cos(psi), psi*torch.sin(psi), gamma
        state_out = torch.stack([x, y, z], -1)

        # p(x) -= 1/2 log det [J_f^T(u)J_f(u)]
        if hasattr(ds, 'log_prob'):
            Jf = batch_jacobian(state_out, state_in)
            ds['log_prob'] -= 0.5 * torch.bmm(
                torch.transpose(Jf, -2, -1), Jf).slogdet()[1]
        ds['state'] = state_out

        return ds

    def inverse(self, ds, **kwargs):
        """ takes torus uv coordinates and projects them to 3D cartesian coordinates

        :param ds:
        :returns:

        """

        raise NotImplemented(
            "Inverse of pinched torus projection is not yet implemented.")

        return ds
