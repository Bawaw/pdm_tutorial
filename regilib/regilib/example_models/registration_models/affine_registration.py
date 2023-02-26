#!/usr/bin/env python3

import torch

class AffineRegistration:
    """Affine registration between point-clouds in correspondence."""
    def _estimate_transform(self, fixed, moving):
        # B = AX, returns X
        affine_transformation_matrix = torch.linalg.lstsq(
            input=torch.cat([moving, torch.ones(moving.shape[0], 1)], -1),
            b=torch.cat([fixed, torch.ones(fixed.shape[0], 1)], -1)
        )[0]

        return affine_transformation_matrix

    def _transform(self, moving, affine_matrix):
        return (torch.cat([moving, torch.ones(moving.shape[0], 1)], -1) @ affine_matrix)[:, :-1]

    def __call__(self, fixed, moving, return_transform=False):
        f_star = moving.clone()
        affine_transformation_matrix = self._estimate_transform(fixed.pos, moving.pos)
        f_star.pos = self._transform(moving.pos, affine_transformation_matrix)

        if return_transform: return f_star, affine_transformation_matrix
        return f_star
