#!/usr/bin/env python3

import numpy as np
import torch
from torch.autograd import grad

def batch_jacobian(y, x):
    """ Compute jacobian in batch format

    input:
        y: torch.Size([B, d2])
        x: torch.Size([B, d1])

    output:
        jacobian dy/dx: torch.Size([B, d2, d1])
    """

    batch = y.shape[0]
    single_y_size = np.prod(y.shape[1:])
    y = y.view(batch, -1)
    vector = torch.ones(batch).to(y)

    # Compute Jacobian row by row.
    # dy_i / dx -> dy / dx
    # (B, D) -> (B, 1, D) -> (B, D, D)
    jac = [torch.autograd.grad(y[:, i], x,
                               grad_outputs=vector,
                               retain_graph=True,
                               create_graph=True)[0].view(batch, -1)
                for i in range(single_y_size)]
    jac = torch.stack(jac, dim=1)

    return jac

def autograd_trace(x_out, x_in, include_e_dz_dx=False):
    """Exact jacobian trace computed using autograd."""

    if include_e_dz_dx:
        J = batch_jacobian(x_out, x_in)
        trJ = torch.stack([j.trace() for j in J])
        e_dzdx = J.view(J.shape[0], -1)
    else:
        trJ = 0.
        for i in range(x_in.shape[1]):
            trJ += grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
        e_dz_dx=None

    return e_dzdx, trJ

def hutchinson_trace_estimator(x_out, x_in, noise):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    e_dzdx = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum('bi,bi->b', e_dzdx, noise)
    return e_dzdx, trJ
