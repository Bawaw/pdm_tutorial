#!/usr/bin/env python3

import torch
import torch.nn as nn
from regilib.core.dynamics.dynamical_state import DynamicalState

class InvertibleModule(nn.Module):
    def forward(self, ds_z, **kwargs):
        raise NotImplementedError()

    def inverse(self, ds_x, **kwargs):
        raise NotImplementedError()

    def freeze(self, freeze):
        for parameter in self.parameters():
            parameter.requires_grad = not freeze
        return self
