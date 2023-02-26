#!/usr/bin/env python3

import torch
from typing import Union, List
from regilib.core.invertible_modules import InvertibleModule
from regilib.core.dynamics.dynamical_state import DynamicalState

class NormalisingFlow(InvertibleModule):
    def __init__(self, base_distribution=None):
        super().__init__()
        self.base_distribution = base_distribution

    def sample_prior(self, n_samples:int) -> torch.Tensor:
        """Sample z from prior distribution p(z) """

        samples = self.base_distribution.rsample([n_samples])
        return samples

    def sample_posterior(
            self, n_samples:int, condition:Union[
                torch.Tensor, None]=None, **kwargs) -> DynamicalState:
        """Sample x from posterior distribution p(x) """

        state_samples = self.sample_prior(n_samples)
        state_samples.requires_grad = True

        ds = DynamicalState(state=state_samples)
        if condition is not None: ds['condition'] = condition

        return self.forward(ds, **kwargs)

    def forward(self, ds: DynamicalState) -> DynamicalState:
        ds.add_or_create(
            'log_prob', self.base_distribution.log_prob(ds['state']))

        return ds

    def inverse(self, ds: DynamicalState) -> DynamicalState:
        ds.add_or_create(
            'log_prob', self.base_distribution.log_prob(ds['state']))

        return ds


class SequentialNormalisingFlow(NormalisingFlow):
    def __init__(self, base_distribution, transforms: List[InvertibleModule]):
        super().__init__(base_distribution)
        self.transforms = transforms

    def forward(self, ds: DynamicalState) -> DynamicalState:
        ds.add_or_create(
            'log_prob', self.base_distribution.log_prob(ds['state']))

        for t in self.transforms:
            ds = t.forward(ds)

        return ds

    def inverse(self, ds: DynamicalState) -> DynamicalState:
        for t in reversed(self.transforms):
            ds = t.inverse(ds)

        ds['log_prob'] += self.base_distribution.log_prob(ds['state'])

        return ds
