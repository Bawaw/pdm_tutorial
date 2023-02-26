#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.distributions as tdist

class MultivariateNormal(nn.Module):
    def __init__(self, loc, scale):
        super().__init__()
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    @property
    def dist(self):
        return tdist.MultivariateNormal(self.loc, self.scale)

    def log_prob(self, value):
        return self.dist.log_prob(value)

    def rsample(self, sample_shape=torch.Size([])):
        return self.dist.rsample(sample_shape)

    def sample(self, sample_shape=torch.Size([])):
        return self.dist.sample(sample_shape)

class VonMises(nn.Module):
    def __init__(self, loc, concentration):
        super().__init__()
        self.register_buffer("loc", loc)
        self.register_buffer("concentration", concentration)

    @property
    def dist(self):
        return torch.distributions.VonMises(
            loc=self.loc, concentration=self.concentration)

    def log_prob(self, value):
        print('Warning: using 1d von mises')
        return self.dist.log_prob(value).sum(-1)

    def rsample(self, sample_shape=torch.Size([])):
        # rsample is not supported
        breakpoint()
        return self.sample(sample_shape)

    def sample(self, sample_shape=torch.Size([])):
        return self.dist.sample(sample_shape)
