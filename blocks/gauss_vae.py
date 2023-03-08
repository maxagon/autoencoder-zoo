import math

import torch
import torch.nn as nn

class GaussianDropoutScheduler():
    def __init__(self, start_dropout = 0.2, end_dropount = 0.0, warmup_steps = 20000, exp_decay = 10000):
        self.blocks = []
        self.counter = 0
        self.start_dropout = start_dropout
        self.end_dropount = end_dropount
        self.warmup_steps = warmup_steps
        self.exp_decay = exp_decay

    def make(self):
        dropout_block = GaussianDropout(dropout=self.start_dropout)
        self.blocks.append(dropout_block)
        return dropout_block

    def step(self):
        self.counter = self.counter + 1
        if self.counter > self.warmup_steps:
            step = self.counter - self.warmup_steps
            current_dropout = self.end_dropout + (self.start_dropout - self.end_dropout) * math.exp(-step / self.exp_decay)
            self.update(current_dropout)

    def update(self, dropout):
        for b in self.blocks:
            b.set_dropout(dropout)

# Gaussian dropout as an information bottleneck layer
# http://bayesiandeeplearning.org/2021/papers/40.pdf
class GaussianDropout(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.set_dropout(dropout=dropout)

    def set_dropout(self, dropout):
        self.std = math.sqrt(dropout / (1.0 - dropout))

    def forward(self, x):
        out = x
        noise = torch.normal(mean=1.0, std=self.std, size=x.shape, device=out.device)
        out = out * noise
        return out

class GaussianDistribution(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)

    def sample(self):
        std = torch.exp(0.5 * self.logvar)
        x = self.mean + std * torch.randn(self.mean.shape, device=self.parameters.device)
        return x

    def kl(self):
        shape_dim = len(self.mean.shape)
        sum_dim = []
        for i in range(1, shape_dim):
            sum_dim.append(i)
        var = torch.exp(self.logvar)
        result = torch.sum(torch.pow(self.mean, 2)  + var - 1.0 - self.logvar, dim=sum_dim)
        return result.mean(dim=0)