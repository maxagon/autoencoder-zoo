import math

import torch
import torch.nn as nn

# Gaussian dropout as an information bottleneck layer
# http://bayesiandeeplearning.org/2021/papers/40.pdf
class GaussianDropout(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.std = math.sqrt(dropout / (1.0 - dropout))
    def forward(self, x):
        out = x
        noise = torch.normal(mean=1.0, std=self.std, size=x.shape).to(out.device)
        out = out * noise
        return out

class GaussianDistribution(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)

    def sample(self):
        std = torch.exp(0.5 * self.logvar)
        x = self.mean + std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self):
        shape_dim = len(self.mean.shape)
        sum_dim = []
        for i in range(1, shape_dim):
            sum_dim.append(i)
        var = torch.exp(self.logvar)
        result = torch.sum(torch.pow(self.mean, 2)  + var - 1.0 - self.logvar, dim=sum_dim)
        return result.mean(dim=0)