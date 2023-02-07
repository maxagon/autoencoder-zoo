from enum import Enum
import math

import torch
import torch.nn as nn

class InitMode(Enum):
    FanIn = 1,
    FanOut = 2,
    FanAvr = 3

class InitDisribution(Enum):
    Normal = 1,
    Uniform = 2

class InitParams():
    def __init__(self, mode : InitMode = InitMode.FanAvr, distribution : InitDisribution  = InitDisribution.Normal, gain = 1.0, scale = 1.0):
        self.mode = mode
        self.distribution = distribution
        self.gain = gain
        self.scale = scale

@torch.no_grad()
def variance_scaling_init(
    tensor : torch.Tensor,
    init_params : InitParams
):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    scale = init_params.scale

    if init_params.mode == InitMode.FanIn:
        scale /= fan_in
    elif init_params.mode == InitMode.FanOut:
        scale /= fan_out
    elif init_params.mode == InitMode.FanAvr:
        scale /= (fan_in + fan_out) / 2
    else:
        assert 0, "Unknown init mode: {}".format(init_params.mode)

    if init_params.distribution == InitDisribution.Normal:
        std = math.sqrt(scale)
        return init_params.gain * tensor.normal_(0, std)
    elif init_params.distribution == InitDisribution.Uniform:
        bound = math.sqrt(3 * scale)
        return init_params.gain * tensor.uniform_(-bound, bound)
    else:
        assert 0, "Unknown ditribution: {}".format(init_params.distribution)
    return torch.zeros_like(tensor)
