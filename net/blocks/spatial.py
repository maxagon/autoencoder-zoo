from enum import Enum

import torch.nn as nn

from . import init
from . import nonlinear as nl
from . import filters


class InterpolationMode(Enum):
    Nearest = ("nearest",)
    Linear = ("linear",)
    Bilinear = "bilinear"


class Upscale(nn.Module):
    def __init__(self, mode: InterpolationMode = InterpolationMode.Nearest):
        super().__init__()
        self.mode = mode.value

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2.0, mode=self.mode)


class Downscale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=2, stride=2)


class ConvolutionDownscale(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )
        init.variance_scaling_init(
            self.model.weight, init_params=init.InitParams(mode=init.InitMode.FanOut)
        )

    def forward(self, x):
        return self.model(x)


class ConvolutionUpscale(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.ConvTranspose2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )
        init.variance_scaling_init(
            self.model.weight,
            init_params=init.InitParams(
                mode=init.InitMode.FanAvr,
                distribution=init.InitDisribution.Uniform,
                gain=1.0,
                scale=1.0,
            ),
        )

    def forward(self, x):
        return self.model(x)


class LanczosUpscale(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        model = []
        model.append(filters.LanczosSampler2D())
        if in_dim != out_dim:
            model.append(
                nn.Conv2d(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            )
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
