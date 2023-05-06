from enum import Enum

import torch
import torch.nn as nn

from einops import repeat

from . import init
from . import nonlinear as nl
from . import filters
from . import feedforward


class InterpolationMode(Enum):
    Nearest = "nearest"
    Linear = "linear"
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


# https://github.com/lucidrains/lightweight-gan
# https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
# named SP-conv in the paper, but basically a pixel unshuffle
class PixelShuffleDownsample(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.to_out = feedforward.Conv2DBlock(
            in_dim=in_dim * 4, out_dim=out_dim, kernel_rad=0
        )

    def forward(self, x):
        out = x
        out = nn.functional.pixel_unshuffle(out, 2)
        out = self.to_out(out)
        return out


# https://github.com/lucidrains/lightweight-gan
class PixelShuffleUpscale(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        assert in_dim % 4 == 0
        self.to_out = feedforward.Conv2DBlock(
            in_dim=in_dim, out_dim=out_dim * 4, kernel_rad=0
        )
        self.init_conv_(self.to_out)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        out = x
        out = self.to_out(out)
        out = nn.functional.silu(out)
        out = nn.functional.pixel_shuffle(out, 2)
        return out


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
