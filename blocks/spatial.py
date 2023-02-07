import torch
import torch.nn as nn
from einops import rearrange

from filters import LanczosSampler2D

class Upscale(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

class Downscale(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

class ConvolutionDownscale(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=2, stride=2, padding=0, bias=False)
    def forward(self, x):
        return self.model(x)

class ConvolutionUpscale(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.model = nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=2, stride=2, padding=0, bias=False)
    def forward(self, x):
        return self.model(x)

class LanczosUpscale(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        model = []
        model.append(LanczosSampler2D(in_dim))
        if in_dim != out_dim:
            model.append(nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*model)
    def forward(self, x):
        return self.model(x)
