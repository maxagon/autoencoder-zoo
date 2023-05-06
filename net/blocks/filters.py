import math

import torch
import torch.nn as nn
from einops import rearrange


def sinc(x):
    if x == 0.0:
        return 1.0
    x = math.pi * x
    return math.sin(x) / x


def lanczos(x):
    if -3.0 <= x and x < 3.0:
        return sinc(x) * sinc(x / 3.0)
    return 0.0


def create_lanczos_kernel5(sub_pixel_offset_x, sub_pixel_offset_y):
    offset = [-2.0, -1.0, 0.0, 1.0, 2.0]
    kernel = torch.zeros(size=[1, 1, 5, 5])
    sum = 0.0
    for x in range(5):
        for y in range(5):
            ox = offset[x] + sub_pixel_offset_x
            oy = offset[y] + sub_pixel_offset_y
            weight = lanczos(ox) * lanczos(oy)
            sum += weight
            kernel[0][0][x][y] = weight
    kernel = kernel
    return kernel


def create_lanczos_upscale_weights():
    o1 = 0.5
    o2 = -0.5
    w = create_lanczos_kernel5(o1, o1)
    w = torch.cat([w, create_lanczos_kernel5(o1, o2)], dim=0)
    w = torch.cat([w, create_lanczos_kernel5(o2, o1)], dim=0)
    w = torch.cat([w, create_lanczos_kernel5(o2, o2)], dim=0)
    return w


def conv_filter_1d(filter, normalize=True):
    size = len(filter)
    kernel = torch.zeros(size=[1, 1, size, size])
    sum = 0.0
    for x in range(size):
        for y in range(size):
            weight = filter[x] * filter[y]
            kernel[0][0][x][y] = weight
            sum += weight
    if normalize:
        for x in range(size):
            for y in range(size):
                kernel[0][0][x][y] = kernel[0][0][x][y] / sum
    return kernel


# https://en.wikipedia.org/wiki/Lanczos_resampling
# we can interpret each input channel as separate "batch" element so we can run single filter for each channel
# at least don't have to deal with cuda kernel
class LanczosSampler2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("filters", create_lanczos_upscale_weights())
        self.pad = nn.ReplicationPad2d(padding=(2, 2, 2, 2))

    def forward(self, x):
        bs = x.shape[0]
        out = x
        out = self.pad(out)
        out = rearrange(out, "b c x y -> (b c) x y")
        out = torch.unsqueeze(out, dim=1)
        out = nn.functional.conv2d(out, self.filters)
        out = rearrange(out, "(b c) (nx ny) x y -> b c (x nx) (y ny)", nx=2, ny=2, b=bs)
        return out


class LinearFilter2D(nn.Module):
    def __init__(self, filter, dim, normalize=True):
        super().__init__()
        kernel = conv_filter_1d(filter, normalize)
        kernel = kernel.tile((dim, 1, 1, 1))
        self.register_buffer("kernel", kernel)
        self.pad = nn.ReflectionPad2d(len(filter) // 2)
        self.dim = dim

    def forward(self, x):
        out = x
        out = self.pad(out)
        out = torch.nn.functional.conv2d(out, self.kernel, groups=self.dim)
        return out


def simulate_upscale(h_pos, w_pos, filter_size):
    tensor = torch.zeros(size=(1, 1, filter_size + 1, filter_size + 1))
    h_pos_start = h_pos * 2
    w_pos_start = w_pos * 2
    for h in range(2):
        for w in range(2):
            pos_h = h_pos_start + h
            pos_w = w_pos_start + w
            if pos_h < 0 or pos_h >= tensor.shape[2]:
                continue
            if pos_w < 0 or pos_w >= tensor.shape[3]:
                continue
            tensor[0, 0, pos_h, pos_w] = 1.0
    return tensor


def calculate_response(grid, filter):
    return torch.mean(torch.nn.functional.conv2d(grid, filter))


class AliasingNonlinearityFilter5(nn.Module):
    def __init__(self, dim, base_filter, normalize=True):
        super().__init__()
        assert len(base_filter) == 5
        base_filter_2d = conv_filter_1d(base_filter, normalize)
        filter_range_pixels = 3
        kernel = torch.zeros(size=(1, 1, filter_range_pixels, filter_range_pixels))
        for h in range(filter_range_pixels):
            for w in range(filter_range_pixels):
                kernel[0, 0, h, w] = calculate_response(
                    simulate_upscale(h, w, 5), base_filter_2d
                )
        kernel = kernel.tile((dim, 1, 1, 1))
        self.register_buffer("kernel", kernel)
        self.pad = nn.ReflectionPad2d(filter_range_pixels // 2)
        self.dim = dim

    def forward(self, x):
        out = x
        out = self.pad(out)
        out = torch.nn.functional.conv2d(out, self.kernel, groups=self.dim)
        return out
