from typing import Optional
import math

import torch
import torch.nn as nn
from einops import rearrange

from . import init
from . import nonlinear as nl


class Linear(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        dropout=0.0,
        nonlinearity: Optional[nl.Nonlinearity] = None,
        init_params: Optional[init.InitParams] = None,
    ):
        super().__init__()

        # dropout
        self.dropout = None
        if dropout != 0.0:
            self.dropout = nn.Dropout2d(dropout)

        model = [nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)]

        # init
        if bias:
            nn.init.zeros_(model[-1].bias)
        if init_params == None:
            init_params = init.InitParams()
            if nonlinearity != None:
                init_params.gain = nonlinearity.calc_gain()
        init.variance_scaling_init(model[-1].weight, init_params)

        # nonlinearity
        if nonlinearity != None:
            model += [nonlinearity]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x
        if self.dropout != None:
            out = self.dropout(out)
        out = self.model(out)
        return out


class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_rad,
        pad_type="none",
        bias=True,
        dropout=0.0,
        stride=1,
        nonlinearity: Optional[nl.Nonlinearity] = None,
        init_params: Optional[init.InitParams] = None,
        groups=1,
    ):
        super().__init__()

        # dropout
        self.dropout = None
        if dropout != 0.0:
            self.dropout = nn.Dropout2d(dropout)

        model = []

        # pad
        if pad_type == "replicate":
            model += [nn.ReplicationPad2d(kernel_rad)]
        elif pad_type == "reflect":
            model += [nn.ReflectionPad2d(kernel_rad)]
        elif pad_type == "zero":
            model += [nn.ZeroPad2d(kernel_rad)]
        elif pad_type != "none":
            assert 0, "Wrong padding type: {}".format(pad_type)

        # conv
        kernel_size = kernel_rad * 2 + 1
        model += [
            nn.Conv2d(in_dim, out_dim, kernel_size, stride, bias=bias, groups=groups)
        ]

        # init
        if bias:
            nn.init.zeros_(model[-1].bias)
        if init_params == None:
            init_params = init.InitParams()
            if nonlinearity != None:
                init_params.gain = nonlinearity.calc_gain()
        init.variance_scaling_init(model[-1].weight, init_params)

        # nonlinearity
        if nonlinearity != None:
            model += [nonlinearity]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x
        if self.dropout != None:
            out = self.dropout(out)
        out = self.model(out)
        return out


class DepthwiseConv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_rad, pad_type="none"):
        super().__init__()
        self.blocks = nn.Sequential(
            Conv2DBlock(
                in_dim=in_dim,
                out_dim=in_dim,
                kernel_rad=kernel_rad,
                pad_type=pad_type,
                groups=in_dim,
            ),
            Conv2DBlock(in_dim=in_dim, out_dim=out_dim, kernel_rad=0),
        )

    def forward(self, x):
        return self.blocks(x)


class SelfAttentionCNN(nn.Module):
    def __init__(self, in_dim, attend_dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.scale = math.sqrt(attend_dim)

        qkv_dim = attend_dim * heads
        self.q = Conv2DBlock(in_dim=in_dim, out_dim=qkv_dim, kernel_rad=0, bias=False)
        self.k = Conv2DBlock(in_dim=in_dim, out_dim=qkv_dim, kernel_rad=0, bias=False)
        self.v = Conv2DBlock(in_dim=in_dim, out_dim=qkv_dim, kernel_rad=0, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = Conv2DBlock(
            in_dim=qkv_dim, out_dim=in_dim, kernel_rad=0, bias=False
        )

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = rearrange(q, "b (h c) y x -> b h (y x) c", h=self.heads)
        k = rearrange(k, "b (h c) y x -> b h (y x) c", h=self.heads)
        v = rearrange(v, "b (h c) y x -> b h (y x) c", h=self.heads)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h (y x) c -> b (h c) y x", x=w, y=h)
        out = self.to_out(out)

        return out
