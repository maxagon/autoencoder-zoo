from typing import Optional

import torch
import torch.nn as nn

import init

class Linear(nn.Module):
    DEFAULT_INIT = init.InitParams()
    def __init__(self, in_dim, out_dim, bias=True, dropout=0.0, init_params : Optional[init.InitParams] = None):
        super().__init__()

        # dropout
        self.dropout = None
        if dropout != 0.0:
            self.dropout = nn.Dropout2d(dropout)

        self.layer = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)

        # init
        if bias:
            nn.init.zeros_(self.layer.bias)
        if init_params == None:
            init_params = Linear.DEFAULT_INIT
        self.layer.weight = init_params

    def forward(self, x):
        out = x
        if self.dropout != None:
            out = self.dropout(out)
        out = self.layer(out)

class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_rad, pad_type='none', norm='none', nonlinearity='none', bias=False, dropout=0.0, stride=1, img_size=None, init_params : Optional[init.InitParams] = None):
        super().__init__()

        # dropout
        self.dropout = None
        if dropout != 0.0:
            self.dropout = nn.Dropout2d(dropout)

        model = []

        # pad
        if (norm != 'none'):
            assert(not bias)
        if pad_type == 'replicate':
            model += [nn.ReplicationPad2d(kernel_rad)]
        elif pad_type == 'reflect':
            model += [nn.ReflectionPad2d(kernel_rad)]
        elif pad_type == 'zero':
            model += [nn.ZeroPad2d(kernel_rad)]
        elif pad_type != 'none':
            assert 0, "Wrong padding type: {}".format(pad_type)

        # conv
        kernel_size = kernel_rad * 2 + 1
        model += [nn.Conv2d(in_dim, out_dim, kernel_size, stride, bias=bias, groups=1)]

        # init
        if bias:
            nn.init.zeros_(self.model[-1].bias)
        if init_params == None:
            init_params = init.InitParams()
            model[-1].weight = init_params

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x
        if self.dropout != None:
            out = self.dropout(out)
        out = self.model(out)
        return out
