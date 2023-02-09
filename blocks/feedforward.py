from typing import Optional

import torch.nn as nn

import blocks.init as init
import blocks.nonlinear as nl

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, dropout=0.0, nonlinearity : Optional[nl.Nonlinearity] = None, init_params : Optional[init.InitParams] = None):
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

class Conv2DBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_rad, pad_type='none', bias=True, dropout=0.0, stride=1, 
        nonlinearity : Optional[nl.Nonlinearity] = None, init_params : Optional[init.InitParams] = None):
        super().__init__()

        # dropout
        self.dropout = None
        if dropout != 0.0:
            self.dropout = nn.Dropout2d(dropout)

        model = []

        # pad
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
