import torch
import torch.nn as nn

import init

class Linear(nn.Module):
    DEFAULT_INIT = init.InitParams()
    def __init__(self, in_dim, out_dim, bias=True, init_params : init.InitParams = None):
        super().__init__()
        self.layer = nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)

        if bias:
            nn.init.zeros_(self.layer.bias)
        if init_params == None:
            init_params = Linear.DEFAULT_INIT
        self.layer.weight = init_params

    def forward(self, x):
        return self.layer(x)
