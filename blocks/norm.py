import torch
import torch.nn as nn
from einops import rearrange

class BigBatchNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.norm = nn.BatchNorm1d(num_features=1)

    def forward(self, x):
        _,c,h,w = x.size()
        x = rearrange(x, 'b c h w -> b (c h w)').unsqueeze(dim=1)
        x = self.norm(x)
        x = x.squeeze(dim=1)
        x = rearrange(x, 'b (c h w) -> b c h w', c=c, h=h, w=w)
        x = x * self.weight.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1) + self.bias.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return x
