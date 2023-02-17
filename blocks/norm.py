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

class ModLayerNorm(nn.Module):
    def __init__(self, num_features, amplitude_mod : nn.Module, affine_mod=True, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1,num_features,1))
        self.bias = nn.Parameter(torch.zeros(1,num_features,1))
        self.affine_mod = affine_mod
        self.eps = eps
        self.ampl = amplitude_mod
        self.num_features = num_features

    def forward(self, x, emb=None):
        _,c,h,w = x.size()
        assert c == self.num_features, "Input tensor features: {} expected features: {}".format(str(c), str(self.num_features))
        x = rearrange(x, 'b c h w -> b c (h w)')
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)
        var_sq = (var + self.eps).rsqrt()
        x = (x - mean) * var_sq
        amp_in = mean.squeeze(dim=-1)
        if emb != None:
            amp_in = torch.cat([amp_in, emb], dim=1)
        if self.affine_mod:
            amp_w, amp_b = self.ampl(amp_in).chunk(2, dim=1)
            amp_w = amp_w.unsqueeze(dim=-1)
            amp_b = amp_b.unsqueeze(dim=-1)
            x = x * amp_w + amp_b
        else:
            amp_w = self.ampl(amp_in)
            amp_w = amp_w.unsqueeze(dim=-1)
            x = x * amp_w
        x = rearrange(x, 'b c (h w) -> b c h w', h=h, w=w)
        return x
