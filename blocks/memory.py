import torch
import torch.nn as nn

import blocks.feedforward as ff

from einops import rearrange

class ChannelwiseMemory(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.register_parameter("memory", nn.Parameter(torch.rand(size=[in_channels, out_channels])))

    def forward(self, x):
        attention = nn.functional.softmax(x, dim=1)
        result = torch.einsum("ki,bkhw->bihw", self.memory, attention)
        return result

class ChannelwiseMemoryMultiHead(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, head_dim):
        super().__init__()
        self.head_out = out_channels // n_heads
        self.register_parameter("memory", nn.Parameter(torch.rand(size=[n_heads, head_dim, self.head_out])))
        self.n_heads = n_heads
        self.to_mem_key = ff.Conv2DBlock(in_dim=in_channels, out_dim=self.n_heads*head_dim, kernel_rad=0, bias=True)
    def forward(self, x):
        mem_key = self.to_mem_key(x)
        mem_key = rearrange(mem_key, "b (c m) h w -> b m c h w", m=self.n_heads)
        mem_key = nn.functional.softmax(mem_key, dim=2)
        result = torch.einsum("nki,bnkhw->bnihw", self.memory, mem_key)
        result = rearrange(result, "b m c h w -> b (m c) h w", m=self.n_heads)
        return result