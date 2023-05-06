import torch
import torch.nn as nn

import net

from einops import rearrange


class LongMemory(nn.Module):
    def __init__(self, dim, codebook_len) -> None:
        super().__init__()
        self.register_parameter(
            "memory", nn.Parameter(torch.rand(size=[codebook_len, dim]))
        )

    def forward(self, x):
        similarity = torch.einsum("kc,bchw->bkhw", self.memory, x)  # b codebook_len h w
        similarity = nn.functional.softmax(similarity, dim=1)
        result = torch.einsum("kc,bkhw->bchw", self.memory, similarity)
        return result


class ChannelwiseMemory(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.register_parameter(
            "memory", nn.Parameter(torch.rand(size=[in_channels, out_channels]))
        )

    def forward(self, x):
        attention = nn.functional.softmax(x, dim=1)
        result = torch.einsum("ki,bkhw->bihw", self.memory, attention)
        return result


class ChannelwiseMemoryMultiHead(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, head_dim):
        super().__init__()
        assert out_channels % n_heads == 0
        self.head_out = out_channels // n_heads
        self.register_parameter(
            "memory",
            nn.Parameter(
                (torch.rand(size=[n_heads, head_dim, self.head_out]) * 2.0 - 1.0)
            ),
        )
        self.n_heads = n_heads
        self.to_mem_key = net.Conv2DBlock(
            in_dim=in_channels, out_dim=self.n_heads * head_dim, kernel_rad=0, bias=True
        )

    def forward(self, x):
        mem_key = self.to_mem_key(x)
        mem_key = rearrange(mem_key, "b (c m) h w -> b m c h w", m=self.n_heads)
        mem_key = nn.functional.softmax(mem_key, dim=2)
        result = torch.einsum("nki,bnkhw->bnihw", self.memory, mem_key)
        result = rearrange(result, "b m c h w -> b (m c) h w", m=self.n_heads)
        return result


class ConvMemory(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.register_parameter(
            "memory",
            nn.Parameter(
                torch.rand(size=[in_channels * kernel_size * kernel_size, out_channels])
            ),
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        out = x
        out = torch.nn.functional.unfold(
            out, kernel_size=self.kernel_size, padding=self.kernel_size // 2
        )
        out = nn.functional.softmax(out, dim=1)
        out = torch.einsum("ki,bkl->bil", self.memory, out)
        out = rearrange(out, "b c (h w) -> b c h w", h=h, w=w)
        return out


class ConvMemoryMultiHead(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads, head_dim, kernel_size):
        super().__init__()
        self.head_out = out_channels // n_heads
        self.kernel_size = kernel_size
        self.register_parameter(
            "memory",
            nn.Parameter(
                (
                    torch.rand(
                        size=[
                            n_heads,
                            head_dim * kernel_size * kernel_size,
                            self.head_out,
                        ]
                    )
                    * 2.0
                    - 1.0
                )
            ),
        )
        self.n_heads = n_heads
        self.to_mem_key = net.Conv2DBlock(
            in_dim=in_channels, out_dim=self.n_heads * head_dim, kernel_rad=0, bias=True
        )
        self.pad = nn.ZeroPad2d(
            (
                self.kernel_size // 2,
                self.kernel_size % 2,
                self.kernel_size // 2,
                self.kernel_size % 2,
            )
        )

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        out = x
        out = self.to_mem_key(out)
        out = self.pad(out)
        out = torch.nn.functional.unfold(out, kernel_size=self.kernel_size)
        out = rearrange(out, "b (c m) l -> b m c l", m=self.n_heads)
        out = nn.functional.softmax(out, dim=2)
        out = torch.einsum("nki,bnkl->bnil", self.memory, out)
        out = rearrange(out, "b m c (h w) -> b (m c) h w", m=self.n_heads, h=h, w=w)
        return out


class PathMemory(nn.Module):
    def __init__(self, path_size, in_channels, out_channels, head_dim, n_heads):
        super().__init__()
        dim_in = path_size * path_size * in_channels
        head_out = path_size * path_size * out_channels // n_heads
        self.path_size = path_size
        self.heads = n_heads
        self.linear = net.Linear(
            in_dim=dim_in, out_dim=path_size * path_size * head_dim
        )
        self.register_parameter(
            "memory",
            nn.Parameter((torch.rand(size=[n_heads, head_dim, head_out]) * 2.0 - 1.0)),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        ph = h // self.path_size
        out = x
        out = rearrange(
            out,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.path_size,
            p2=self.path_size,
        )
        out = self.linear(out)
        out = rearrange(out, "b p (c m)-> b m c p", m=self.heads)
        out = nn.functional.softmax(out, dim=2)
        out = torch.einsum("nki,bnkl->bnil", self.memory, out)
        out = rearrange(
            out,
            "b m (p1 p2 c) (h w) -> b (m c) (h p1) (w p2)",
            m=self.heads,
            p1=self.path_size,
            p2=self.path_size,
            h=ph,
        )
        return out


class KeyChannelwiseMemory(nn.Module):
    def __init__(self, in_dim, memory_dim, out_dim):
        super().__init__()
        self.register_parameter(
            "key", nn.Parameter((torch.rand(size=[in_dim, memory_dim])))
        )
        self.register_parameter(
            "memory", nn.Parameter(torch.rand(size=[memory_dim, out_dim]) * 2.0 - 1.0)
        )

    def forward(self, x):
        out = x
        out = torch.einsum("ki,bkhw->bihw", self.key, out)
        out = nn.functional.softmax(out, dim=2)
        out = torch.einsum("ki,bkhw->bihw", self.memory, out)
        return out


class KeyChannelwiseMemoryMultiHead(nn.Module):
    def __init__(self, in_dim, n_heads, key_dim, memory_dim, head_dim, out_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_out = out_dim // n_heads
        self.register_parameter(
            "key", nn.Parameter((torch.rand(size=[n_heads, key_dim, memory_dim])))
        )
        self.register_parameter(
            "memory",
            nn.Parameter(torch.rand(size=[n_heads, memory_dim, head_dim]) * 0.5 - 0.25),
        )
        self.to_mem_key = net.Conv2DBlock(
            in_dim=in_dim, out_dim=self.n_heads * key_dim, kernel_rad=0, bias=True
        )
        self.to_out = None
        if out_dim != head_dim * n_heads:
            self.to_out = net.Conv2DBlock(
                in_dim=head_dim * n_heads, out_dim=out_dim, kernel_rad=0, bias=True
            )

    def forward(self, x):
        out = x
        out = self.to_mem_key(out)
        out = rearrange(out, "b (c m) h w -> b m c h w", m=self.n_heads)
        out = torch.einsum("nki,bnkhw->bnihw", self.key, out)
        out = nn.functional.softmax(out, dim=2)
        out = torch.einsum("nki,bnkhw->bnihw", self.memory, out)
        out = rearrange(out, "b m c h w -> b (m c) h w", m=self.n_heads)
        if self.to_out != None:
            out = self.to_out(out)
        return out


class KeyConvMemoryMultiHead(nn.Module):
    def __init__(
        self, in_dim, n_heads, key_dim, memory_dim, head_dim, out_dim, kernel_size
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_out = out_dim // n_heads
        self.kernel_size = kernel_size
        self.register_parameter(
            "key",
            nn.Parameter(
                (
                    torch.rand(
                        size=[n_heads, key_dim * kernel_size * kernel_size, memory_dim]
                    )
                )
            ),
        )
        self.register_parameter(
            "memory",
            nn.Parameter(torch.rand(size=[n_heads, memory_dim, head_dim]) * 2.0 - 1.0),
        )
        self.to_mem_key = net.Conv2DBlock(
            in_dim=in_dim, out_dim=self.n_heads * key_dim, kernel_rad=0, bias=True
        )
        self.pad = nn.ZeroPad2d(
            (
                self.kernel_size // 2,
                self.kernel_size % 2,
                self.kernel_size // 2,
                self.kernel_size % 2,
            )
        )
        self.to_out = None
        if out_dim != head_dim * n_heads:
            self.to_out = net.Conv2DBlock(
                in_dim=head_dim * n_heads, out_dim=out_dim, kernel_rad=0, bias=True
            )

    def forward(self, x):
        b, c, h, w = x.shape
        out = x
        out = self.to_mem_key(out)
        out = self.pad(out)
        out = torch.nn.functional.unfold(out, kernel_size=self.kernel_size)
        out = rearrange(out, "b (c m) l -> b m c l", m=self.n_heads)
        out = torch.einsum("nki,bnkl->bnil", self.key, out)
        out = nn.functional.softmax(out, dim=2)
        out = torch.einsum("nki,bnkl->bnil", self.memory, out)
        out = rearrange(out, "b m c (h w) -> b (m c) h w", m=self.n_heads, h=h, w=w)
        if self.to_out != None:
            out = self.to_out(out)
        return out
