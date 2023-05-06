import torch
import torch.nn as nn

import net

from einops import rearrange, pack, unpack, repeat, reduce


class ConvKernel(nn.Module):
    def __init__(self, dim, kernel_size) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn((dim, dim, kernel_size, kernel_size))
        )  # out_c in_c k k
        nn.init.kaiming_normal_(
            self.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu"
        )

    def forward(self, batch_size):
        return repeat(self.weight, "... -> b ...", b=batch_size)  # b out_c in_c k k


# Sample-adaptive kernel selection from GigaGAN https://arxiv.org/abs/2303.05511
class FilterBankKernel(nn.Module):
    def __init__(self, bank_filters, dim, kernel_size) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn((bank_filters, dim, dim, kernel_size, kernel_size))
        )  # bf out_c in_c k k
        self.bank_filters = bank_filters

    def forward(self, bank_request):
        assert (
            len(bank_request.shape) == 2 and bank_request.shape[1] == self.bank_filters
        ), "Incorrect bank request. Expected (batch size, bank filters)."

        batch_size = bank_request.shape[0]
        weights = repeat(self.weight, "... -> b ...", b=batch_size)

        bank_request = nn.functional.softmax(bank_request, dim=1)
        bank_request = rearrange(bank_request, "b f -> b f 1 1 1 1")

        kernel = reduce(weights * bank_request, "b f ... -> b ...", "sum")
        return kernel


class SparseFilterBankKernel(nn.Module):
    def __init__(self, dim, bank_filters, bank_filter_dim, kernel_size) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn((bank_filters, bank_filter_dim, dim, kernel_size, kernel_size))
        )  # bf bfd in_c k k
        self.bank_filters = bank_filters
        assert dim % bank_filter_dim == 0
        self.channel_groups = dim // bank_filter_dim

    def forward(self, bank_request):
        # bank_request (batch size, channel groups, bank filters)
        bank_request = nn.functional.softmax(bank_request, dim=2)
        bank_request = rearrange(bank_request, "b g f -> b g f 1 1 1 1")
        weights = rearrange(self.weight, "... -> 1 1 ...")
        kernel = weights * bank_request
        kernel = reduce(kernel, "b g f ... -> b g ...", "sum")
        kernel = rearrange(kernel, "b g o ... -> b (g o) ...")
        return kernel


# StyleGAN2 modulation/demodulation https://arxiv.org/abs/1912.04958
class StyleModulator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, weights, style):
        return weights * (1.0 + style)


class NormModulator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, weights):
        norm = weights**2  # b out_c in_c k k
        norm = torch.sum(norm, dim=(2, 3, 4), keepdim=True)  # b out_c 1 1 1
        eps = 1e-8
        norm = torch.rsqrt(norm + eps)  # b out_c 1 1 1
        weights = weights * norm  # b out_c in_c k k
        return weights


def conv2d_kernel_batch(x, kernel):
    out = x
    b, dim_out, dim_in, kh, kw = kernel.shape
    assert (
        x.shape[1] == dim_in
    ), f"Incorrect dimensions. Kernel shape: {kernel.shape}. Image shape: {x.shape}"
    assert kh == kw and kh % 2 == 1, "Only symmetrical odd kernels supported"
    pad = kh // 2
    # pack batch elements into conv groups (each batch element has it own kernel computed above)
    weights = rearrange(weights, "b o i h w -> (b o) i h w")
    out = rearrange(out, "b c h w -> 1 (b c) h w")
    out = nn.functional.conv2d(out, weights, groups=b, padding=pad)
    out = rearrange(out, "1 (b c) h w -> b c h w")
    return out


class ModulatedConv(nn.Module):
    def __init__(self, dim, kernel_rad, style_mod=True, demodulate=True) -> None:
        super().__init__()
        kernel_size = kernel_rad * 2 + 1
        self.kernel_gen = ConvKernel(dim=dim, kernel_size=kernel_size)

        self.style_mod = None
        if style_mod:
            self.style_mod = StyleModulator()

        self.demodulate = None
        if demodulate:
            self.demodulate = NormModulator()

    def forward(self, x, style=None):
        kernel = self.kernel_gen(x.shape[0])

        if self.style_mod != None:
            assert style != None
            kernel = self.style_mod(kernel, style)

        if self.demodulate != None:
            kernel = self.demodulate(kernel)

        out = conv2d_kernel_batch(x, kernel)
        return out


class BankModulatedConv(nn.Module):
    def __init__(
        self,
        dim,
        kernel_rad,
        bank_filters,
        bank_filter_dim,
        style_mod=True,
        demodulate=True,
    ) -> None:
        super().__init__()
        kernel_size = kernel_rad * 2 + 1
        if dim == bank_filter_dim:
            self.kernel_gen = FilterBankKernel(
                bank_filters=bank_filters, dim=dim, kernel_size=kernel_size
            )
        else:
            self.kernel_gen = SparseFilterBankKernel(
                dim=dim,
                bank_filters=bank_filters,
                bank_filter_dim=bank_filter_dim,
                kernel_size=kernel_size,
            )
        self.kernel_gen = ConvKernel(dim=dim, kernel_size=kernel_size)

        self.style_mod = None
        if style_mod:
            self.style_mod = StyleModulator()

        self.demodulate = None
        if demodulate:
            self.demodulate = NormModulator()

    def forward(self, x, bank_request, style=None):
        kernel = self.kernel_gen(bank_request)

        if self.style_mod != None:
            assert style != None
            kernel = self.style_mod(kernel, style)

        if self.demodulate != None:
            kernel = self.demodulate(kernel)

        out = conv2d_kernel_batch(x, kernel)
        return out
