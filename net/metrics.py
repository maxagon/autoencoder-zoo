import torch
import torch.nn as nn
import torchmetrics as tm

from loss import lpips


class SSIM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, grountruth: torch.Tensor):
        assert input.device == grountruth.device
        with torch.no_grad():
            return tm.functional.structural_similarity_index_measure(
                input, grountruth.to(input.dtype), data_range=2.0, reduction="none"
            )


class PSNR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, grountruth: torch.Tensor):
        assert input.device == grountruth.device
        with torch.no_grad():
            return tm.functional.peak_signal_noise_ratio(
                input, grountruth.to(input.dtype), data_range=2.0, reduction="none"
            )


class LPIPS(nn.Module):
    def __init__(self, lpips_model: lpips.LPIPS):
        super().__init__()
        self.lpips_model = lpips_model

    def forward(self, input: torch.Tensor, grountruth: torch.Tensor):
        assert input.device == grountruth.device == self.lpips_model.device
        return self.lpips_model(input, grountruth)
