import torch
import torch.nn as nn
import torchmetrics as tm


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
