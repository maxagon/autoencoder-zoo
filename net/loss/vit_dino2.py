import torch
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np


class VitDinoV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        # original repo https://github.com/facebookresearch/dinov2
        # uses .ToTensor() and .Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225) transform
        # forward call expect image in [-1.0, 1.0] range
        # normalize input image from range Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) to range Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        mean_expected = np.array([0.485, 0.456, 0.406])
        std_expected = np.array([0.229, 0.224, 0.225])
        mean = 2.0 * mean_expected - 1.0
        std = 2.0 * std_expected
        self.scaling = transforms.Normalize(tuple(mean.tolist()), tuple(std.tolist()))

    def forward(self, input, target):
        input_scaled = self.scaling(input)
        target_scaled = self.scaling(target)
        loss = -torch.nn.functional.cosine_similarity(
            self.vit_model(input_scaled), self.vit_model(target_scaled)
        )
        loss = torch.mean(loss)
        return loss
