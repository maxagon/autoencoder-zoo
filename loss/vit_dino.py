import torch
import torch.nn as nn

class VitDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.loss = nn.MSELoss()

    def forward(self, input, target):
        return self.loss(self.vit_model(self.input), self.vit_model(self.target))
