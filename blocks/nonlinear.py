from abc import abstractmethod

import torch.nn as nn

class Nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def calc_gain(self):
        pass

class ReLU(Nonlinearity):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.relu(x, inplace=False)

    def calc_gain(self):
        return nn.init.calculate_gain("relu")
