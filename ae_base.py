import torch.nn as nn

from abc import abstractmethod

class AEBase(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode_features(self, input):
        pass

    @abstractmethod
    def encode_bottleneck(self, input):
        pass

    def encode(self, input):
        out = input
        out = self.encode_features(out)
        out = self.encode_bottleneck(out)
        return out

    @abstractmethod
    def decode_expand(self, input):
        pass

    @abstractmethod
    def decode_features(self, input):
        pass

    def decode(self, input):
        out = input
        out = self.decode_expand(out)
        out = self.decode_features(out)
        return out

    @abstractmethod
    def calc_cold_loss(self, encode_result, decode_result):
        pass

    @abstractmethod
    def calc_hot_loss(self, encode_result, decode_result):
        pass

    def forward(self, input):
        out = input
        out = self.encode(input)
        out = self.decode(input)
        return out
