import torch
import torch.nn as nn

from abc import abstractmethod

class AEBase(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def encode(self, input):
        pass

    @abstractmethod
    def sample(self, input):
        pass

    @abstractmethod
    def decode(self, input):
        pass

    @abstractmethod
    def calc_cold_loss(self, encode_result, decode_result):
        pass

    @abstractmethod
    def calc_hot_loss(self, encode_result, decode_result):
        pass