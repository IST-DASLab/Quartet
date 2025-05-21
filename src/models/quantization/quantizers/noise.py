import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .base import BaseQuantizer


class NoiseQuantizer(BaseQuantizer):
    def __init__(self, bits=4):
        super().__init__(bits)
        self.noise_scale = 2 ** (-bits)
        self.counter = 0

    @torch.compiler.disable()
    def forward(self, x):
        std = torch.std(x, correction=0, dim=-1, keepdim=True)
        return x + torch.randn_like(x) * std * self.noise_scale
