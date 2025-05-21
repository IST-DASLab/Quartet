import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseQuantizer(nn.Module):
    def __init__(self, bits=4):
        super().__init__()
        self.bits = bits
        self.n_levels = 2**bits
        
    def forward(self, x):
        raise NotImplementedError
    
    def re_randomize(self):
        pass


class NoQuantizer(BaseQuantizer):
    def __init__(self, **kwargs):
        super().__init__(16)

    def forward(self, x):
        return x


OPTIMAL_GAUSSIAN_SCALES = {
    1: 0.7978845587140913,
    1.585: 1.2240089519030855,
    2: 1.4935346200015913,
    3: 2.051068354131873,
    4: 2.513930578568423,
    5: 2.9160938834961225,
    6: 3.276597282593217,
    7: 3.6010497188221655,
    8: 3.884938678807525,
}
