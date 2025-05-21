import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


from .backward import EW_EtX_Scheme, BACKWARD_SCHEMES
from .quantizers import NoQuantizer, QUANTIZER_CLASSES


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        weight_quantizer=None,
        activation_quantizer=None,
        gradient_quantizer=None,
        backward_scheme=None,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)
        if weight_quantizer is None:
            weight_quantizer = NoQuantizer()
        if activation_quantizer is None:
            activation_quantizer = NoQuantizer()
        if gradient_quantizer is None:
            gradient_quantizer = NoQuantizer()
        if backward_scheme is None:
            backward_scheme = EW_EtX_Scheme()
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer
        self.backward_scheme = backward_scheme
        self.backward_scheme.g_quantizer = gradient_quantizer

    def forward(self, x): 
        xq = self.activation_quantizer(x)
        wq = self.weight_quantizer(self.weight)
        return self.backward_scheme(
            xq,
            wq,
        )
