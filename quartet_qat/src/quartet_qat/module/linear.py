import torch
from torch import nn
import torch.nn.functional as F

from fast_hadamard_transform import hadamard_transform

from ..utils import QuartetDtype, QuartetConfig
from .linear_fns import (
    forward_quantize,
    Quartet4x4MasterFn,
    Quartet4x4NoMasterFn,
    Quartet4x16MasterFn,
)


class QuartetLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: QuartetConfig, bias: bool = True, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.config = config
        
        # Quantized tensors buffers
        match self.config.forward_dtype:
            case QuartetDtype.MXFP4:
                self.register_buffer(
                    "qweight",
                    torch.empty(self.weight.shape[0], self.weight.shape[1] // 2, dtype=torch.uint8, device=self.weight.device),
                )
            case QuartetDtype.MXFP8:
                self.register_buffer(
                    "qweight",
                    torch.empty(*self.weight.shape, dtype=torch.uint8, device=self.weight.device),
                )
            case _:
                raise ValueError(f"Unsupported forward dtype: {config.forward_dtype}")
        self.register_buffer(
            "scales",
            torch.empty(self.weight.shape[0], self.weight.shape[1] // 32, dtype=torch.uint8, device=self.weight.device),
        )
        
        # Rotation matrices buffers
        self.register_buffer(
            "forward_hadamard_matrix",
            torch.empty(self.config.hadamard_group_size, self.config.hadamard_group_size, dtype=self.weight.dtype, device=self.weight.device),
        )
        self.register_buffer(
            "backward_hadamard_matrix",
            torch.empty(self.config.hadamard_group_size, self.config.hadamard_group_size, dtype=self.weight.dtype, device=self.weight.device),
        )
    
    @torch.no_grad()
    def pre_forward(self):
        # Generate rotation matrices
        assert self.weight.shape[1] % self.config.hadamard_group_size == 0, "Weight shape must be divisible by hadamard group size"
        assert self.weight.data.is_cuda, "Weight must be on CUDA"
        self.forward_hadamard_matrix = nn.Parameter(
            hadamard_transform(
                torch.eye(self.config.hadamard_group_size, dtype=self.weight.dtype, device=self.weight.device),
                scale=self.config.hadamard_group_size ** -0.5,
            )
        )
        self.backward_hadamard_matrix = nn.Parameter(
            hadamard_transform(
                torch.eye(self.config.hadamard_group_size, dtype=self.weight.dtype, device=self.weight.device),
                scale=self.config.hadamard_group_size ** -0.5,
            )
        )
        
        # Quantize weights
        if self.config.store_master_weights:
            self.qweight = None
            self.scales = None
        else:
            weight_q, scales, _ = forward_quantize(self.weight, self.forward_hadamard_matrix, self.config.forward_dtype, self.config.forward_quest)
            self.qweight = nn.Parameter(weight_q, requires_grad=False)
            self.scales = nn.Parameter(scales.view(dtype=torch.uint8), requires_grad=False)
            self.weight = None

    def forward(self, x) -> torch.Tensor:
        match (self.config.forward_dtype, self.config.backward_dtype, self.config.store_master_weights):
            case (QuartetDtype.MXFP4, QuartetDtype.MXFP4, True):
                return Quartet4x4MasterFn.apply(
                    x, self.weight, self.bias, self.forward_hadamard_matrix, self.backward_hadamard_matrix, self.config.forward_dtype, self.config.forward_quest,
                )
            case (QuartetDtype.MXFP4, QuartetDtype.MXFP4, False):
                return Quartet4x4NoMasterFn.apply(
                    x, self.qweight, self.scales, self.bias, self.forward_hadamard_matrix, self.backward_hadamard_matrix, self.config.forward_dtype, self.config.forward_quest,
                )
            case (QuartetDtype.MXFP4, QuartetDtype.BF16, True):
                return Quartet4x16MasterFn.apply(
                    x, self.weight, self.bias, self.forward_hadamard_matrix, self.config.forward_dtype, self.config.forward_quest,
                )
            case _:
                raise ValueError(f"Forward dtype: {self.config.forward_dtype}, backward dtype: {self.config.backward_dtype}, store_master_weights: {self.config.store_master_weights} isn't supported yet.")
