import torch
from torch import nn
import torch.nn.functional as F

from fast_hadamard_transform import hadamard_transform

from ..utils import QuartetDtype, QuartetConfig
from .linear_fns import forward_quantize, QuartetMasterWeightsFn, QuartetNoMasterWeightsFn


class QuartetLinear(nn.Linear):
    def __init__(self, *args, config: QuartetConfig, **kwargs):
        super().__init__(*args, **kwargs)

        self.config = config
        
        # Quantized tensors buffers
        match self.config.forward_dtype:
            case QuartetDtype.MXFP4:
                self.register_buffer(
                    "weight_q",
                    torch.empty(self.weight.shape[0], self.weight.shape[1] // 2, dtype=torch.uint8, device=self.weight.device),
                )
            case QuartetDtype.MXFP8:
                self.register_buffer(
                    "weight_q",
                    torch.empty(*self.weight.shape, dtype=torch.uint8, device=self.weight.device),
                )
            case _:
                raise ValueError(f"Unsupported forward dtype: {config.forward_dtype}")
        self.register_buffer(
            "shared_exponents",
            torch.empty(self.weight.shape[0], self.weight.shape[1] // 32, dtype=torch.float8_e8m0fnu, device=self.weight.device),
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
            self.weight_q = None
            self.shared_exponents = None
        else:
            weight_q, shared_exponents, _ = forward_quantize(self.weight, self.forward_hadamard_matrix, self.config.forward_dtype)
            self.weight_q = nn.Parameter(weight_q, requires_grad=False)
            self.shared_exponents = nn.Parameter(shared_exponents, requires_grad=False)
            self.weight = None

    def forward(self, x) -> torch.Tensor:
        if self.config.store_master_weights:
            return QuartetMasterWeightsFn.apply(
                x, self.weight, self.bias, self.forward_hadamard_matrix, self.backward_hadamard_matrix, self.config.forward_dtype,
            )
        else:
            return QuartetNoMasterWeightsFn.apply(
                x, self.weight_q, self.shared_exponents, self.bias, self.forward_hadamard_matrix, self.backward_hadamard_matrix, self.config.forward_dtype,
            )

