from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from qutlass import matmul_mxf4_bf16_tn, fusedQuantize
from qutlass.utils import to_blocked

from fast_hadamard_transform import hadamard_transform

from ..utils import QuartetDtype


@dataclass
class QuartetLinearConfig:
    forward_dtype: QuartetDtype = QuartetDtype.MXFP4
    backward_dtype: QuartetDtype = QuartetDtype.MXFP4
    store_master_weights: bool = False
    hadamard_group_size: int = 32


class QuartetLinear(nn.Linear):
    def __init__(self, *args, config: QuartetLinearConfig, **kwargs):
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
            torch.empty(self.weight.shape[0] * self.weight.shape[1] // 32, dtype=torch.float8_e8m0fnu, device=self.weight.device),
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
        
    def forward_quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        match self.config.forward_dtype:
            case QuartetDtype.MXFP4:
                x_hf_e2m1, x_hf_e8m0, _ = fusedQuantize(x, self.forward_hadamard_matrix)
                shared_exps = to_blocked(x_hf_e8m0)
                return x_hf_e2m1, shared_exps
            case QuartetDtype.MXFP8:
                raise NotImplementedError("MXFP8 is not supported for forward quantization yet")
            case _:
                raise ValueError(f"Unsupported forward dtype: {self.config.forward_dtype}")
    
    @torch.no_grad()
    def pre_forward(self):
        # Generate rotation matrices
        assert self.weight.shape[1] % self.config.hadamard_group_size == 0, "Weight shape must be divisible by hadamard group size"
        assert self.weight.data.is_cuda, "Weight must be on CUDA"
        self.forward_hadamard_matrix.copy_(
            hadamard_transform(
                torch.eye(self.config.hadamard_group_size, dtype=self.weight.dtype, device=self.weight.device),
                scale=self.config.hadamard_group_size ** -0.5,
            )
        )
        self.backward_hadamard_matrix.copy_(
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
            weight_q, shared_exponents = self.forward_quantize(self.weight)
            self.weight_q.copy_(weight_q)
            self.shared_exponents.copy_(shared_exponents)
            self.weight = None
    
    @torch.compile()
    @torch.inference_mode()
    def forward(self, x) -> torch.Tensor:
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_shared_exponents = self.forward_quantize(x_flat)

        # Quantize weights
        if self.config.store_master_weights:
            weight_q, shared_exponents = self.forward_quantize(self.weight)
        else:
            weight_q, shared_exponents = self.weight_q, self.shared_exponents

        y = matmul_mxf4_bf16_tn(x_flat_q, weight_q, x_flat_shared_exponents, shared_exponents, 1.)
        
        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if self.bias is not None:
            y += self.bias
        return y
