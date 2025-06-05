from typing import Optional

import torch
from torch import nn
from torch.autograd import Function

from qutlass import matmul_mxf4_bf16_tn, fusedQuantize
from qutlass.utils import to_blocked

from quartet_qat.utils.dtypes import QuartetDtype


def forward_quantize(x: torch.Tensor, hadamard_matrix: torch.Tensor, dtype: QuartetDtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    match dtype:
        case QuartetDtype.MXFP4:
            x_hf_e2m1, x_hf_e8m0, x_mask = fusedQuantize(x, hadamard_matrix)
            shared_exps = to_blocked(x_hf_e8m0)
            return x_hf_e2m1, shared_exps, x_mask
        case QuartetDtype.MXFP8:
            raise NotImplementedError("MXFP8 is not supported for forward quantization yet")
        case _:
            raise ValueError(f"Unsupported forward dtype: {dtype}")


class QuartetMasterWeightsFn(Function):
    @staticmethod
    @torch.compile()
    @torch.inference_mode()
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], hadamard_matrix: torch.Tensor, dtype: QuartetDtype):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_shared_exponents, x_flat_mask = forward_quantize(x_flat, hadamard_matrix, dtype)

        # Quantize weights
        weight_q, weight_shared_exponents, weight_mask = forward_quantize(weight, hadamard_matrix, dtype)

        y = matmul_mxf4_bf16_tn(x_flat_q, weight_q, x_flat_shared_exponents, weight_shared_exponents, 1.)
        
        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if bias is not None:
            y += bias
        
        ctx.x_shape = x.shape
        ctx.dtype = dtype
        ctx.save_for_backward(x_flat_q, weight_q, x_flat_shared_exponents, weight_shared_exponents, x_flat_mask, weight_mask, hadamard_matrix)
        
        return y
    
    @staticmethod
    @torch.compile()
    @torch.inference_mode()
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("Backward pass is not implemented yet")


class QuartetNoMasterWeightsFn(Function):
    @staticmethod
    @torch.compile()
    @torch.inference_mode()
    def forward(ctx, x: torch.Tensor, weight_q: torch.Tensor, weight_shared_exponents: torch.Tensor, bias: Optional[torch.Tensor], hadamard_matrix: torch.Tensor, dtype: QuartetDtype):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_shared_exponents, x_flat_mask = forward_quantize(x_flat, hadamard_matrix, dtype)

        y = matmul_mxf4_bf16_tn(x_flat_q, weight_q, x_flat_shared_exponents, weight_shared_exponents, 1.)
        
        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if bias is not None:
            y += bias
        
        ctx.x_shape = x.shape
        ctx.dtype = dtype
        ctx.save_for_backward(x_flat_q, weight_q, x_flat_shared_exponents, weight_shared_exponents, x_flat_mask, hadamard_matrix)
        
        return y
    
    @staticmethod
    @torch.compile()
    @torch.inference_mode()
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("Backward pass is not implemented yet")