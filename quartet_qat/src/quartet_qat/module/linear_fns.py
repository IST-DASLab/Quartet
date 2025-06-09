from typing import Optional

import torch
from torch import nn
from torch.autograd import Function

from qutlass import matmul_mxf4_bf16_tn, fusedQuantize, fusedQuantize_bwd
from qutlass.utils import to_blocked
import quartet

from ..utils import QuartetDtype


@torch.compile()
@torch.inference_mode()
def _unpack_mask(clip_mask: torch.Tensor) -> torch.Tensor:
    clip_mask_unpacked_dq = torch.zeros(*clip_mask.shape[:-1], clip_mask.size(-1) * 8, dtype=torch.bool, device=clip_mask.device)
    for i in range(8):
        clip_mask_unpacked_dq[..., i::8] = (clip_mask >> i) & 1
    return clip_mask_unpacked_dq


def forward_quantize(x: torch.Tensor, hadamard_matrix: torch.Tensor, dtype: QuartetDtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    match dtype:
        case QuartetDtype.MXFP4:
            return fusedQuantize(x, hadamard_matrix)
        case QuartetDtype.MXFP8:
            raise NotImplementedError("MXFP8 is not supported for forward quantization yet")
        case _:
            raise ValueError(f"Unsupported forward dtype: {dtype}")


class QuartetMasterWeightsFn(Function):
    @staticmethod
    @torch.compile()
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], forward_hadamard_matrix: torch.Tensor, backward_hadamard_matrix: torch.Tensor, dtype: QuartetDtype):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_shared_exponents, x_flat_mask = forward_quantize(x_flat, forward_hadamard_matrix, dtype)

        # Quantize weights
        weight_q, weight_shared_exponents, weight_mask = forward_quantize(weight, forward_hadamard_matrix, dtype)

        y = matmul_mxf4_bf16_tn(x_flat_q, weight_q, to_blocked(x_flat_shared_exponents), to_blocked(weight_shared_exponents), 1.)
        
        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if bias is not None:
            y += bias
        
        ctx.x_shape = x.shape
        ctx.dtype = dtype
        ctx.bias_present = bias is not None
        ctx.save_for_backward(
            x_flat_q,
            weight_q,
            x_flat_shared_exponents,
            weight_shared_exponents,
            x_flat_mask,
            weight_mask,
            forward_hadamard_matrix,
            backward_hadamard_matrix,
        )
        
        return y
    
    @staticmethod
    @torch.compile()
    def backward(ctx, grad_output: torch.Tensor):
        x_flat_q, weight_q, x_flat_shared_exponents, weight_shared_exponents, x_flat_mask, weight_mask, forward_hadamard_matrix, backward_hadamard_matrix = ctx.saved_tensors

        backward_hadamard_matrix = backward_hadamard_matrix * (
            torch.randint(0, 2, (32,), device=backward_hadamard_matrix.device, dtype=backward_hadamard_matrix.dtype)
            * 2. - 1.
        )

        grad_output_q, grad_output_shared_exponents = fusedQuantize_bwd(
            grad_output.flatten(end_dim=-2),
            backward_hadamard_matrix
        )

        weight_qtq, weight_qt_shared_exponents = quartet.backward_qt_bf16(weight_q, weight_shared_exponents, backward_hadamard_matrix, alpha=1.)
        grad_input = matmul_mxf4_bf16_tn(grad_output_q, weight_qtq, to_blocked(grad_output_shared_exponents), to_blocked(weight_qt_shared_exponents), 1. / 9.)

        x_flat_mask = _unpack_mask(x_flat_mask)
        grad_input = (
            (grad_input.view(-1, 32) * x_flat_mask.view(-1, 32).to(grad_input.dtype))
            @ forward_hadamard_matrix.T
        ).view(ctx.x_shape)

        grad_output_tq, grad_output_t_shared_exponents = quartet.backward_t_bf16(grad_output.flatten(end_dim=-2), backward_hadamard_matrix)
        x_flat_qtq, x_flat_qt_shared_exponents = quartet.backward_qt_bf16(x_flat_q, x_flat_shared_exponents, backward_hadamard_matrix, alpha=1.)
        grad_output_t_shared_exponents = to_blocked(grad_output_t_shared_exponents)
        x_flat_qt_shared_exponents = to_blocked(x_flat_qt_shared_exponents)
        grad_weight_hf = matmul_mxf4_bf16_tn(grad_output_tq, x_flat_qtq, grad_output_t_shared_exponents, x_flat_qt_shared_exponents, 1. / 9.)

        weight_mask = _unpack_mask(weight_mask)
        grad_weight = (
            (grad_weight_hf.view(-1, 32) * weight_mask.view(-1, 32).to(grad_weight_hf.dtype))
            @ forward_hadamard_matrix.T
        ).view(grad_output.size(-1), weight_q.size(-1) * 2)
        
        grad_bias = grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None

        return grad_input, grad_weight, grad_bias, None, None, None


class QuartetNoMasterWeightsFn(Function):
    @staticmethod
    @torch.compile()
    def forward(ctx, x: torch.Tensor, weight_q: torch.Tensor, weight_shared_exponents: torch.Tensor, bias: Optional[torch.Tensor], forward_hadamard_matrix: torch.Tensor, backward_hadamard_matrix: torch.Tensor, dtype: QuartetDtype):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_shared_exponents, x_flat_mask = forward_quantize(x_flat, forward_hadamard_matrix, dtype)

        y = matmul_mxf4_bf16_tn(x_flat_q, weight_q, to_blocked(x_flat_shared_exponents), to_blocked(weight_shared_exponents), 1.)
        
        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if bias is not None:
            y += bias
        
        ctx.x_shape = x.shape
        ctx.dtype = dtype
        ctx.bias_present = bias is not None
        ctx.save_for_backward(
            weight_q,
            weight_shared_exponents,
            x_flat_mask,
            forward_hadamard_matrix,
            backward_hadamard_matrix,
        )
        
        return y
    
    @staticmethod
    @torch.compile()
    def backward(ctx, grad_output: torch.Tensor):
        weight_q, weight_shared_exponents, x_flat_mask, forward_hadamard_matrix, backward_hadamard_matrix = ctx.saved_tensors

        backward_hadamard_matrix = backward_hadamard_matrix * (
            torch.randint(0, 2, (32,), device=backward_hadamard_matrix.device, dtype=backward_hadamard_matrix.dtype)
            * 2. - 1.
        )

        grad_output_q, grad_output_shared_exponents = fusedQuantize_bwd(
            grad_output.flatten(end_dim=-2),
            backward_hadamard_matrix
        )

        weight_qtq, weight_qt_shared_exponents = quartet.backward_qt_bf16(weight_q, weight_shared_exponents, backward_hadamard_matrix, alpha=1.)
        grad_input = matmul_mxf4_bf16_tn(grad_output_q, weight_qtq, to_blocked(grad_output_shared_exponents), to_blocked(weight_qt_shared_exponents), 1. / 9.)

        x_flat_mask = _unpack_mask(x_flat_mask)
        grad_input = (
            (grad_input.view(-1, 32) * x_flat_mask.view(-1, 32).to(grad_input.dtype))
            @ forward_hadamard_matrix.T
        ).view(ctx.x_shape)
        
        grad_bias = grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None

        return grad_input, None, None, grad_bias, None, None, None
