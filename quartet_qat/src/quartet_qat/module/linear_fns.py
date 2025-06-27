from typing import Optional

import torch
from torch import nn
from torch.autograd import Function

from qutlass import matmul_mxf4_bf16_tn, fusedQuantizeMx
from qutlass.utils import to_blocked
import quartet

from ..utils import QuartetDtype


@torch.library.custom_op("quartet::fused_quantize_op", mutates_args=())
def fused_quantize_mx_op(x_flat: torch.Tensor, hadamard_matrix: torch.Tensor, forward_method: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_flat.shape[-1] % 32 != 0:
        raise ValueError(f"x_flat.shape[-1] % 32 != 0: {x_flat.shape}")
    qweight, scales = fusedQuantizeMx(x_flat, hadamard_matrix, method=forward_method)
    return qweight, scales, None


@fused_quantize_mx_op.register_fake
def _(x_flat, hadamard_matrix, forward_method):
    return (
        x_flat.new_empty(x_flat.shape[0], x_flat.shape[1] // 2, dtype=torch.uint8),
        x_flat.new_empty(x_flat.shape[0], x_flat.shape[1] // 32, dtype=torch.uint8),
        x_flat.new_empty(x_flat.shape[0], x_flat.shape[1] // 8, dtype=torch.uint8) if quest else None,
    )


@torch.library.custom_op("quartet::matmul_mxf4_bf16_tn", mutates_args=())
def matmul_mxf4_bf16_tn_op(x: torch.Tensor, w: torch.Tensor, xs: torch.Tensor, ws: torch.Tensor, alpha: float) -> torch.Tensor:
    if x.shape[-1] % 32 != 0:
        raise ValueError(f"x.shape[-1] % 32 != 0: {x.shape}")
    return matmul_mxf4_bf16_tn(x, w, xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu), alpha)


@matmul_mxf4_bf16_tn_op.register_fake
def _(x, w, xs, ws, alpha):
    return x.new_empty(*x.shape[:-1], w.shape[0], dtype=torch.bfloat16)


FP4_GRID = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.bfloat16)


def dequantize_mxf4_bf16(xq: torch.Tensor, exps: torch.Tensor) -> torch.Tensor:
    xq_unpacked = torch.stack([xq&0xF, xq>>4], dim=-1).to(torch.int32)
    x_dequantized = FP4_GRID.to(xq.device)[xq_unpacked]
    
    float_exps = 2**(exps.view(dtype=torch.uint8).to(torch.int32) - 127).to(torch.bfloat16)
    return (x_dequantized.view(-1, 32) * float_exps.view(-1, 1)).view(*xq.shape[:-1], -1)


# @torch.compile()
@torch.inference_mode()
def _unpack_mask(clip_mask: torch.Tensor) -> torch.Tensor:
    clip_mask_unpacked_dq = torch.zeros(*clip_mask.shape[:-1], clip_mask.size(-1) * 8, dtype=torch.bool, device=clip_mask.device)
    for i in range(8):
        clip_mask_unpacked_dq[..., i::8] = (clip_mask >> i) & 1
    return clip_mask_unpacked_dq


def forward_quantize(x: torch.Tensor, hadamard_matrix: torch.Tensor, dtype: QuartetDtype, forward_method: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    match dtype:
        case QuartetDtype.MXFP4:
            return fused_quantize_mx_op(x, hadamard_matrix, forward_method)
        case QuartetDtype.MXFP8:
            raise NotImplementedError("MXFP8 is not supported for forward quantization yet")
        case _:
            raise ValueError(f"Unsupported forward dtype: {dtype}")


class Quartet4x4MasterFn(Function):
    @staticmethod
    # @torch.compile()
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], forward_hadamard_matrix: torch.Tensor, backward_hadamard_matrix: torch.Tensor, dtype: QuartetDtype, forward_method: str):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_scales, x_flat_mask = forward_quantize(x_flat, forward_hadamard_matrix, dtype, forward_method)

        # Quantize weights
        weight_q, weight_scales, weight_mask = forward_quantize(weight, forward_hadamard_matrix, dtype, forward_method)

        y = matmul_mxf4_bf16_tn_op(x_flat_q, weight_q, to_blocked(x_flat_scales), to_blocked(weight_scales), 1. / 9.)
        
        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if bias is not None:
            y += bias
        
        ctx.x_shape = x.shape
        ctx.dtype = dtype
        ctx.bias_present = bias is not None
        ctx.save_for_backward(
            x_flat_q,
            weight_q,
            x_flat_scales,
            weight_scales,
            x_flat_mask,
            weight_mask,
            forward_hadamard_matrix,
            backward_hadamard_matrix,
        )
        
        return y
    
    @staticmethod
    # @torch.compile()
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("Backward pass is not implemented for Quartet4x4MasterFn yet")
        x_flat_q, weight_q, x_flat_scales, weight_scales, x_flat_mask, weight_mask, forward_hadamard_matrix, backward_hadamard_matrix = ctx.saved_tensors

        backward_hadamard_matrix = backward_hadamard_matrix * (
            torch.randint(0, 2, (32,), device=backward_hadamard_matrix.device, dtype=backward_hadamard_matrix.dtype)
            * 2. - 1.
        )

        grad_output_q, grad_output_scales = fusedQuantize_bwd(
            grad_output.flatten(end_dim=-2),
            backward_hadamard_matrix
        )

        weight_qtq, weight_qt_scales = quartet.backward_qt_bf16(weight_q, weight_scales, backward_hadamard_matrix, alpha=1.)
        grad_input = matmul_mxf4_bf16_tn_op(grad_output_q, weight_qtq, to_blocked(grad_output_scales), to_blocked(weight_qt_scales), 1. / 9.)

        x_flat_mask = _unpack_mask(x_flat_mask)
        grad_input = (
            (grad_input.view(-1, 32) * x_flat_mask.view(-1, 32).to(grad_input.dtype))
            @ forward_hadamard_matrix.T
        ).view(ctx.x_shape)

        grad_output_tq, grad_output_t_scales = quartet.backward_t_bf16(grad_output.flatten(end_dim=-2), backward_hadamard_matrix)
        x_flat_qtq, x_flat_qt_scales = quartet.backward_qt_bf16(x_flat_q, x_flat_scales, backward_hadamard_matrix, alpha=1.)
        grad_weight_hf = matmul_mxf4_bf16_tn_op(grad_output_tq, x_flat_qtq, to_blocked(grad_output_t_scales), to_blocked(x_flat_qt_scales), 1. / 9.)

        weight_mask = _unpack_mask(weight_mask)
        grad_weight = (
            (grad_weight_hf.view(-1, 32) * weight_mask.view(-1, 32).to(grad_weight_hf.dtype))
            @ forward_hadamard_matrix.T
        ).view(grad_output.size(-1), weight_q.size(-1) * 2)
        
        grad_bias = grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None

        return grad_input, grad_weight, grad_bias, None, None, None


class Quartet4x4NoMasterFn(Function):
    @staticmethod
    # @torch.compile()
    def forward(ctx, x: torch.Tensor, weight_q: torch.Tensor, weight_scales: torch.Tensor, bias: Optional[torch.Tensor], forward_hadamard_matrix: torch.Tensor, backward_hadamard_matrix: torch.Tensor, dtype: QuartetDtype, forward_method: str):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_scales, x_flat_mask = forward_quantize(x_flat, forward_hadamard_matrix, dtype, forward_method)

        y = matmul_mxf4_bf16_tn_op(x_flat_q, weight_q, to_blocked(x_flat_scales), to_blocked(weight_scales), 1. / 9.)
        
        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if bias is not None:
            y += bias
        
        ctx.x_shape = x.shape
        ctx.dtype = dtype
        ctx.bias_present = bias is not None
        ctx.save_for_backward(
            weight_q,
            weight_scales,
            x_flat_mask,
            forward_hadamard_matrix,
            backward_hadamard_matrix,
        )
        
        return y
    
    @staticmethod
    # @torch.compile()
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("Backward pass is not implemented for Quartet4x4NoMasterFn yet")
        weight_q, weight_scales, x_flat_mask, forward_hadamard_matrix, backward_hadamard_matrix = ctx.saved_tensors

        backward_hadamard_matrix = backward_hadamard_matrix * (
            torch.randint(0, 2, (32,), device=backward_hadamard_matrix.device, dtype=backward_hadamard_matrix.dtype)
            * 2. - 1.
        )

        grad_output_q, grad_output_scales = fusedQuantize_bwd(
            grad_output.flatten(end_dim=-2),
            backward_hadamard_matrix
        )

        weight_qtq, weight_qt_scales = quartet.backward_qt_bf16(weight_q, weight_scales, backward_hadamard_matrix, alpha=1.)
        grad_input = matmul_mxf4_bf16_tn_op(grad_output_q, weight_qtq, to_blocked(grad_output_scales), to_blocked(weight_qt_scales), 1. / 9.)

        x_flat_mask = _unpack_mask(x_flat_mask)
        grad_input = (
            (grad_input.view(-1, 32) * x_flat_mask.view(-1, 32).to(grad_input.dtype))
            @ forward_hadamard_matrix.T
        ).view(ctx.x_shape)
        
        grad_bias = grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None

        return grad_input, None, None, grad_bias, None, None, None


class Quartet4x16MasterFn(Function):
    @staticmethod
    # @torch.compile()
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], forward_hadamard_matrix: torch.Tensor, dtype: QuartetDtype, forward_method: str):
        x_flat = x.contiguous().flatten(end_dim=-2)

        # Quantize input
        x_flat_q, x_flat_scales, x_flat_mask = forward_quantize(x_flat, forward_hadamard_matrix, dtype, forward_method)

        # Quantize weights
        weight_q, weight_scales, weight_mask = forward_quantize(weight, forward_hadamard_matrix, dtype, forward_method)

        y = matmul_mxf4_bf16_tn_op(x_flat_q, weight_q, to_blocked(x_flat_scales), to_blocked(weight_scales), 1. / 9.)
        
        y = y.unflatten(dim=0, sizes=x.shape[:-1])
        if bias is not None:
            y += bias
        
        ctx.x_shape = x.shape
        ctx.dtype = dtype
        ctx.bias_present = bias is not None
        ctx.save_for_backward(
            x_flat_q,
            weight_q,
            x_flat_scales,
            weight_scales,
            x_flat_mask,
            weight_mask,
            forward_hadamard_matrix,
        )
        
        return y
    
    @staticmethod
    # @torch.compile()
    def backward(ctx, grad_output: torch.Tensor):
        x_flat_q, weight_q, x_flat_scales, weight_scales, x_flat_mask, weight_mask, forward_hadamard_matrix = ctx.saved_tensors

        x_flat_dequantized = dequantize_mxf4_bf16(x_flat_q, x_flat_scales) / 3.
        weight_dequantized = dequantize_mxf4_bf16(weight_q, weight_scales) / 3.
        grad_output_flat = grad_output.flatten(end_dim=-2)
        
        grad_input = torch.einsum("...j,ji->...i", grad_output_flat, weight_dequantized) 
        x_flat_mask = _unpack_mask(x_flat_mask)
        grad_input = (
            (grad_input.view(-1, 32) * x_flat_mask.view(-1, 32).to(grad_input.dtype))
            @ forward_hadamard_matrix.T
        ).view(ctx.x_shape)
        
        grad_weight = torch.einsum("...j,...i->ji", grad_output_flat, x_flat_dequantized)
        weight_mask = _unpack_mask(weight_mask)
        grad_weight = (
            (grad_weight.view(-1, 32) * weight_mask.view(-1, 32).to(grad_weight.dtype))
            @ forward_hadamard_matrix.T
        ).view(grad_output.size(-1), weight_q.size(-1) * 2)
        
        grad_bias = grad_output.flatten(end_dim=-2).sum(dim=0) if ctx.bias_present else None

        return grad_input, grad_weight, grad_bias, None, None
