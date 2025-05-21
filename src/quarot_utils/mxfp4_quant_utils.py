import math
import transformers
import torch
from .utils import *
from .hadamard_utils import *
from fast_hadamard_transform import hadamard_transform


import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32 * 32}),
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)

@triton.jit
def mxfp4_quant_kernel(
    x_ptr,
    output_ptr,
    clip_mask_ptr,
    n_elements: tl.constexpr,
    group_size: tl.constexpr,
    quest: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):    
    
    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)
    

    
    # group
    x_grouped = tl.reshape(x_flat, (BLOCK_SIZE // group_size, group_size))
    
    # scale
    if quest:
        mean_squared = tl.sum(x_grouped * x_grouped, axis=-1, keep_dims=True) / group_size
        mean = tl.sum(x_grouped, axis=-1, keep_dims=True) / group_size
        std = tl.sqrt(mean_squared - mean * mean)
        scales = (2.92247856 / 6.0) * std + 1e-8
        shared_exps = tl.exp2(tl.floor(tl.log2(scales)))
        x_had_scaled = x_grouped / shared_exps
    else:
        scales = tl.max(tl.abs(x_grouped), axis=-1, keep_dims=True)
        shared_exps = tl.exp2(tl.floor(tl.log2(scales)) - 2) 
        x_had_scaled = x_grouped / shared_exps * (3/4) # 3/4 is constant. In CUDA, scale the GEMM output by 16/9
    
    # quantize
    x_had_scaled_abs = tl.abs(x_had_scaled)
    x_had_scaled_sign = tl.where(
        x_had_scaled > 0,
        1,
        -1,
    )
  
    x_fp4 = tl.where(
        x_had_scaled_abs > 5,
        6,
        tl.where(
            x_had_scaled_abs > 3.5,
            4,
            tl.where(
                x_had_scaled_abs > 2.5,
                3,
                tl.where(
                    x_had_scaled_abs > 1.75,
                    2,
                    tl.where(
                        x_had_scaled_abs > 1.25,
                        1.5,
                        tl.where(
                            x_had_scaled_abs > 0.75,
                            1,
                            tl.where(
                                x_had_scaled_abs > 0.25,
                                0.5,
                                0,
                            )
                        )
                    )
                )
            )
        )
    ) * x_had_scaled_sign

    
    # dequantize
    if quest:
        x_dequantized = x_fp4 * shared_exps
        tl.store(
            clip_mask_ptr + offsets,
            tl.reshape(x_had_scaled_abs < 6, (BLOCK_SIZE,)),
            mask=mask
        )
    else:
        x_dequantized = x_fp4 * shared_exps * (4/3) # 3/4 is constant. In CUDA, scale the GEMM output by 16/9
    
    # Reshape back to flat form for storage
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))
    
    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)







