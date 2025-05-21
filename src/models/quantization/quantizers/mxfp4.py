import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm

from fast_hadamard_transform import hadamard_transform

from models.quantization.quantizers.base import BaseQuantizer
from models.quantization.quantizers.mxfp4_triton import mxfp4_forward_kernel_wrapper


class AlbertTsengQuantizerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hadamard_matrix, stochastic_round):
        x_dequantized, _ = mxfp4_forward_kernel_wrapper(
            x,
            hadamard_matrix,
            return_clip_mask=False,
            stochastic_round=stochastic_round,
            quest=False,
            gaussian_scale=3/4,
        )
        ctx.save_for_backward(hadamard_matrix)
        ctx.x_shape = x.shape
        return x_dequantized
    
    @staticmethod
    def backward(ctx, grad_output):
        hadamard_matrix, = ctx.saved_tensors
        grad_input = grad_output.view(-1, hadamard_matrix.shape[0]) @ hadamard_matrix.T
        return grad_input.view(ctx.x_shape), None, None


class AlbertTsengQuantizer(BaseQuantizer):
    def __init__(self, hadamard_dim=32, stochastic=True, rerotate=None):
        super().__init__(4)
        
        self.hadamard_dim = hadamard_dim
        self.hadamard_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.float32, device="cuda"), scale=hadamard_dim**-0.5
        )
        self.rerotate = rerotate
        self.stochastic = stochastic

    def forward(self, x):
        self.hadamard_matrix = self.hadamard_matrix.to(x.device).to(x.dtype)
        return AlbertTsengQuantizerFn.apply(x, self.hadamard_matrix, self.stochastic)

    def re_randomize(self):
        if self.rerotate == "signs":
            self.hadamard_matrix = self.hadamard_matrix @ torch.diag(
                torch.randint(
                    0, 2, (self.hadamard_dim,),
                    device=self.hadamard_matrix.device,
                    dtype=self.hadamard_matrix.dtype
                ) * 2 - 1
            )
        elif self.rerotate == "O32":
            gaussian_matrix = torch.randn(self.hadamard_dim, self.hadamard_dim, device=self.hadamard_matrix.device, dtype=self.hadamard_matrix.dtype)
            svd = torch.linalg.svd(gaussian_matrix)
            self.hadamard_matrix = svd[0] @ svd[2]
        elif self.rerotate is None:
            pass
        else:
            raise ValueError(f"Invalid rerotate value: {self.rerotate}")


class AlignedAlbertTsengQuantizerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hadamard_matrix):
        x_dequantized, _ = mxfp4_forward_kernel_wrapper(
            x,
            hadamard_matrix,
            return_clip_mask=False,
            stochastic_round=False,
            quest=False,
            gaussian_scale=3/4,
        )
        x_dequantized *= 1.009276
        ctx.save_for_backward(hadamard_matrix)
        ctx.x_shape = x.shape
        return x_dequantized
    
    @staticmethod
    def backward(ctx, grad_output):
        hadamard_matrix, = ctx.saved_tensors
        grad_input = grad_output.view(-1, hadamard_matrix.shape[0]) @ hadamard_matrix.T
        return grad_input.view(ctx.x_shape), None


class AlignedAlbertTsengQuantizer(BaseQuantizer):
    def __init__(self, hadamard_dim=32, rerotate=None):
        super().__init__(4)
        
        self.hadamard_dim = hadamard_dim
        self.hadamard_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.float32, device="cuda"), scale=hadamard_dim**-0.5
        )
        self.rerotate = rerotate

    def forward(self, x):
        self.hadamard_matrix = self.hadamard_matrix.to(x.device).to(x.dtype)
        return AlignedAlbertTsengQuantizerFn.apply(x, self.hadamard_matrix)
    
    def re_randomize(self):
        if self.rerotate == "signs":
            self.hadamard_matrix = self.hadamard_matrix @ torch.diag(
                torch.randint(
                    0, 2, (self.hadamard_dim,),
                    device=self.hadamard_matrix.device,
                    dtype=self.hadamard_matrix.dtype
                ) * 2 - 1
            )
        elif self.rerotate == "O32":
            gaussian_matrix = torch.randn(self.hadamard_dim, self.hadamard_dim, device=self.hadamard_matrix.device, dtype=self.hadamard_matrix.dtype)
            svd = torch.linalg.svd(gaussian_matrix)
            self.hadamard_matrix = svd[0] @ svd[2]
        elif self.rerotate is None:
            pass
        else:
            raise ValueError(f"Invalid rerotate value: {self.rerotate}")


class QuestMXFP4QuantizerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hadamard_matrix):
        x_dequantized, mask = mxfp4_forward_kernel_wrapper(
            x,
            hadamard_matrix,
            return_clip_mask=True,
            stochastic_round=False,
            quest=True,
            gaussian_scale=2.92247856 / 6.0,
        )
        ctx.save_for_backward(hadamard_matrix, mask)
        ctx.x_shape = x.shape
        return x_dequantized
    
    @staticmethod
    def backward(ctx, grad_output):
        hadamard_matrix, mask = ctx.saved_tensors
        grad_input = (grad_output * mask.to(grad_output.dtype)).view(-1, hadamard_matrix.shape[0]) @ hadamard_matrix.T
        return grad_input.view(ctx.x_shape), None, None
    

class QuestMXFP4Quantizer(BaseQuantizer):
    def __init__(self, hadamard_dim=32, rerotate=None):
        super().__init__(4)
        
        self.hadamard_dim = hadamard_dim
        self.hadamard_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.float32, device="cuda"), scale=hadamard_dim**-0.5
        )
        self.rerotate = rerotate

    def forward(self, x):
        self.hadamard_matrix = self.hadamard_matrix.to(x.device).to(x.dtype)
        return QuestMXFP4QuantizerFn.apply(x, self.hadamard_matrix)

    def re_randomize(self):
        if self.rerotate == "signs":
            self.hadamard_matrix = self.hadamard_matrix @ torch.diag(
                torch.randint(
                    0, 2, (self.hadamard_dim,),
                    device=self.hadamard_matrix.device,
                    dtype=self.hadamard_matrix.dtype
                ) * 2 - 1
            )
        elif self.rerotate == "O32":
            gaussian_matrix = torch.randn(self.hadamard_dim, self.hadamard_dim, device=self.hadamard_matrix.device, dtype=self.hadamard_matrix.dtype)
            svd = torch.linalg.svd(gaussian_matrix)
            self.hadamard_matrix = svd[0] @ svd[2]
        elif self.rerotate is None:
            pass
        else:
            raise ValueError(f"Invalid rerotate value: {self.rerotate}")
