import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from fast_hadamard_transform import hadamard_transform

from .base import BaseQuantizer, OPTIMAL_GAUSSIAN_SCALES


class UniformQuantizer(BaseQuantizer):
    def forward(self, x):
        scale = torch.max(torch.abs(x), dim=-1, keepdim=True)[0] + 1e-8
        step = scale * 2 / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        return x + (xq - x).detach()

class HalfHadamardUniformQuantizer(BaseQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def forward(self, x):
        x_had = (x.reshape(-1, 128) @ self.aux_matrix.to(x.device).to(x.dtype)).reshape(x.shape)
        with torch.no_grad():
            scale = torch.max(torch.abs(x_had), dim=-1, keepdim=True)[0] + 1e-8
            step = scale * 2 / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        return x_had + (xq - x_had).detach()


class LSQQuantizer(nn.Module):
    """
    Implementation of LSQ quantizer from https://arxiv.org/abs/1902.08153
    LSQ uses a learnable step size for quantization. This learnable step size(alpha) is initialized using the optimal gaussian scale
    ans must be normalized with a weight decay.
    """

    def __init__(self, bits=4, raise_zero=True, all_positive=False, **kwargs):
        super().__init__()
        # NOTE: raise_zero should never be used with FP quantization

        self.bits = bits
        self.n_levels = 2**bits
        self.all_positive = all_positive
        self.raise_zero = raise_zero

        self.q_min, self.q_max = self.get_dtype_bounds()

        self.is_alpha_init = False
        self.alpha_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def get_dtype_bounds(self):
        if not self.all_positive:
            q_min = -self.n_levels / 2
            q_max = self.n_levels / 2 - 1
        else:
            q_min = 0
            q_max = self.n_levels - 1
        return q_min, q_max

    def cast(self, x):
        # This method can be inherited to use any casting, e.g. int, fp(e2m1, e1m2,...), optimal gaussian, etc.
        # NOTE: raise_zero should never be used with FP quantization
        return x.round()

    def ste_cast(self, x):
        return (self.cast(x) - x).detach() + x

    def grad_scale(self, x, scale):
        return (x - x * scale).detach() + x * scale

    @torch.no_grad()
    def get_initial_step_value(self, x):
        return (
            torch.mean(torch.abs(x.detach())) * 2 / (np.sqrt(self.q_max))
        )  # LSQ initialization

    def get_learnable_step(self, x):
        if not self.is_alpha_init:
            with torch.no_grad():
                step = self.get_initial_step_value(x)
                self.alpha_weight.data.multiply_(
                    torch.tensor(
                        step,
                        dtype=self.alpha_weight.dtype,
                        device=self.alpha_weight.device,
                    )
                )
            self.is_alpha_init = True
        return self.alpha_weight

    def forward(self, x):
        step = self.get_learnable_step(x)
        step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
        xs = x / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc)
        xq = xscr * step

        return xq + step * 1e-9  # extra term to ensure gradient flow


class LSQPlusWeightQuantizer(LSQQuantizer):
    @torch.no_grad()
    def get_initial_step_value(self, x):
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * torch.sqrt(torch.mean(x**2)) + 1e-8
        step = 2 * scale / (self.n_levels - 1)
        return step


class LSQPlusActivationQuantizer(LSQPlusWeightQuantizer):
    def __init__(self, bits=4, raise_zero=True, all_positive=False, **kwargs):
        super().__init__(bits, raise_zero, all_positive, **kwargs)
        self.beta_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.is_beta_init = False

    @torch.no_grad()
    def get_initial_bias_value(self, x):
        return x.min() - self.alpha_weight * self.q_min

    def get_learnable_bias(self, x):
        if not self.is_beta_init:
            with torch.no_grad():
                bias = self.get_initial_bias_value(x)
                self.beta_weight.data.add_(
                    torch.tensor(
                        bias,
                        dtype=self.beta_weight.dtype,
                        device=self.beta_weight.device,
                    )
                )
            self.is_beta_init = True
        return self.beta_weight

    def forward(self, x):
        step = self.get_learnable_step(x)
        step = self.grad_scale(step, 1.0 / np.sqrt(x.numel() * self.q_max))
        bias = self.get_learnable_bias(x)
        bias = self.grad_scale(bias, 1.0 / np.sqrt(x.numel() * self.q_max))
        xs = (x - bias) / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            xscr = self.ste_cast(xsc)
        xq = xscr * step + bias
        return xq + step * 1e-9  # extra term to ensure gradient flow


class PACTQuantizer(LSQQuantizer):
    """
    Implementation of PACT quantizer from https://arxiv.org/abs/1805.06085
    PACT and LSQ are quite similar and do the same thing for forward pass.
    The difference is in the backward pass where PACT does not perform a full gradient flow.
    """

    def forward(self, x):
        step = self.get_learnable_step(x)
        xs = x / step
        if self.raise_zero:
            xsc = torch.clamp(xs - 1 / 2, self.q_min, self.q_max)
            with torch.no_grad():
                clamp_mask = ~torch.isclose(xsc, xs - 1 / 2)
            xscr = self.ste_cast(xsc) + 1 / 2
        else:
            xsc = torch.clamp(xs, self.q_min, self.q_max)
            with torch.no_grad():
                clamp_mask = ~torch.isclose(xsc, xs)
            xscr = self.ste_cast(xsc)
        xq = xscr * step
        xq = xq * clamp_mask + (xq - xq * clamp_mask).detach()
        return xq + step * 1e-9  # extra term to ensure gradient flow
