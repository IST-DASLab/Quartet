import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import integrate
from scipy.stats import norm

from fast_hadamard_transform import hadamard_transform

from .base import BaseQuantizer, OPTIMAL_GAUSSIAN_SCALES


class STEQuantizer(BaseQuantizer):
    def __init__(self, bits=4, centered=True):
        super().__init__(bits)
        self.centered = centered

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, -scale * (self.n_levels - 2) / self.n_levels, scale)
            xq = torch.round(x_clip / step) * step

        return x + (xq - x).detach()


class ClipQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0):
        super().__init__(bits, centered)
        self.clip_scale = clip_scale

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            mask = (torch.abs(x) <= scale * self.clip_scale).float()
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step
            mask = (
                (neg_scale * self.clip_scale <= x) & (x <= scale * self.clip_scale)
            ).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardClipQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0, hadamard_dim=128):
        super().__init__(bits, centered)
        self.matrix = None
        self.clip_scale = clip_scale
        
        self.hadamard_dim = hadamard_dim
        self.aux_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.bfloat16, device="cuda"), scale=hadamard_dim**-0.5
        )

    def forward(self, x):
        self.aux_matrix = self.aux_matrix.to(x.device).to(x.dtype)
        x_had = (x.view(-1, self.hadamard_dim) @ self.aux_matrix.T).view_as(x)
        with torch.no_grad():
            scale = (
                OPTIMAL_GAUSSIAN_SCALES[self.bits]
                * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
                + 1e-8
            )
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
                mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
                mask = (
                    (neg_scale * self.clip_scale <= x_had)
                    & (x_had <= scale * self.clip_scale)
                ).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardClipQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, clip_scale: float = 1.0, hadamard_dim=128):
        super().__init__(bits, centered)
        self.matrix = None
        self.clip_scale = clip_scale
        
        self.hadamard_dim = hadamard_dim
        self.aux_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.bfloat16, device="cuda"), scale=hadamard_dim**-0.5
        )

    def forward(self, x):
        self.aux_matrix = self.aux_matrix.to(x.device).to(x.dtype)
        x_had = (x.view(-1, self.hadamard_dim) @ self.aux_matrix.T).view_as(x)
        with torch.no_grad():
            scale = (
                OPTIMAL_GAUSSIAN_SCALES[self.bits]
                * torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
                + 1e-8
            )
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
                mask = (torch.abs(x_had) <= scale * self.clip_scale).float()
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
                mask = (
                    (neg_scale * self.clip_scale <= x_had)
                    & (x_had <= scale * self.clip_scale)
                ).float()
            xq = (xq.view(-1, self.hadamard_dim) @ self.aux_matrix).view_as(x)

        grad_flow_output = (x_had * mask).view(-1, self.hadamard_dim) @ self.aux_matrix

        return grad_flow_output + (xq - grad_flow_output).detach()


class MaskedSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_had, scale, step, trust, std):
        x_clip = torch.clamp(x_had, -scale, scale)
        xq = torch.round(x_clip / step + 0.5) * step - 0.5 * step
        mask = (torch.abs(xq - x_had) <= std * trust)
        ctx.save_for_backward(mask)
        return xq

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        grad_input = torch.where(mask, grad_output, 0)
        return grad_input, None, None, None, None


class HalfHadamardTrustQuantizer(STEQuantizer):
    def __init__(self, bits=4, trust=None, hadamard_dim=128):
        super().__init__(bits, True)
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust
        
        self.hadamard_dim = hadamard_dim
        self.aux_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.bfloat16, device="cuda"), scale=hadamard_dim**-0.5
        )

    def forward(self, x):
        self.aux_matrix = self.aux_matrix.to(x.device).to(x.dtype)

        x_had = (x.view(-1, self.hadamard_dim) @ self.aux_matrix.T).view_as(x)
        with torch.no_grad():
            std = torch.std(x_had, dim=-1, correction=0, keepdim=True)
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            step = 2 * scale / (self.n_levels - 1)
        return MaskedSTEQuantize.apply(x_had, scale, step, self.trust, std)


class TrustQuantizer(STEQuantizer):
    def __init__(self, bits=4, centered=True, trust=None):
        super().__init__(bits, centered)

        # in terms of std
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
        if self.centered:
            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
        else:
            neg_scale = -scale * (self.n_levels - 2)
            step = 2 * scale / self.n_levels
            x_clip = torch.clamp(x, neg_scale, scale)
            xq = torch.round(x_clip / step) * step

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HadamardTrustQuantizer(TrustQuantizer):
    def __init__(self, bits=4, trust=None, hadamard_dim=128):
        super().__init__(bits, True, trust)
        self.hadamard_dim = hadamard_dim
        self.aux_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.bfloat16, device="cuda"), scale=hadamard_dim**-0.5
        )

    def forward(self, x):
        self.aux_matrix = self.aux_matrix.to(x.device).to(x.dtype)

        x_had = (x.view(-1, self.hadamard_dim) @ self.aux_matrix.T).view_as(x)
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True))
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8
            if self.centered:
                step = 2 * scale / (self.n_levels - 1)
                x_clip = torch.clamp(x_had, -scale, scale)
                xq = torch.round(x_clip / step + 1 / 2) * step - step / 2
            else:
                neg_scale = -scale * (self.n_levels - 2)
                step = 2 * scale / self.n_levels
                x_clip = torch.clamp(x_had, neg_scale, scale)
                xq = torch.round(x_clip / step) * step
            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = (xq.view(-1, self.hadamard_dim) @ self.aux_matrix).view_as(x)

        grad_flow_output = ((x_had * mask).view(-1, self.hadamard_dim) @ self.aux_matrix).view_as(x)

        return grad_flow_output + (xq - grad_flow_output).detach()


class GaussianSTEQuantizer(BaseQuantizer):
    def __init__(self, bits=4):
        super().__init__(bits)
        self.register_buffer("levels", self._compute_gaussian_levels())

    def _compute_gaussian_levels(self):
        levels = np.linspace(-3, 3, self.n_levels)
        boundaries = np.zeros(self.n_levels + 1)

        for _ in range(20):
            boundaries[1:-1] = (levels[1:] + levels[:-1]) / 2
            boundaries[0] = -float("inf")
            boundaries[-1] = float("inf")

            new_levels = []
            for i in range(self.n_levels):
                b_left, b_right = boundaries[i], boundaries[i + 1]

                def f(x):
                    return x * norm.pdf(x)

                integral_num = integrate.quad(f, b_left, b_right)[0]
                integral_den = integrate.quad(norm.pdf, b_left, b_right)[0]
                if integral_den > 1e-10:
                    new_levels.append(integral_num / integral_den)
                else:
                    new_levels.append(levels[i])
            levels = np.array(new_levels)
        return torch.tensor(levels, dtype=torch.float32)

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        return x + (xq - x).detach()


class GaussianClipQuantizer(GaussianSTEQuantizer):
    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        mask = (x_norm.abs() <= self.levels[-1]).float()
        return x * mask + (xq - x * mask).detach()


class GaussianTrustQuantizer(GaussianSTEQuantizer):
    def __init__(self, bits=4, trust=None):
        super().__init__(bits)
        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-8
        x_norm = x / std
        expanded_input = x_norm.unsqueeze(-1)
        distances = torch.abs(expanded_input - self.levels)
        indices = torch.argmin(distances, dim=-1)
        xq_norm = self.levels[indices]
        xq = xq_norm * std

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardGaussianClipQuantizer(GaussianClipQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4):
        super().__init__(bits)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (x_norm.abs() <= self.levels[-1]).float()

        grad_flow_output = x_had * mask

        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardGaussianClipQuantizer(GaussianClipQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4):
        super().__init__(bits)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            xq = xq @ self.matrix.T
            mask = (x_norm.abs() <= self.levels[-1]).float()

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


class HalfHadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask
        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardGaussianTrustQuantizer(GaussianTrustQuantizer):
    aux_matrix = hadamard_transform(
        torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=2 ** (-7 / 2)
    )

    def __init__(self, bits=4, trust=None):
        super().__init__(bits, trust)
        self.matrix = None

    def forward(self, x):
        if self.matrix is None:
            self.matrix = torch.block_diag(
                *[self.aux_matrix.to(x.device).to(x.dtype)] * (x.shape[-1] // 128),
            )

        x_had = x @ self.matrix
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            x_norm = x_had / std
            expanded_input = x_norm.unsqueeze(-1)
            distances = torch.abs(expanded_input - self.levels)
            indices = torch.argmin(distances, dim=-1)
            xq_norm = self.levels[indices]
            xq = xq_norm * std

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = xq @ self.matrix.T

        grad_flow_output = (x_had * mask) @ self.matrix.T

        return grad_flow_output + (xq - grad_flow_output).detach()


FP4_LEVELS = [
    -2.92247856,
    -1.94831904,
    -1.46123928,
    -0.97415952,
    -0.73061964,
    -0.48707976,
    -0.24353988,
    0.0,
    0.0,
    0.24353988,
    0.48707976,
    0.73061964,
    0.97415952,
    1.46123928,
    1.94831904,
    2.92247856,
]


class FP4STEQuantizer(GaussianSTEQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer("levels", torch.tensor(FP4_LEVELS))


class FP4ClipQuantizer(GaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer("levels", torch.tensor(FP4_LEVELS))


class FP4TrustQuantizer(GaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer("levels", torch.tensor(FP4_LEVELS))


class HalfHadamardFP4ClipQuantizer(HalfHadamardGaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer("levels", torch.tensor(FP4_LEVELS))


class HadamardFP4ClipQuantizer(HadamardGaussianClipQuantizer):
    def __init__(self):
        super().__init__(4)
        self.register_buffer("levels", torch.tensor(FP4_LEVELS))


class HalfHadamardFP4TrustQuantizer(HalfHadamardGaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer("levels", torch.tensor(FP4_LEVELS))

        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2


class HadamardFP4TrustQuantizer(HadamardGaussianTrustQuantizer):
    def __init__(self, trust=None):
        super().__init__(4, trust)
        self.register_buffer("levels", torch.tensor(FP4_LEVELS))

        if trust is None:
            trust = (self.levels[-1] - self.levels[-2]) / 2


class FourEightMaskedQuantizer(BaseQuantizer):
    def __init__(self, p=2.0):
        super().__init__(16)
        self.p = p

    def forward(self, x):
        x_reshaped = x.reshape(-1, 4, 2)
        _, idx = x_reshaped.norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        mask = torch.ones_like(x_reshaped, dtype=torch.bool)
        mask[torch.arange(x_reshaped.size(0)).repeat(2, 1).T, idx, :] = False
        mask = mask.reshape(x.shape).float()

        return x * mask


class FourEightSTEQuantizer(BaseQuantizer):
    def __init__(self, bits=4, p: float = 2.0):
        super().__init__(bits)
        self.p = p

    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        return x + (xq - x).detach()


class FourEightClipQuantizer(FourEightSTEQuantizer):
    def forward(self, x):
        scale = (
            OPTIMAL_GAUSSIAN_SCALES[self.bits]
            * torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
            + 1e-8
        )

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        mask = (torch.abs(x) <= scale).float()
        return x * mask + (xq - x * mask).detach()


class FourEightTrustQuantizer(FourEightSTEQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0):
        super().__init__(bits, p)
        if trust is None:
            trust = OPTIMAL_GAUSSIAN_SCALES[self.bits] / (self.n_levels - 1)
        self.trust = trust

    def forward(self, x):
        std = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std + 1e-8

        step = 2 * scale / (self.n_levels - 1)
        x_clip = torch.clamp(x, -scale, scale)
        xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

        _, idx = (
            x.reshape(-1, 4, 2).norm(p=self.p, dim=-1).topk(k=2, dim=-1, largest=False)
        )
        xq = xq.reshape(-1, 4, 2)
        xq[
            torch.arange(xq.size(0)).repeat(2, 1).T,
            idx,
        ] = 0.0
        xq = xq.reshape(x.shape)

        mask = (torch.abs(xq - x) <= std * self.trust).float()
        return x * mask + (xq - x * mask).detach()


class HalfHadamardFourEightTrustQuantizer(HadamardTrustQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0, hadamard_dim=128):
        super().__init__(bits, trust)
        self.p = p
        
        self.hadamard_dim = hadamard_dim
        self.aux_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.bfloat16, device="cuda"), scale=hadamard_dim**-0.5
        )

    def forward(self, x):
        self.aux_matrix = self.aux_matrix.to(x.device).to(x.dtype)

        x_had = (x.view(-1, self.hadamard_dim) @ self.aux_matrix.T).view_as(x)
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std

            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

            _, idx = (
                x_had.reshape(-1, 4, 2)
                .norm(p=self.p, dim=-1)
                .topk(k=2, dim=-1, largest=False)
            )
            xq = xq.reshape(-1, 4, 2)
            xq[
                torch.arange(xq.size(0)).repeat(2, 1).T,
                idx,
            ] = 0.0
            xq = xq.reshape(x.shape)

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask

        return grad_flow_output + (xq - grad_flow_output).detach()
    
    
class HalfHadamardTwoFourTrustQuantizer(HadamardTrustQuantizer):
    def __init__(self, bits=8, trust=None):
        super().__init__(bits, trust)

    def forward(self, x):
        self.aux_matrix = self.aux_matrix.to(x.device).to(x.dtype)

        x_had = (x.view(-1, 128) @ self.aux_matrix.T).view(x.shape)
        
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std

            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

            _, idx = (
                x_had.reshape(-1, 4)
                .abs()
                .topk(k=2, dim=-1, largest=False)
            )
            xq = xq.reshape(-1, 4)
            batch_indices = torch.arange(xq.size(0)).unsqueeze(1).expand(-1, 2)
            xq[batch_indices.reshape(-1), idx.reshape(-1)] = 0.0
            xq = xq.reshape(x.shape)

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()

        grad_flow_output = x_had * mask

        return grad_flow_output + (xq - grad_flow_output).detach()


class HadamardFourEightTrustQuantizer(HadamardTrustQuantizer):
    def __init__(self, bits=4, trust=None, p: float = 2.0, hadamard_dim=128):
        super().__init__(bits, trust)
        self.p = p
        
        self.hadamard_dim = hadamard_dim
        self.aux_matrix = hadamard_transform(
            torch.eye(hadamard_dim, dtype=torch.bfloat16, device="cuda"), scale=hadamard_dim**-0.5
        )

    def forward(self, x):
        self.aux_matrix = self.aux_matrix.to(x.device).to(x.dtype)

        x_had = (x.view(-1, self.hadamard_dim) @ self.aux_matrix.T).view_as(x)
        with torch.no_grad():
            std = torch.sqrt(torch.mean(x_had**2, dim=-1, keepdim=True)) + 1e-8
            scale = OPTIMAL_GAUSSIAN_SCALES[self.bits] * std

            step = 2 * scale / (self.n_levels - 1)
            x_clip = torch.clamp(x_had, -scale, scale)
            xq = torch.round(x_clip / step + 1 / 2) * step - step / 2

            _, idx = (
                x_had.reshape(-1, 4, 2)
                .norm(p=self.p, dim=-1)
                .topk(k=2, dim=-1, largest=False)
            )
            xq = xq.reshape(-1, 4, 2)
            xq[
                torch.arange(xq.size(0)).repeat(2, 1).T,
                idx,
            ] = 0.0
            xq = xq.reshape(x.shape)

            mask = (torch.abs(xq - x_had) <= std * self.trust).float()
            xq = (xq.view(-1, self.hadamard_dim) @ self.aux_matrix).view_as(x)

        grad_flow_output = (x_had * mask).view(-1, self.hadamard_dim) @ self.aux_matrix

        return grad_flow_output + (xq - grad_flow_output).detach()
