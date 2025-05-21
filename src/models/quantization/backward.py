import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .quantizers import NoQuantizer

# ===== EW_EtX =====
class EW_EtX_Scheme(nn.Module):
    def __init__(self):
        super().__init__()
        self.g_quantizer = None

    def forward(self, x, w):
        return F.linear(x, w, None)


# ===== Q(E)W_Q(Et)X =====
class QEW_QEtXFn(Function):
    @staticmethod
    def forward(ctx, x, w, g_quantizer):
        ctx.save_for_backward(x, w)
        ctx.g_quantizer = g_quantizer
        return F.linear(x, w, None)
    
    @staticmethod
    def backward(ctx, e):
        x, w = ctx.saved_tensors

        grad_x = F.linear(
            ctx.g_quantizer(e),
            w.T,
            None,
        ) # Q(E)W

        batch_seq_dim = math.prod(x.shape[:-1])
        grad_w = torch.einsum(
            "ib,bj->ij",
            ctx.g_quantizer(e.reshape(batch_seq_dim, -1).T.contiguous()),
            x.reshape(batch_seq_dim, -1),
        ) # Q(Et)X

        return grad_x, grad_w, None


class QEW_QEtX_Scheme(EW_EtX_Scheme):
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        return QEW_QEtXFn.apply(x, w, self.g_quantizer)


# ===== Q(E)Q(Wt)t_Q(Et)Q(Xt)t =====
class QEQWtt_QEtQXttFn(Function):
    @staticmethod
    def forward(ctx, x, w, g_quantizer):
        ctx.save_for_backward(x, w)
        ctx.g_quantizer = g_quantizer
        return F.linear(x, w, None)
    
    @torch.compile
    @staticmethod
    def backward(ctx, e):
        x, w = ctx.saved_tensors

        ctx.g_quantizer.re_randomize()
        
        grad_x = F.linear(
            ctx.g_quantizer(e),
            ctx.g_quantizer(w.T.contiguous()),
            None,
        ) # Q(E)Q(W)


        batch_seq_dim = math.prod(x.shape[:-1])
        grad_w = torch.einsum(
            "ib,jb->ij",
            ctx.g_quantizer(e.reshape(batch_seq_dim, -1).T.contiguous()),
            ctx.g_quantizer(x.reshape(batch_seq_dim, -1).T.contiguous()),
        ) # Q(Et)Q(Xt)t

        return grad_x, grad_w, None


class QEQWtt_QEtQXtt_Scheme(QEW_QEtX_Scheme):
    def __init__(self):
        super().__init__()

    def forward(self, x, w):
        return QEQWtt_QEtQXttFn.apply(x, w, self.g_quantizer)


# ===== BACKWARD SCHEMES =====
BACKWARD_SCHEMES = {
    "EW_EtX": EW_EtX_Scheme,
    "Q(E)W_Q(Et)X": QEW_QEtX_Scheme,
    "Q(E)Q(Wt)t_Q(Et)Q(Xt)t": QEQWtt_QEtQXtt_Scheme,
}
