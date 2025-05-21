import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .base import BaseQuantizer

class BinaryQuantize(Function):
    '''
        binary quantize function, from IR-Net
        (https://github.com/htqin/IR-Net/blob/master/CIFAR-10/ResNet20/1w1a/modules/binaryfunction.py)
    ''' 
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input)
        ctx.k = k
        ctx.t = t
        out = torch.sign(input)
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        k, t = ctx.k, ctx.t
        grad_input = k * t * (1-torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class AdaBinWeightQuantizer(BaseQuantizer):
    def __init__(self, k=10, t=0.1):
        super().__init__(1)
        self.k = k
        self.t = t

    @torch.compiler.disable()
    def forward(self, w):
        beta_w = w.mean(dim=1, keepdim=True)
        alpha_w = torch.sqrt(((w - beta_w) ** 2).sum(dim=1) / w.shape[1]).view(-1, 1)
        w_normalized = (w - beta_w) / alpha_w
        wb = BinaryQuantize.apply(w_normalized, self.k, self.t)
        weight = wb * alpha_w + beta_w
        return weight


class AdaBinActivationQuantizer(BaseQuantizer):
    def __init__(self, alpha=1.0, beta=0.0):
        super().__init__(1)
        self.alpha = alpha
        self.beta = beta
        
    def gradient_approx(self, x):
        out_forward = torch.sign(x)
        
        # Use nested torch.where to define the differentiable surrogate:
        # f(x) = -1         if x < -1
        #      = x*x + 2*x   if -1 <= x < 0
        #      = -x*x + 2*x  if 0 <= x < 1
        #      = 1           if x >= 1
        piecewise = torch.where(
            x < -1, -1,
            torch.where(
                x < 0, x * x + 2 * x,
                torch.where(
                    x < 1, -x * x + 2 * x,
                    1,
                )
            )
        )
        
        out = piecewise - piecewise.detach() + out_forward.detach() 
        return out


    def forward(self, x):
        x = (x-self.beta)/self.alpha
        x = self.gradient_approx(x)
        return self.alpha*(x + self.beta)
