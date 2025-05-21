import math
import transformers
import torch
from .utils import *
from .hadamard_utils import *
from fast_hadamard_transform import hadamard_transform

def get_int_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def two_compl(x, bits: int):
    return torch.where(x < 0, 2 ** bits + x, x)

def mxfp4_sym_quant_dequant(x, quest=False):
    import triton
    from .mxfp4_quant_utils import mxfp4_quant_kernel
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    # Launch optimized kernel
    mxfp4_quant_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        clip_mask_ptr=None,
        n_elements=n_elements,
        group_size=32,
        quest=False,
    )
    return output


class ActQuantizer(torch.nn.Module):

    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16

    def free(self):
        self.zero = None
        self.scale = None

    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        if self.format == 'mxfp4':
            return mxfp4_sym_quant_dequant(x, quest=False).to(x_dtype)
        elif self.sym:
            return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
        return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits, format='fp', groupsize=-1, sym=False, clip_ratio=1.0):
        if format == 'int':
            _, self.maxq = get_int_minq_maxq(bits, sym)
        
        if format == 'mxfp4': assert bits == 4, 'MXFP4 quantization only supports 4 bits!'
        self.format = format
        self.bits = bits
        
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16 or self.format == 'mxfp4':
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            cleanup_memory(verbos=False)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, module:torch.nn.Linear):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        return str_

    def forward(self, x):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:
            
            if self.fp32_had: # Full Hadamard in FP32
                x = matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else: # Full Hadamard in FP16
                x = matmul_hadU_cuda(x, self.had_K, self.K)
            
        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!
            
            if self.fp32_had:
                x = x.float()
                
            init_shape = x.shape
            if self.K == 1:
                # x = fast_hadamard_transform.hadamard_transform(x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim).transpose(1, 2),
                #                                                scale=1/math.sqrt(init_shape[-1]//self.had_dim)).transpose(1, 2)
                aux_matrix = hadamard_transform(
                                    torch.eye(self.had_dim, dtype=x.dtype, device="cuda"),
                                    scale=1/math.sqrt(self.had_dim)
                         )
                matrix = torch.block_diag(
                    *[aux_matrix] * (x.shape[-1] // self.had_dim)
                    ).to(x.device)
                x = x @ matrix
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1]//self.had_dim)
                
            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16: #Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16: #Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


# [exp]=0 =>      0.[mantissa] * 2^Elow
# [exp]=1 =>      1.[mantissa] * 2^Elow
# [exp]=2 =>  (2~3).[mantissa] * 2^Elow = 1.[mantissa] * 2^(Elow+1)
# [exp]=3 =>  (4~7).[mantissa] * 2^Elow = 1.[mantissa] * 2^(Elow+2)
@torch.compile(dynamic=True)
def cast_to_fp_ExMy(x, Elow, Mhigh, Qmin, Qmax):
    F32_EXP_BIAS = 127
    F32_MIN_NORMAL = 2 ** (-F32_EXP_BIAS + 1)
    epsilon=F32_MIN_NORMAL
    sign, x_abs = x.sign(), x.abs()
    expo = torch.floor(torch.log2(x.abs() + epsilon))
    expo = torch.clamp(expo, min=Elow)
    mant = x_abs / (2 ** expo)
    
    mant_int = torch.floor(mant)
    mant_frac = mant - mant_int
    mant_frac = mant_frac * (Mhigh + 1)
    
    mant_frac.clamp_(0, Mhigh + 1).round_()

    mant_q = mant_int + mant_frac / (Mhigh + 1)
    y = sign * (2 ** expo) * mant_q
    
    y.clamp_(Qmin, Qmax)    # deal with overflow and underflow
                            # (caused by stochastic rounding)
    return y


def mxfp4_find_params(x):
    blocksize = 32
    F32_EXP_BIAS = 127
    F32_MIN_NORMAL = 2 ** (-F32_EXP_BIAS + 1)
    epsilon = F32_MIN_NORMAL
    original_shape = x.shape
    reshaped_x = x.reshape(-1, x.shape[-1] // blocksize, blocksize)
    absmax_per_block = torch.max(reshaped_x.abs(), dim=-1, keepdim=True)[0]
    absmax_per_block[absmax_per_block == 0] += epsilon
    
    
    
    Elow = 0        # `Elow` is FLEXIBLE, determining `BIAS`
    Ehigh = 2      # `Ehigh` depends on `Elow`
    Mhigh = 1
    Qmax = 6.0
    Qmin = -6.0
    
    scale_per_block = (2 * absmax_per_block) / (Qmax - Qmin)
    scale_per_block = scale_per_block.to(x)
    
    # Microscaling's original setting
    scale_per_block = torch.floor(torch.log2(absmax_per_block)) - Ehigh 
    scale_per_block.clamp_(-127, 127)
    scale_per_block = 2 ** scale_per_block
    
    scale_per_block = scale_per_block.repeat(1, 1, 1, blocksize).reshape(original_shape)
    return scale_per_block 


class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, format='fp', perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
    ):
        if format == 'mxfp4': 
            assert bits == 4, 'MXFP4 quantization only supports 4 bits!'
            assert sym, 'MXFP4 quantization only supports symmetric quantization!'
            if mse:
                print('Warning: MSE is not supported for MXFP4 quantization (will be ignored)!')

        self.format = format
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym and format == 'int':
            self.maxq = torch.tensor(2**(bits-1)-1)
        elif sym == False and format == 'int':
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        elif self.format == 'mxfp4':
            self.scale = mxfp4_find_params(x)
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        if self.format == 'mxfp4':
           raise NotImplementedError('MXFP4 quantization does not support quantization yet! This is because of the hack we did in GPTQ code now (deadlining...) so we remove this to make sure no one use it :D.')
        
        
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x


    def ready(self):
        if self.bits == 16:
            return True
        return torch.all(self.scale != 0)



def add_actquant(module, name='', layers=[torch.nn.Linear,
                                          ActQuantWrapper]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)

def find_qlayers(module, layers=[torch.nn.Linear,
                                ActQuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
