import math
import time
import tqdm
import torch
import torch.nn as nn
from .utils import *
from .quant_utils import *
import logging

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from models.quantization import QuantizedLinear

class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]
                if self.quantizer.format == 'mxfp4':
                    # let's have the most inefficient code possible for this (deadlining...)
                    w_dtype = w.dtype
                    current_column_id = i1 + i
                    Elow = 0        
                    Ehigh = 2      
                    Mhigh = 1
                    Qmax = 6.0
                    Qmin = -6.0
                    current_scale = self.quantizer.scale[:, current_column_id].to(w_dtype).to(w.device)
                    q = (cast_to_fp_ExMy(w  * (3/4) / current_scale, Elow, Mhigh, Qmin, Qmax).to(w_dtype) * current_scale * (4/3)).to(w_dtype)
                else:
                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        cleanup_memory(verbos=False)
        
        
@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    logging.info('-----GPTQ Quantization-----')
    
    llama_type = quest_type = False
    if 'meta' in args.model_name:
        llama_type = True
        use_cache = model.config.use_cache
        model.config.use_cache = False
    elif 'QuEST' in args.model_name:
        quest_type = True
    else:
        raise ValueError(f'Unknown model {args.model_name}')

    
    if llama_type:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    elif quest_type:
        layers = model.transformer.h
        model.transformer.wte = model.transformer.wte.to(dev)
        
    layers[0] = layers[0].to(dev)
    

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'freqs_cis': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if llama_type:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_embeddings'] = kwargs['position_embeddings']
            elif quest_type:
                cache['freqs_cis'] = kwargs['freqs_cis']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    
    if llama_type:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        attention_mask = cache['attention_mask']
        position_embeddings = cache['position_embeddings']
    else:
        model.transformer.wte = model.transformer.wte.cpu()
        freqs_cis = cache['freqs_cis']
        
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    

    quantizers = {}
    if llama_type:
        sequential = [
                ['self_attn.k_proj.module', 'self_attn.v_proj.module', 'self_attn.q_proj.module'],
                ['self_attn.o_proj.module'],
                ['mlp.up_proj.module', 'mlp.gate_proj.module'],
                ['mlp.down_proj.module']
            ]
    elif quest_type:
        sequential = [
                ['attn.c_attn.module'],
                ['attn.c_proj.module'],
                ['mlp.w12.module'],
                ['mlp.c_proj.module']
            ]
    for i in range(len(layers)):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = find_qlayers(layer, layers=[torch.nn.Linear,
                                           QuantizedLinear])
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.w_bits
                layer_weight_sym = not(args.w_asym)
                layer_format = args.w_format
                if 'lm_head' in name:
                    layer_weight_bits = 16
                    layer_format = 'fp'
                    continue
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.w_clip, format=layer_format
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                if llama_type:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                elif quest_type:
                    outs[j] = layer(inps[j].unsqueeze(0), freqs_cis=freqs_cis) 
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.w_groupsize
                # if gptq[name].quantizer.format == 'mxfp4':
                    # layer_w_groupsize = 32
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            if llama_type:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
            elif quest_type:
                outs[j] = layer(inps[j].unsqueeze(0), freqs_cis=freqs_cis)

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if llama_type:
        model.config.use_cache = use_cache
    cleanup_memory(verbos=True)
    logging.info('-----GPTQ Quantization Done-----\n')
    return quantizers



       
@torch.no_grad()
def rtn_fwrd(model, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    
    
    llama_type = quest_type = False
    if 'meta' in args.model_name:
        llama_type = True
    elif 'QuEST' in args.model_name:
        quest_type = True
    else:
        raise ValueError(f'Unknown model {args.model_name}')


    if llama_type:
        layers = model.model.layers
    elif quest_type:
        layers = model.transformer.h
        
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = find_qlayers(layer,
                              layers=[torch.nn.Linear,
                                      QuantizedLinear])

        for name in subset:
            layer_weight_bits = args.w_bits
            layer_weight_format = args.w_format
            if 'lm_head' in name:
                layer_weight_bits = 16
                layer_weight_format = 'fp'
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip, format=layer_weight_format
            )
            W = subset[name].weight.data
            if quantizer.format == 'mxfp4':
                subset[name].weight.data = mxfp4_sym_quant_dequant(W).to(
                    next(iter(layer.parameters())).dtype)
            else:
                quantizer.find_params(W)
                subset[name].weight.data = quantizer.quantize(W).to(
                    next(iter(layer.parameters())).dtype)
                quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    cleanup_memory(verbos=True)
    return quantizers
