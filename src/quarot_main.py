from quarot_utils import utils, model_utils, data_utils, quant_utils, rotation_utils, gptq_utils, eval_utils, hadamard_utils
import torch
import transformers


# QuEST imports
from models.utils import get_model as quest_get_model
import distributed
import config
from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from quarot_utils.utils import load_quest_checkpoints
from quarot_utils.hadamard_utils import is_pow2
from models.quantization import QuantizedLinear
from quarot_utils.quant_utils import ActQuantWrapper
from huggingface_hub import snapshot_download
import json

def update_quest_args(args, local_dir):
    json_path = f"{local_dir}/summary.json"
    with open(json_path, 'r') as f:
        json_args = json.load(f)['args']
    
    # Step 4: Update args from JSON where keys match
    for key, value in json_args.items():
        if hasattr(args, key) and 'wandb' not in key:
            setattr(args, key, value)
    return args


def main():
    args, rem_args, parser = utils.parser_gen()
    args = config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )
    distributed_backend = distributed.single.SinlgeNodeBackend(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    transformers.set_seed(args.seed)
    
    
    if 'QuEST' in args.model_name:
        
        class PseudoDdp(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self._orig_mod = torch.nn.ModuleDict({
                    "module": model,
                })
                
    
        # This will download the entire repository (including .bin, config, tokenizer etc.)
        local_dir = snapshot_download(repo_id=f"daslab-testing/{args.model_name}")
        print(f"Model downloaded to: {local_dir}")
        
        args = update_quest_args(args, local_dir)
        
        model =  PseudoDdp(quest_get_model(args))
        
        model = load_quest_checkpoints(args, model, local_dir)
        
        model.eval()
        model.seqlen = args.sequence_length
        model.config.hidden_size = model.config.n_embd
        model.config.num_attention_heads = model.config.n_head
        model.config.intermediate_size = model.transformer.h[0].mlp.c_proj.in_features
        model.tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
    else:
        model = model_utils.get_model(args.model_name, args.hf_token)
    model.eval()
    
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project)
        wandb.config.update(args)
    
    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(model, args)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)
            
        quant_utils.add_actquant(model,
                                 layers=[
                                        torch.nn.Linear, 
                                        ActQuantWrapper, 
                                        QuantizedLinear]
                                 ) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model,
                                           layers=[
                                               torch.nn.Linear, 
                                               ActQuantWrapper, 
                                               QuantizedLinear]
                                           )
        for name in qlayers:
            if 'down_proj' in name or 'mlp.c_proj' in name:
                if is_pow2(model.config.intermediate_size):
                    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                else:
                    had_dim = 1
                    while had_dim < model.config.intermediate_size:
                        had_dim *= 2
                        if model.config.intermediate_size % had_dim != 0:
                            had_dim //= 2
                            break    
                    had_K, K = hadamard_utils.get_hadK(had_dim)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_dim = had_dim
                    
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model,
                                 layers=[
                                        torch.nn.Linear, 
                                        ActQuantWrapper, 
                                        QuantizedLinear]) #Add Activation Wrapper to the model as the rest of the code assumes it is present
        
                
    if args.w_bits < 16:
        save_dict = {}
            
        if not args.w_rtn: # GPTQ Weight Quantization
            assert "llama" in args.model_name or 'QuEST' in args.model_name, "Only llama and QuEST models are supported for GPTQ!"
            
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model_name,
                seqlen=model.seqlen, eval_mode=False,
                tokenizer=model.tokenizer if 'QuEST' in args.model_name else None,
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            
    

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0 and "llama" in args.model_name:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
        
        for name in qlayers:            
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = 1.0
            layer_act_format = args.a_format
            
            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                              groupsize=args.v_groupsize,
                                              sym=not(args.v_asym),
                                              clip_ratio=args.v_clip_ratio)
            
            if 'lm_head' in name: #Skip lm_head quantization   
                layer_input_bits = 16
                layer_act_format = 'fp'
            
            if 'down_proj' in name: #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

                
            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip,
                                              format=layer_act_format)

    if args.k_bits < 16:
        assert 'QuEST' not in args.model_name, "QuEST does not support k quantization yet!"
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                          "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            config=model.config,
                            **k_quant_config)
        
        
    if 'QuEST' in args.model_name:
        dataset = data_utils.get_quest_dataset('wikitext2')
        test_dataset = dataset["test"]
        
        test_loader = data_utils.prepare_test_dataloader(
            dataset=test_dataset, 
            tokenizer=model.tokenizer, 
            batch_size=4,
            seqlen=model.seqlen
        )
        dataset_ppl = eval_utils.quest_evaluate_ppl(model.to(utils.DEV), 
                                                     testloader=test_loader)
        print(f'Loaded model perplexity: {dataset_ppl}')
        if args.wandb:
            wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})
        return
        
    # Evaluating on dataset
    testloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model_name,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True,
            tokenizer=model.tokenizer if 'QuEST' in args.model_name else None,
        )

    
    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    if args.wandb:
        wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})



if __name__ == '__main__':
    main()
