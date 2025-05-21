import torch
import logging
from tqdm import tqdm


@torch.no_grad()
def evaluator(model, testenc, dev, args):

    model.eval()

    if 'meta' in args.model_name:
        llama_type = True
        opt_type = False
        
        use_cache = model.config.use_cache
        model.config.use_cache = False
    elif 'QuEST' in args.model_name:
        llama_type = False
        quest_type = True
    else:
        raise ValueError(f'Unknown model {args.model_name}')


    

    if llama_type:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
    elif quest_type:
        layers = model.transformer.h
        model.transformer.wte = model.transformer.wte.to(dev)

    layers[0] = layers[0].to(dev)

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)  # (nsamples, seqlen)

    batch_size = args.bsz
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    inps = [0] * nbatches
    cache = {'i': 0, 'attention_mask': None}
    class Catcher(torch.nn.Module):
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
   
    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    if llama_type:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        position_embeddings = cache['position_embeddings']
        
    elif quest_type:
        model.transformer.wte = model.transformer.wte.cpu()
        freqs_cis = cache['freqs_cis']

    torch.cuda.empty_cache()
    outs = [0] * nbatches
    attention_mask = cache['attention_mask']

    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)

        for j in range(nbatches):
            if llama_type:
                outs[j] = layer(inps[j], attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
            elif quest_type:
                 outs[j] = layer(inps[j], freqs_cis=freqs_cis)   
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        
    model = model.cpu()
    del outs

    if llama_type:
        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)
    elif quest_type:
        if model.transformer.ln_f is not None:
            model.transformer.ln_f = model.transformer.ln_f.to(dev)

    total_memory = torch.cuda.get_device_properties(torch.device("cuda:0")).total_memory/ (1024 ** 3)
    if total_memory < 38 and args.bsz > 16:
        logging.info(f"GPU memory is less than 38GB, moving model to CPU")
        model = model.cpu()
    else:
        logging.info(f"GPU memory is more than 38GB, using GPU for final layer")
        model.lm_head = model.lm_head.to(dev) 
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
    # use tqdm
    
    for i in tqdm(range(nbatches), desc="(Loss Calc.) Batches"):
        hidden_states = inps[i].to(model.lm_head.weight.device) #move to the same device as lm_head
        if llama_type:
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
        elif quest_type:
            if model.transformer.ln_f is not None:
                hidden_states = model.transformer.ln_f(hidden_states)
        
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels.to(shift_logits.device))
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    if llama_type:
        model.config.use_cache = use_cache
    logging.info(f'\n{args.eval_dataset.upper()} PPL: {ppl.item():.3f}')
    return ppl.item()


@torch.no_grad()
def quest_evaluate_ppl(
    model, 
    testloader
):
    '''
        Evaluate the perplexity of the model on the test set (Faster than layer-by-layer evaluation)
        Args:
            model: The model to evaluate.
            testloader: The test dataloader.
    '''
    
    import time

    start_time = time.time()

    model.eval()
            

    loss_list_val = []
    import torch.nn.functional as F
    logging.info("Evaluating perplexity...")
    for batch in tqdm(testloader, desc="Evaluating perplexity", unit="batch"):
        
        inputs = batch['input_ids'].cuda()[:, :-1].contiguous()
        targets = batch['input_ids'].cuda()[:, 1:].contiguous()
        
        outputs = model(inputs, targets=targets, get_logits=True)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
      
    val_loss = torch.stack(loss_list_val).mean().item()
    ppl = 2.71828**val_loss
    
    print(f"Validation perplexity: {ppl}")
    

    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )
    return ppl
