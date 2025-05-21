from .llama import Llama, RMSNorm
from .base import GPTBase, LayerNorm
import torch

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)


def get_model(args):
    """Return the right model"""
    if args.model == "base":
        model = GPTBase(args)
        if args.use_pretrained != "none":
            model.from_pretrained(args.use_pretrained)
        return model
    elif args.model == "llama":
        model = Llama(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
