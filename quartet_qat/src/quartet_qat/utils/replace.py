import torch
from torch import nn

from .config import QuartetConfig

def replace_with_quartet_linear(
    model,
    quartet_linear_config: QuartetConfig,
    current_key_name=None,
    has_been_replaced=False,
):
    from ..module import QuartetLinear
    """
    Public method that recursively replaces the Linear layers of the given model with HIGGS quantized layers.
    `accelerate` is needed to use this method. Returns the converted model and a boolean that indicates if the
    conversion has been successful or not.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`HiggsConfig`):
            The quantization config object that contains the quantization parameters.
        current_key_name (`list`, *optional*):
            A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced (`bool`, *optional*):
            A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.
    """

    from accelerate import init_empty_weights

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear):
            # Check if the current key is not in the `quantization_config.modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(current_key_name_str.endswith(key) for key in quartet_linear_config.modules_to_not_convert):
                with init_empty_weights():
                    in_features = module.in_features
                    out_features = module.out_features

                    model._modules[name] = QuartetLinear(
                        in_features,
                        out_features,
                        config=quartet_linear_config,
                        bias=module.bias is not None,
                    )
                    has_been_replaced = True

                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_quartet_linear(
                module,
                quartet_linear_config=quartet_linear_config,
                current_key_name=current_key_name,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced