"""Exact LLM LoRA selection and auditable freezing."""
import re

import torch


LLM_PROJECTION = re.compile(r'model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))')
LLM_PROJECTIONS_PER_LAYER = 7   # number of names the pattern above matches per layer

LORA_RANK, LORA_ALPHA = 32, 64
LORA_DROPOUT, LORA_BIAS, LORA_SEED = 0.0, 'none', 43


def install_lora(model, *, rank=LORA_RANK, alpha=LORA_ALPHA, seed=LORA_SEED,
                 new_parameters=()):
    from peft import LoraConfig, get_peft_model
    names = [name for name, _ in model.named_modules() if LLM_PROJECTION.fullmatch(name)]
    if len(names) != model.config.num_hidden_layers * LLM_PROJECTIONS_PER_LAYER:
        raise ValueError('Incomplete LLM projection whitelist')
    # Objects survive PEFT wrapping; suffix matching never selects ViT/projector.
    new_parameters = list(new_parameters)
    model.requires_grad_(False)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha,
            lora_dropout=LORA_DROPOUT, bias=LORA_BIAS, target_modules=names,
            task_type='CAUSAL_LM'))
    for parameter in new_parameters:
        parameter.requires_grad_(True)
        parameter.data = parameter.data.float()
    for name, parameter in model.named_parameters():
        if 'lora_' in name:
            parameter.data = parameter.data.float()
        if 'vision_tower' in name or 'mm_projector' in name:
            parameter.requires_grad_(False)
    return model


def merge_lora_weights(model):
    """Fold every LoRA branch into its frozen base weight for inference.

    Exact in real arithmetic, ``Wx + BAx * scaling == (W + BA * scaling)x``; the only
    loss is rounding the FP32 delta into the base weight dtype. Each wrapped Linear is
    replaced by its base layer, which also removes the two extra matmuls per
    projection from the forward path. Returns the number of merged projections.
    """
    names = [name for name, module in model.named_modules()
             if isinstance(getattr(module, 'lora_A', None), torch.nn.ModuleDict)
             and 'default' in module.lora_A]
    if len(names) != model.config.num_hidden_layers * LLM_PROJECTIONS_PER_LAYER:
        raise ValueError('Incomplete LoRA whitelist; refusing a partial merge')
    for name in names:
        module = model.get_submodule(name)
        base = module.base_layer
        delta = (module.lora_B['default'].weight.detach().float()
                 @ module.lora_A['default'].weight.detach().float())
        base.weight.data += (delta * module.scaling['default']).to(base.weight.dtype)
        parent, attribute = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), attribute, base)
    return len(names)


def audit_parameters(model, *, new_parameters=()):
    new_ids = {id(p) for p in new_parameters}
    records, seen = [], set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        kind = 'new_module' if id(parameter) in new_ids else ('lora' if 'lora_' in name else 'frozen')
        if parameter.requires_grad and kind == 'frozen':
            raise ValueError(f'Unexpected trainable base parameter: {name}')
        records.append(dict(name=name, shape=list(parameter.shape), dtype=str(parameter.dtype),
                            requires_grad=parameter.requires_grad,
                            optimizer_group=kind if parameter.requires_grad else None,
                            numel=parameter.numel()))
    return records
