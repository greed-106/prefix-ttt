"""Exact LLM LoRA selection and auditable freezing."""
import re

import torch


LLM_PROJECTION = re.compile(r'model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))')


def install_lora(model, *, rank=32, alpha=64, seed=43, new_parameters=()):
    from peft import LoraConfig, get_peft_model
    names = [name for name, _ in model.named_modules() if LLM_PROJECTION.fullmatch(name)]
    if len(names) != model.config.num_hidden_layers * 7:
        raise ValueError('Incomplete LLM projection whitelist')
    # Objects survive PEFT wrapping; suffix matching never selects ViT/projector.
    new_parameters = list(new_parameters)
    model.requires_grad_(False)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha,
            lora_dropout=0.0, bias='none', target_modules=names, task_type='CAUSAL_LM'))
    for parameter in new_parameters:
        parameter.requires_grad_(True)
        parameter.data = parameter.data.float()
    for name, parameter in model.named_parameters():
        if 'lora_' in name:
            parameter.data = parameter.data.float()
        if 'vision_tower' in name or 'mm_projector' in name:
            parameter.requires_grad_(False)
    return model


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
