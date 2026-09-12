"""CPU-testable loss and trajectory primitives, shared by future A/B runners.

These functions do not constitute a full training entrypoint.
"""
import math

import torch
from torch.nn import functional as F

from prefix_ttt.model.labels import IGNORE_INDEX


EFFECTIVE_BATCH_SIZE = 128   # every optimizer step spans this many samples

NEW_MODULE_LR = 1e-4         # phase B: the Prefix-TTT parameters
LORA_LR = 2e-5               # phase B: the LoRA adapters
BASE_LR = 2e-5               # phase B full fine-tuning: the pretrained base weights
MATRIX_WEIGHT_DECAY = 0.01   # only on matrices, never on norms or gates
WARMUP_FRACTION = 0.03
GRAD_CLIP = 1.0
ADAM_BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
ENERGY_EPS = 1e-6            # guards the normalised transfer diagnostic
PILOT_MIN_SAMPLES = 50_000   # the pilot split of the fixed trajectory


def accumulation_steps(world_size, micro_batch_size, effective_batch_size=EFFECTIVE_BATCH_SIZE):
    divisor = world_size * micro_batch_size
    if world_size <= 0 or micro_batch_size <= 0 or effective_batch_size % divisor:
        raise ValueError("effective batch must be divisible by positive world_size * micro_batch")
    return effective_batch_size // divisor


def shifted_target_count(labels):
    return int(labels[..., 1:].ne(IGNORE_INDEX).sum())


def token_normalized_ce(logits, labels, global_target_count, world_size=1):
    """Microbatch contribution to one DDP-averaged accumulation step.

Pass the total target count across ALL ranks and microbatches of the group.
Do not divide this loss by gradient_accumulation_steps again. A microbatch
with zero targets contributes differentiable zero; an empty group is an error.
    """
    if global_target_count <= 0 or world_size <= 0:
        raise ValueError("empty accumulation group or invalid world size")
    summed = F.cross_entropy(logits[..., :-1, :].float().reshape(-1, logits.shape[-1]),
                             labels[..., 1:].reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum")
    return summed * world_size / global_target_count


def residual_transfer_loss(readout, full, local, visual, valid):
    """Per-sample A loss for ONE layer, before o_proj; caller sums layers.

Returning per-sample values permits exact normalization by actual global
sample count, including the final short accumulation group.
    """
    if visual.dtype != torch.bool or valid.dtype != torch.bool or visual.shape != valid.shape:
        raise ValueError("visual and valid must be boolean [B,T]")
    if (visual & ~valid).any() or not valid.any(dim=1).all():
        raise ValueError("invalid modality masks or empty sample")
    target = (full - local).detach().float()
    error = (readout.float() - target).square().mean(dim=(-1, -2))
    energy = full.detach().float().square().mean(dim=(-1, -2))
    result = error.new_zeros(error.shape[0])
    modalities = error.new_zeros(error.shape[0])
    for mask in (visual & valid, ~visual & valid):
        count = mask.sum(dim=1)
        active = count > 0
        denom = count.clamp_min(1)
        numerator = (error * mask).sum(dim=1) / denom
        norm = (energy * mask).sum(dim=1) / denom + 1e-6
        result = result + torch.where(active, numerator / norm, 0)
        modalities = modalities + active
    return result / modalities


def trajectory(train_samples, effective_batch_size=EFFECTIVE_BATCH_SIZE,
               pilot_min_samples=PILOT_MIN_SAMPLES):
    if train_samples <= 0 or effective_batch_size <= 0:
        raise ValueError("positive training set and batch required")
    total = math.ceil(train_samples / effective_batch_size)
    pilot = math.ceil(pilot_min_samples / effective_batch_size)
    if pilot > total or train_samples < pilot_min_samples:
        raise ValueError("train set is too small for the specified Pilot")
    return {"total_steps": total, "pilot_step": pilot,
            "pilot_samples": min(pilot * effective_batch_size, train_samples)}


def cosine_factor(step, total_steps, warmup_fraction=WARMUP_FRACTION):
    """HF-style integer-ceil warmup, parameterized by FULL stage trajectory."""
    if total_steps <= 0 or not 0 <= warmup_fraction < 1:
        raise ValueError("invalid schedule")
    warmup = math.ceil(total_steps * warmup_fraction)
    if step < warmup:
        return step / max(1, warmup)
    progress = min(max((step - warmup) / max(1, total_steps - warmup), 0), 1)
    return 0.5 * (1 + math.cos(math.pi * progress))


def optimizer_groups(model, new_parameters=(), allow_base=False):
    """Exact B learning rates/decay, using the final parameter audit whitelist."""
    from prefix_ttt.model.trainability import audit_parameters
    records = audit_parameters(model, new_parameters=new_parameters, allow_base=allow_base)
    by_name = dict(model.named_parameters())
    learning_rates = {'new_module': NEW_MODULE_LR, 'base': BASE_LR, 'lora': LORA_LR}
    groups = {}
    seen = set()
    for record in records:
        if not record['requires_grad']:
            continue
        parameter = by_name[record['name']]
        if id(parameter) in seen:
            raise ValueError('Duplicate optimizer parameter')
        seen.add(id(parameter))
        kind = record['optimizer_group']
        learning_rate = learning_rates[kind]
        decay = MATRIX_WEIGHT_DECAY if parameter.ndim >= 2 else 0.0
        key = (kind, decay)
        group = groups.setdefault(key, {'params': [], 'lr': learning_rate,
                                        'weight_decay': decay, 'name': kind})
        group['params'].append(parameter)
    if not groups:
        raise ValueError('No trainable parameters')
    return list(groups.values())
