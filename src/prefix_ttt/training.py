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
VISUAL_QUERY_LIMIT = 64      # supervised text queries after the image, per sample
TEACHER_CHECK_TOLERANCE = 0.05   # max relative gap when re-deriving the teacher's attention
A_STAGE_PASSES = 3           # phase A: passes over the fixed 50k subset
KD_WEIGHT = 1.0              # phase B: weight of the frozen teacher's KL term
KD_TEMPERATURE = 1.0         # phase B: distillation temperature (tau), no annealing


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


def full_attention_transfer_loss(readout, target, visual, valid):
    """Per-sample A loss for ONE layer, before o_proj; caller sums layers.

P32 keeps no local branch, so the branch must reproduce the teacher's complete
attention output. Visual and text queries are scored separately, each against the
teacher's own energy on that modality, so a text majority cannot hide a visual
error. Per-sample values permit exact normalization by the actual global sample
count, including the final short accumulation group.
    """
    if visual.dtype != torch.bool or valid.dtype != torch.bool or visual.shape != valid.shape:
        raise ValueError("visual and valid must be boolean [B,T]")
    if (visual & ~valid).any() or not valid.any(dim=1).all():
        raise ValueError("invalid modality masks or empty sample")
    error = (readout.float() - target.detach().float()).square().mean(dim=(-1, -2))
    energy = target.detach().float().square().mean(dim=(-1, -2))
    result = error.new_zeros(error.shape[0])
    modalities = error.new_zeros(error.shape[0])
    for mask in (visual & valid, ~visual & valid):
        count = mask.sum(dim=1)
        active = count > 0
        denom = count.clamp_min(1)
        numerator = (error * mask).sum(dim=1) / denom
        norm = (energy * mask).sum(dim=1) / denom + ENERGY_EPS
        result = result + torch.where(active, numerator / norm, 0)
        modalities = modalities + active
    return result / modalities


def visual_transfer_loss(readout_visual, target_visual, positions, energy):
    """Per-sample A vision loss for ONE layer, before o_proj.

``positions`` selects the supervised text queries, ``energy`` is the teacher's
complete-output per-token energy averaged over the sample's valid tokens. That one
denominator is shared with the full-output term, so a small vision contribution is
never rescaled into a large relative error.
    """
    if positions.dtype != torch.bool or positions.ndim != 2:
        raise ValueError("positions must be boolean [B,T]")
    error = (readout_visual.float() - target_visual.detach().float()).square().mean(dim=(-1, -2))
    count = positions.sum(dim=1)
    active = count > 0
    denom = count.clamp_min(1)
    return torch.where(active, (error * positions).sum(dim=1) / denom / (energy + ENERGY_EPS), 0)


def visual_queries(valid, image_token_mask, limit=VISUAL_QUERY_LIMIT):
    """First ``limit`` text queries after the complete image, per sample.

Label-side only: no token is added and no question is changed. Samples without an
image are excluded because their vision contribution is identically zero; the
expansion refuses multi-image samples upstream, so "the image" is unambiguous.
    """
    if valid.dtype != torch.bool or valid.shape != image_token_mask.shape:
        raise ValueError("valid and image_token_mask must be boolean [B,T]")
    if limit <= 0:
        raise ValueError("limit must be positive")
    index = torch.arange(valid.shape[1], device=valid.device)[None].expand_as(valid)
    last_image = index.masked_fill(~image_token_mask, -1).max(dim=1).values
    selected = valid & ~image_token_mask & (index > last_image[:, None])
    order = selected.long().cumsum(dim=1)
    return selected & (order <= limit) & (last_image >= 0)[:, None]


def kd_loss(student_logits, teacher_logits, labels, global_target_count,
            temperature=KD_TEMPERATURE):
    """Teacher-to-student KL on the same shifted assistant positions as the CE.

The frozen teacher supplies a target distribution only: no hidden state crosses the
boundary and no alternative answer is generated. Each rank contributes its own token
sum divided by the GLOBAL target count, exactly like the :func:`token_normalized_ce`
call site in the manual-sum training loop; the gradient reduction is a plain sum, so
no world-size factor may be applied here.
    """
    if global_target_count <= 0 or temperature <= 0:
        raise ValueError("invalid KD normalization or temperature")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if tuple(labels.shape) != tuple(student_logits.shape[:2]):
        raise ValueError("labels must align with the logits' batch and sequence")
    student = F.log_softmax(student_logits[..., :-1, :].float() / temperature, dim=-1)
    teacher = F.log_softmax(teacher_logits[..., :-1, :].float() / temperature, dim=-1)
    per_token = F.kl_div(student, teacher, reduction="none", log_target=True).sum(-1)
    mask = labels[..., 1:].ne(IGNORE_INDEX)
    return per_token.masked_fill(~mask, 0).sum() / global_target_count * temperature ** 2


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
