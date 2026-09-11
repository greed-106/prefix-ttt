"""True batched block-local attention; block indices count valid tokens."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from prefix_ttt.ops import LOCAL_BLOCK_SIZE


def local_attention(q, k, v, valid=None, block_size=LOCAL_BLOCK_SIZE):
    """Training/prefill [B,T,H,D] path with only block_size² score tiles.

    This standalone path starts at logical position zero. Cached segmented
    prefill is deliberately not implemented here; callers must not treat it
    as a streaming API.
    """
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("expected matching [B,T,H,D] inputs")
    if block_size != LOCAL_BLOCK_SIZE:
        raise ValueError(f"Prefix-TTT Local block size is fixed at {LOCAL_BLOCK_SIZE}")
    b, t, h, d = q.shape
    if valid is None:
        valid = torch.ones((b, t), dtype=torch.bool, device=q.device)
    if valid.shape != (b, t) or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [B,T]")
    if t == 0:
        return v.clone()
    # Overallocate by physical T to avoid device synchronization for max length.
    blocks = (t + block_size - 1) // block_size
    rows, physical = valid.nonzero(as_tuple=True)
    logical = valid.long().cumsum(1)[rows, physical] - 1
    flat = rows * blocks * block_size + logical
    packed = []
    for x in (q, k, v):  # Three projections, never a loop over tokens/blocks.
        p = x.new_zeros((b * blocks * block_size, h, d))
        p = p.index_copy(0, flat, x[rows, physical])
        packed.append(p.reshape(b * blocks, block_size, h, d).transpose(1, 2))
    # Valid tokens are contiguous in each packed block; trailing slots cannot
    # affect valid queries under the causal mask and are discarded below.
    out = F.scaled_dot_product_attention(*packed, is_causal=True, dropout_p=0.0)
    out = out.transpose(1, 2).reshape(b * blocks * block_size, h, d)
    result = v.new_zeros((b * t, h, d))
    result = result.index_copy(0, rows * t + physical, out[flat])
    return result.reshape(b, t, h, d)


@dataclass
class LocalCache:
    """Current incomplete logical block, stored in fixed 32-slot buffers."""

    key: torch.Tensor
    value: torch.Tensor
    lengths: torch.Tensor
    seen_tokens: torch.Tensor


def local_attention_decode(q, k, v, valid, cache):
    """Decode fast path: one token per row attending to its block plus itself.

    The visible key set is the same as the packed path produces - the valid tail of
    the current 32-token block plus the incoming token - but slots past that tail are
    excluded by a boolean mask instead of being packed away, which removes the
    nonzero/cumsum/index_copy machinery from every decoded token.
    """
    if q.shape[1] != 1 or q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("local_attention_decode expects [B,1,H,D] inputs")
    b, _, h, d = q.shape
    if cache.key.shape != (b, LOCAL_BLOCK_SIZE, h, d) or cache.value.shape != cache.key.shape:
        raise ValueError(f"cached KV must have fixed [B,{LOCAL_BLOCK_SIZE},H,D] shape")
    if valid.shape != (b, 1) or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [B,1]")
    lengths = cache.lengths
    slot = lengths.clamp_max(31).view(b, 1, 1, 1).expand(b, 1, h, d)
    key = cache.key.scatter(1, slot, k)
    value = cache.value.scatter(1, slot, v)
    visible = (torch.arange(LOCAL_BLOCK_SIZE, device=q.device)[None]
               < (lengths + valid[:, 0])[:, None])
    out = F.scaled_dot_product_attention(q[:, 0].unsqueeze(2), key.transpose(1, 2),
                                         value.transpose(1, 2),
                                         attn_mask=visible[:, None, None, :])
    seen = cache.seen_tokens + valid[:, 0]
    return out.transpose(1, 2), LocalCache(key, value, seen.remainder(LOCAL_BLOCK_SIZE), seen)


def local_attention_cached(q, k, v, valid=None, cache=None, finished=None):
    """Segmented prefill/decode with gradients through incoming Local KV.

    The cached block starts on an absolute 32-token boundary. Combining it
    with current tokens therefore preserves block visibility in one batched
    SDPA call. Completed blocks are discarded; no prompt recomputation.
    """
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("expected matching [B,T,H,D] inputs")
    b, t, h, d = q.shape
    if valid is None:
        valid = torch.ones((b, t), dtype=torch.bool, device=q.device)
    if valid.shape != (b, t) or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [B,T]")
    if finished is not None:
        if finished.shape != (b,) or finished.dtype != torch.bool:
            raise ValueError("finished must be boolean [B]")
        valid = valid & ~finished[:, None]
    if cache is None:
        cache = LocalCache(k.new_zeros(b, LOCAL_BLOCK_SIZE, h, d),
                           v.new_zeros(b, LOCAL_BLOCK_SIZE, h, d),
                           torch.zeros(b, device=q.device, dtype=torch.long),
                           torch.zeros(b, device=q.device, dtype=torch.long))
    if cache.key.shape != (b, LOCAL_BLOCK_SIZE, h, d) or cache.value.shape != cache.key.shape:
        raise ValueError(f"cached KV must have fixed [B,{LOCAL_BLOCK_SIZE},H,D] shape")
    if cache.lengths.shape != (b,) or cache.seen_tokens.shape != (b,):
        raise ValueError("cached counters must have [B] shape")
    previous_valid = (torch.arange(LOCAL_BLOCK_SIZE, device=q.device)[None]
                      < cache.lengths[:, None])
    combined_valid = torch.cat((previous_valid, valid), dim=1)
    combined_k = torch.cat((cache.key, k), dim=1)
    combined_v = torch.cat((cache.value, v), dim=1)
    combined_q = torch.cat((torch.zeros_like(cache.key), q), dim=1)
    out = local_attention(combined_q, combined_k, combined_v,
                          combined_valid)[:, LOCAL_BLOCK_SIZE:]
    seen = cache.seen_tokens + valid.sum(1)
    lengths = seen.remainder(LOCAL_BLOCK_SIZE)
    counts = combined_valid.sum(1)
    logical = combined_valid.long().cumsum(1) - 1
    tail_start = counts - lengths
    retain = combined_valid & (logical >= tail_start[:, None])
    rows, physical = retain.nonzero(as_tuple=True)
    slot = logical[rows, physical] - tail_start[rows]
    new_buffers = []
    for values in (combined_k, combined_v):
        buffer = values.new_zeros(b * LOCAL_BLOCK_SIZE, h, d)
        buffer = buffer.index_copy(0, rows * LOCAL_BLOCK_SIZE + slot, values[rows, physical])
        new_buffers.append(buffer.reshape(b, LOCAL_BLOCK_SIZE, h, d))
    return out, LocalCache(*new_buffers, lengths, seen)
