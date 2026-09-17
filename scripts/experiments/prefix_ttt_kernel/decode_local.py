"""Experimental Local decode preparation fusion; SDPA arithmetic is unchanged."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from prefix_ttt.ops.local import LocalCache, local_attention_decode as _original


@triton.jit
def _prepare(K, V, OLD_K, OLD_V, LENGTHS, SEEN, VALID,
             NEW_K, NEW_V, VISIBLE, NEXT_LENGTHS, NEXT_SEEN,
             K_B: tl.constexpr, K_T: tl.constexpr, K_H: tl.constexpr, K_D: tl.constexpr,
             V_B: tl.constexpr, V_T: tl.constexpr, V_H: tl.constexpr, V_D: tl.constexpr,
             OK_B: tl.constexpr, OK_T: tl.constexpr, OK_H: tl.constexpr, OK_D: tl.constexpr,
             OV_B: tl.constexpr, OV_T: tl.constexpr, OV_H: tl.constexpr, OV_D: tl.constexpr,
             L_S: tl.constexpr, S_S: tl.constexpr, VALID_S: tl.constexpr,
             H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    batch, tile = tl.program_id(0), tl.program_id(1)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    token, head, dim = offset // (H * D), offset // D % H, offset % D
    length = tl.load(LENGTHS + batch * L_S)
    slot = tl.minimum(length, 31)
    replace = token == slot
    old_k = tl.load(OLD_K + batch * OK_B + token * OK_T + head * OK_H + dim * OK_D,
                    (offset < 32 * H * D) & (token != slot), other=0)
    old_v = tl.load(OLD_V + batch * OV_B + token * OV_T + head * OV_H + dim * OV_D,
                    (offset < 32 * H * D) & (token != slot), other=0)
    key = tl.load(K + batch * K_B + head * K_H + dim * K_D,
                  (offset < 32 * H * D) & replace, other=0)
    value = tl.load(V + batch * V_B + head * V_H + dim * V_D,
                    (offset < 32 * H * D) & replace, other=0)
    tl.store(NEW_K + batch * 32 * H * D + offset, tl.where(replace, key, old_k), offset < 32 * H * D)
    tl.store(NEW_V + batch * 32 * H * D + offset, tl.where(replace, value, old_v), offset < 32 * H * D)
    if tile == 0:
        valid = tl.load(VALID + batch * VALID_S).to(tl.int64)
        seen = tl.load(SEEN + batch * S_S) + valid
        positions = tl.arange(0, 32)
        tl.store(VISIBLE + batch * 32 + positions, positions < length + valid)
        tl.store(NEXT_LENGTHS + batch, seen % 32)
        tl.store(NEXT_SEEN + batch, seen)


def local_attention_decode(q, k, v, valid, cache):
    """Fuse only copies/indexing; preserve fresh buffers and the existing SDPA."""
    if torch.is_grad_enabled() or not q.is_cuda or q.ndim != 4:
        return _original(q, k, v, valid, cache)
    b, t, h, d = q.shape
    if not (t == 1 and q.shape == k.shape == v.shape and q.dtype == k.dtype == v.dtype
            and q.dtype in (torch.bfloat16, torch.float32)
            and cache.key.shape == cache.value.shape == (b, 32, h, d)
            and cache.key.dtype == cache.value.dtype == q.dtype
            and cache.lengths.shape == cache.seen_tokens.shape == (b,)
            and cache.lengths.dtype == cache.seen_tokens.dtype == torch.int64
            and valid.shape == (b, 1) and valid.dtype == torch.bool
            and all(x.device == q.device for x in
                    (k, v, valid, cache.key, cache.value, cache.lengths, cache.seen_tokens))):
        return _original(q, k, v, valid, cache)
    key = torch.empty((b, 32, h, d), dtype=q.dtype, device=q.device)
    value = torch.empty_like(key)
    visible = torch.empty((b, 32), dtype=torch.bool, device=q.device)
    lengths, seen = torch.empty_like(cache.lengths), torch.empty_like(cache.seen_tokens)
    _prepare[(b, triton.cdiv(32 * h * d, 256))](
        k, v, cache.key, cache.value, cache.lengths, cache.seen_tokens, valid,
        key, value, visible, lengths, seen, *k.stride(), *v.stride(), *cache.key.stride(),
        *cache.value.stride(), cache.lengths.stride(0), cache.seen_tokens.stride(0), valid.stride(0),
        h, d, 256, num_warps=4)
    out = F.scaled_dot_product_attention(q[:, 0].unsqueeze(2), key.transpose(1, 2),
                                         value.transpose(1, 2), attn_mask=visible[:, None, None, :])
    return out.transpose(1, 2), LocalCache(key, value, lengths, seen)


@contextmanager
def patch(forward_globals=None):
    """Patch only this experiment's call sites; nesting and exceptions restore."""
    from prefix_ttt.model import hybrid

    namespaces = [hybrid.__dict__]
    if forward_globals is not None and forward_globals is not hybrid.__dict__:
        namespaces.append(forward_globals)
    saved = [namespace['local_attention_decode'] for namespace in namespaces]
    try:
        for namespace in namespaces:
            namespace['local_attention_decode'] = local_attention_decode
        yield
    finally:
        for namespace, original in zip(namespaces, saved):
            namespace['local_attention_decode'] = original
