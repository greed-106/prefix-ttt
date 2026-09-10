"""Small correctness oracles; not production training backends.

Inputs are [B,T,H,R], [B,T,H,R], [B,T,H,D]. Eta is fixed by
the project, even when tests use smaller feature dimensions.
"""

import torch


def _prepare(q, k, v, state, valid):
    if q.shape != k.shape or q.shape[:3] != v.shape[:3] or q.ndim != 4:
        raise ValueError("expected compatible [B,T,H,D] inputs")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = (x.to(dtype) for x in (q, k, v))
    if valid is not None:
        if valid.shape != q.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [B,T]")
        mask = valid[..., None, None]
        q, k, v = (x.masked_fill(~mask, 0) for x in (q, k, v))
    shape = (q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    if state is None:
        state = q.new_zeros(shape)
    elif state.shape != shape:
        raise ValueError("initial state shape mismatch")
    return q, k, v, state.to(dtype)


def sequential_prefix(q, k, v, initial_state=None, valid=None):
    """Inclusive write-then-read oracle, preserving all state gradients."""
    output_dtype = q.dtype
    q, k, v, state = _prepare(q, k, v, initial_state, valid)
    outputs = []
    with torch.autocast(device_type=q.device.type, enabled=False):
        for t in range(q.shape[1]):
            state = state + torch.einsum("bhr,bhd->bhrd", k[:, t], v[:, t]) / 128
            outputs.append(torch.einsum("bhr,bhrd->bhd", q[:, t], state))
    out = torch.stack(outputs, 1) if outputs else v[:, :0]
    return out.to(output_dtype), state


def chunk_prefix(q, k, v, initial_state=None, valid=None, tile_size=64):
    """Exact tile expression, never materializing a state for every token."""
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    output_dtype = q.dtype
    q, k, v, state = _prepare(q, k, v, initial_state, valid)
    outputs = []
    with torch.autocast(device_type=q.device.type, enabled=False):
        for start in range(0, q.shape[1], tile_size):
            qc, kc, vc = (x[:, start:start + tile_size] for x in (q, k, v))
            scores = torch.einsum("bthr,bshr->bhts", qc, kc).tril()
            outputs.append(torch.einsum("bthr,bhrd->bthd", qc, state)
                           + torch.einsum("bhts,bshd->bthd", scores, vc) / 128)
            state = state + torch.einsum("bthr,bthd->bhrd", kc, vc) / 128
    out = torch.cat(outputs, 1) if outputs else v[:, :0]
    return out.to(output_dtype), state
