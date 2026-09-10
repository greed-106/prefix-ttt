"""Explicit, lazy production-backend boundary. No reference fallback."""

import torch


FLA_COMMIT = "c51953382397da5c3b7b8a41e568915b703e2934"


def _pack(q, k, v, valid):
    """Keep sequence boundaries, dropping zero-length rows only for the kernel."""
    if valid.shape != q.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [B,T]")
    lengths = valid.sum(1)
    active_rows = (lengths > 0).nonzero(as_tuple=True)[0]
    cu_seqlens = torch.cat((lengths.new_zeros(1), lengths[active_rows].cumsum(0)))
    return tuple(x[valid].unsqueeze(0) for x in (q, k, v)), active_rows, cu_seqlens


def _run_kernel(q, k, v, state, need_final_state, tile_size, cu_seqlens):
    try:
        if tile_size == 64:
            from fla.ops.linear_attn import chunk_linear_attn
            return chunk_linear_attn(q=q, k=k, v=v / 128.0, scale=1.0,
                                     initial_state=state, output_final_state=need_final_state,
                                     normalize=False, cu_seqlens=cu_seqlens)
        from fla.ops.simple_gla import chunk_simple_gla
        return chunk_simple_gla(q=q, k=k, v=v / 128.0, g=None, g_gamma=None,
                                scale=1.0, initial_state=state, output_final_state=need_final_state,
                                cu_seqlens=cu_seqlens, chunk_size=tile_size)
    except ImportError as exc:
        raise RuntimeError("locked FLA backend is unavailable; no fallback is permitted") from exc


def fla_prefix(q, k, v, initial_state=None, need_final_state=True, valid=None, tile_size=64):
    """FLA prefill, padded independent rows unpacked through cu_seqlens."""
    if not q.is_cuda:
        raise RuntimeError("FLA Prefix-TTT requires a CUDA GPU; CPU/reference only is not GPU validation")
    if q.ndim != 4 or q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("expected compatible [B,T,H,D] inputs")
    if tile_size not in (16, 32, 64, 128):
        raise ValueError("supported tile sizes are 16,32,64,128")
    state_shape = (q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    if initial_state is not None and (initial_state.dtype != torch.float32 or initial_state.shape != state_shape):
        raise ValueError("FLA initial state must use FP32 storage")
    if valid is None:
        valid = torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
    (pq, pk, pv), active, cu_seqlens = _pack(q, k, v, valid)
    base_state = (q.new_zeros(state_shape, dtype=torch.float32)
                  if initial_state is None else initial_state)
    if pq.shape[1] == 0:
        return v * 0, base_state if need_final_state else None
    state = None if initial_state is None else initial_state.index_select(0, active)
    packed_out, final = _run_kernel(pq, pk, pv, state, need_final_state, tile_size, cu_seqlens)
    out = v.new_zeros(v.shape)
    out[valid] = packed_out[0]
    if need_final_state:
        if final.dtype != torch.float32:
            raise RuntimeError("FLA returned non-FP32 final state; backend contract violated")
        final = base_state.index_copy(0, active, final)
    return out, final


def recurrent_step(q, k, v, initial_state=None, valid=None):
    """One token per row, vectorized FP32 write/read, usable on CPU and GPU.

    Incoming features/values are unscaled. This is the production single-step
    recurrence, not a loop or a fallback for full-sequence FLA execution.
    """
    if q.ndim != 4 or q.shape[1] != 1 or q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("recurrent_step expects compatible [B,1,H,D] inputs")
    shape = (q.shape[0], q.shape[2], q.shape[3], v.shape[3])
    if initial_state is None:
        initial_state = q.new_zeros(shape, dtype=torch.float32)
    if initial_state.shape != shape or initial_state.dtype != torch.float32:
        raise ValueError("recurrent initial state must be [B,H,R,D] FP32")
    if valid is None:
        valid = torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
    if valid.shape != q.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be boolean [B,1]")
    # Explicitly disable autocast so both contractions really accumulate FP32.
    with torch.autocast(device_type=q.device.type, enabled=False):
        mask = valid[..., None, None]
        qf, kf, vf = (x.float().masked_fill(~mask, 0) for x in (q, k, v))
        state = initial_state + torch.einsum("bhr,bhd->bhrd", kf[:, 0], vf[:, 0]) / 128.0
        out = torch.einsum("bhr,bhrd->bhd", qf[:, 0], state).unsqueeze(1)
    return out.to(q.dtype), state
