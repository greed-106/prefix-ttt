"""Fused FP32 state writes with the original cuBLAS readout reduction."""

import torch
import triton
import triton.language as tl

from prefix_ttt.ops import ETA


@triton.jit
def _recurrent_write_kernel(
    Q, K, V, State, Valid, MaskedQ, NewState,
    H: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
    Q_B: tl.constexpr, Q_H: tl.constexpr, Q_R: tl.constexpr,
    K_B: tl.constexpr, K_H: tl.constexpr, K_R: tl.constexpr,
    V_B: tl.constexpr, V_H: tl.constexpr, V_D: tl.constexpr,
    S_B: tl.constexpr, S_H: tl.constexpr, S_R: tl.constexpr, S_D: tl.constexpr,
    N_B: tl.constexpr, N_H: tl.constexpr, N_R: tl.constexpr, N_D: tl.constexpr,
    VALID_B: tl.constexpr, WRITE_SCALE: tl.constexpr, BLOCK_R: tl.constexpr,
):
    row_head = tl.program_id(1)
    row, head = row_head // H, row_head % H
    r = tl.arange(0, BLOCK_R)
    d = tl.program_id(0) * 32 + tl.arange(0, 32)
    active = tl.load(Valid + row * VALID_B)
    if tl.program_id(0) == 0:
        q = tl.load(Q + row * Q_B + head * Q_H + r * Q_R, r < R, 0).to(tl.float32)
        tl.store(MaskedQ + row_head * R + r, tl.where(active, q, 0), r < R)
    k = tl.load(K + row * K_B + head * K_H + r * K_R, r < R, 0).to(tl.float32)
    v = tl.load(V + row * V_B + head * V_H + d * V_D, d < D, 0).to(tl.float32)
    mask = (r[:, None] < R) & (d[None, :] < D)
    previous = tl.load(State + row * S_B + head * S_H
                       + r[:, None] * S_R + d[None, :] * S_D, mask, 0)
    update = (k[:, None] * v[None, :]) * WRITE_SCALE
    state = tl.where(active, previous + update, previous)
    tl.store(NewState + row * N_B + head * N_H
             + r[:, None] * N_R + d[None, :] * N_D, state, mask)


def recurrent_decode(q, k, v, state=None, valid=None):
    """Inclusive [B,1,H,R/D] update with unchanged FP32 einsum readout order."""
    batch, _, heads, features = q.shape
    values = v.shape[-1]
    if state is None:
        state = torch.zeros(batch, heads, features, values, dtype=torch.float32, device=q.device)
    if valid is None:
        valid = torch.ones(batch, 1, dtype=torch.bool, device=q.device)
    qf = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    new_state = torch.empty_like(state)
    _recurrent_write_kernel[(triton.cdiv(values, 32), batch * heads)](
        q, k, v, state, valid, qf, new_state, heads, features, values,
        q.stride(0), q.stride(2), q.stride(3),
        k.stride(0), k.stride(2), k.stride(3),
        v.stride(0), v.stride(2), v.stride(3),
        *state.stride(), *new_state.stride(), valid.stride(0), ETA,
        triton.next_power_of_2(features), num_warps=4, enable_fp_fusion=False,
    )
    with torch.autocast(device_type=q.device.type, enabled=False):
        memory = torch.einsum("bhr,bhrd->bhd", qf[:, 0], new_state).unsqueeze(1)
    return memory.to(q.dtype), new_state
