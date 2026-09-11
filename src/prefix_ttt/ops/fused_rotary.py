"""Inference RoPE fusion with the same separate multiply/add rounding as HF."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(x, partner, cosine, sine, first_half, DTYPE: tl.constexpr):
    a = (x * cosine).to(DTYPE).to(tl.float32)
    b = (tl.where(first_half, -partner, partner) * sine).to(DTYPE).to(tl.float32)
    return a + b


@triton.jit
def _rotary_kernel(Q, K, COS, SIN, QOUT, KOUT,
                    Q_B: tl.constexpr, Q_H: tl.constexpr, Q_T: tl.constexpr, Q_D: tl.constexpr,
                    K_B: tl.constexpr, K_H: tl.constexpr, K_T: tl.constexpr, K_D: tl.constexpr,
                    OQ_B: tl.constexpr, OQ_H: tl.constexpr, OQ_T: tl.constexpr, OQ_D: tl.constexpr,
                    OK_B: tl.constexpr, OK_H: tl.constexpr, OK_T: tl.constexpr, OK_D: tl.constexpr,
                    C_B: tl.constexpr, C_T: tl.constexpr, C_D: tl.constexpr,
                    S_B: tl.constexpr, S_T: tl.constexpr, S_D: tl.constexpr,
                    H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                    N: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch, head = offset // (H * T * D), offset // (T * D) % H
    token, dim = offset // D % T, offset % D
    first_half = dim < D // 2
    partner = tl.where(first_half, dim + D // 2, dim - D // 2)
    mask = offset < N
    cosine = tl.load(COS + batch * C_B + token * C_T + dim * C_D, mask, 0).to(tl.float32)
    sine = tl.load(SIN + batch * S_B + token * S_T + dim * S_D, mask, 0).to(tl.float32)
    q_base = batch * Q_B + head * Q_H + token * Q_T
    k_base = batch * K_B + head * K_H + token * K_T
    q = tl.load(Q + q_base + dim * Q_D, mask, 0).to(tl.float32)
    q_partner = tl.load(Q + q_base + partner * Q_D, mask, 0).to(tl.float32)
    k = tl.load(K + k_base + dim * K_D, mask, 0).to(tl.float32)
    k_partner = tl.load(K + k_base + partner * K_D, mask, 0).to(tl.float32)
    q_out = batch * OQ_B + head * OQ_H + token * OQ_T + dim * OQ_D
    k_out = batch * OK_B + head * OK_H + token * OK_T + dim * OK_D
    tl.store(QOUT + q_out, _rotate(q, q_partner, cosine, sine, first_half, Q.dtype.element_ty), mask)
    tl.store(KOUT + k_out, _rotate(k, k_partner, cosine, sine, first_half, K.dtype.element_ty), mask)


def rotary(q, k, cos, sin):
    """Rotate [B,H,T,D] Q/K using [B,T,D] angles, including batch-one angles."""
    _, heads, tokens, dim = q.shape
    qout, kout = torch.empty_like(q), torch.empty_like(k)
    if tokens == 1:
        # HF's elementwise outputs normalize the singleton time stride.
        qout, kout = qout.squeeze(2).unsqueeze(2), kout.squeeze(2).unsqueeze(2)
    _rotary_kernel[(triton.cdiv(q.numel(), 512),)](
        q, k, cos, sin, qout, kout, *q.stride(), *k.stride(),
        *qout.stride(), *kout.stride(),
        0 if cos.shape[0] == 1 else cos.stride(0), cos.stride(1), cos.stride(2),
        0 if sin.shape[0] == 1 else sin.stride(0), sin.stride(1), sin.stride(2),
        heads, tokens, dim, q.numel(), 512, num_warps=4, enable_fp_fusion=False)
    return qout, kout
