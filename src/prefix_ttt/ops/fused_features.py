"""Inference-only feature and readout fusion, preserving dtype rounding points."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from prefix_ttt.ops.features import RMS_EPS, rms_no_affine


EPS = tl.constexpr(RMS_EPS)


@triton.jit
def _mean_square_128(x):
    # ATen Reduce.cuh: four strided values per lane, then ascending shuffles.
    square = x * x
    lane = tl.arange(0, 32)
    total = tl.gather(square, lane, 0) + tl.gather(square, lane + 32, 0)
    total = total + tl.gather(square, lane + 64, 0)
    total = total + tl.gather(square, lane + 96, 0)
    for step in tl.static_range(5):
        offset = 1 << step
        source = tl.where(lane + offset < 32, lane + offset, lane)
        total = total + tl.gather(total, source, 0)
    return tl.gather(total, tl.full((1,), 0, tl.int32), 0) * (1.0 / 128.0)


@triton.jit
def _silu_rms(a, b, DTYPE: tl.constexpr):
    # ATen materializes SiLU, its product and the normalized result separately.
    activated = libdevice.div_rn(a, 1.0 + libdevice.exp(-a)).to(DTYPE).to(tl.float32)
    product = (activated * b).to(DTYPE).to(tl.float32)
    inverse = libdevice.rsqrt(_mean_square_128(product) + EPS)
    return (product * inverse).to(DTYPE)


@triton.jit
def _silu_rms_kernel(A, B, OUT,
                     A_B: tl.constexpr, A_T: tl.constexpr, A_H: tl.constexpr, A_D: tl.constexpr,
                     B_B: tl.constexpr, B_T: tl.constexpr, B_H: tl.constexpr, B_D: tl.constexpr,
                     T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    batch, token, head = row // (T * H), row // H % T, row % H
    dim = tl.arange(0, BLOCK)
    a = tl.load(A + batch * A_B + token * A_T
                + head * A_H + dim * A_D, dim < D, other=0).to(tl.float32)
    b = tl.load(B + batch * B_B + token * B_T
                + head * B_H + dim * B_D, dim < D, other=0).to(tl.float32)
    tl.store(OUT + row * D + dim, _silu_rms(a, b, A.dtype.element_ty), dim < D)


def silu_rms(a, b):
    """Fuse the pointwise tail of two existing [B,T,H,D] projections."""
    batch, tokens, heads, dim = a.shape
    if dim != 128 or batch * tokens * heads < 16:
        return rms_no_affine(F.silu(a) * b)
    out = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    _silu_rms_kernel[(batch * tokens * heads,)](
        a, b, out, *a.stride(), *b.stride(), tokens, heads, dim,
        triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _readout_kernel(MEMORY, GATE, LOCAL, VALID, OUT,
                    M_B: tl.constexpr, M_T: tl.constexpr, M_H: tl.constexpr, M_D: tl.constexpr,
                    G_B: tl.constexpr, G_T: tl.constexpr, G_H: tl.constexpr,
                    L_B: tl.constexpr, L_T: tl.constexpr, L_H: tl.constexpr, L_D: tl.constexpr,
                    V_B: tl.constexpr, V_T: tl.constexpr,
                    T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                    NORMALIZE: tl.constexpr, HAS_LOCAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    batch, token, head = row // (T * H), row // H % T, row % H
    dim = tl.arange(0, BLOCK)
    memory = tl.load(MEMORY + batch * M_B + token * M_T
                     + head * M_H + dim * M_D, dim < D, other=0).to(tl.float32)
    if NORMALIZE:
        inverse = libdevice.rsqrt(_mean_square_128(memory) + EPS)
        memory = (memory * inverse).to(MEMORY.dtype.element_ty).to(tl.float32)
    gate = tl.load(GATE + batch * G_B + token * G_T + head * G_H).to(tl.float32)
    residual = (gate * memory).to(MEMORY.dtype.element_ty).to(tl.float32)
    if HAS_LOCAL:
        local = tl.load(LOCAL + batch * L_B + token * L_T
                        + head * L_H + dim * L_D, dim < D, other=0).to(tl.float32)
        combined = (local + residual).to(OUT.dtype.element_ty)
    else:
        combined = residual.to(OUT.dtype.element_ty)
    valid = tl.load(VALID + batch * V_B + token * V_T)
    tl.store(OUT + row * D + dim, tl.where(valid, combined, 0), dim < D)


def readout(memory, gate, local, valid, *, normalize=True, has_local=True):
    """Apply the signed gate to the prefix read, with or without a local path."""
    batch, tokens, heads, dim = memory.shape
    pointer = local if has_local else memory
    if dim != 128 or batch * tokens * heads < 16:
        value = rms_no_affine(memory) if normalize else memory
        combined = gate[..., None] * value
        if has_local:
            combined = local + combined
        return combined.masked_fill(~valid[..., None, None], 0)
    out = torch.empty(pointer.shape, device=pointer.device, dtype=pointer.dtype)
    _readout_kernel[(batch * tokens * heads,)](
        memory, gate, pointer, valid, out, *memory.stride(), *gate.stride(),
        *pointer.stride(), *valid.stride(), tokens, heads, dim,
        normalize, has_local, triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False)
    return out
