"""Experimental one-warp launches of the existing rounded feature kernels.

Only launch geometry changes. The Triton source, 128-element reduction order,
dtype rounding points and disabled FP fusion are shared with production.
Use patch() only inside the single-threaded benchmark; it restores all bindings.
"""

from contextlib import contextmanager

import torch

from prefix_ttt.ops import fused_features


_ORIGINAL_SILU_RMS = fused_features.silu_rms
_ORIGINAL_READOUT = fused_features.readout


def silu_rms(a, b):
    """Use one warp per supported 128-element projection row."""
    if not (a.is_cuda and b.is_cuda and a.ndim == 4 and a.shape == b.shape
            and a.shape[-1] == 128 and a.numel() // 128 >= 16
            and a.dtype == b.dtype and a.dtype in (torch.bfloat16, torch.float32)):
        return _ORIGINAL_SILU_RMS(a, b)
    batch, tokens, heads, dim = a.shape
    out = torch.empty(a.shape, device=a.device, dtype=a.dtype)
    fused_features._silu_rms_kernel[(batch * tokens * heads,)](
        a, b, out, *a.stride(), *b.stride(), tokens, heads, dim, 128,
        num_warps=1, enable_fp_fusion=False)
    return out


def readout(memory, gate, local, valid):
    """Use one warp per supported 128-element memory readout row."""
    if not (memory.is_cuda and local.is_cuda and gate.is_cuda and valid.is_cuda
            and memory.ndim == 4 and memory.shape == local.shape
            and memory.shape[-1] == 128 and memory.numel() // 128 >= 16
            and memory.dtype == local.dtype
            and memory.dtype in (torch.bfloat16, torch.float32)):
        return _ORIGINAL_READOUT(memory, gate, local, valid)
    batch, tokens, heads, dim = memory.shape
    out = torch.empty(local.shape, device=local.device, dtype=local.dtype)
    fused_features._readout_kernel[(batch * tokens * heads,)](
        memory, gate, local, valid, out, *memory.stride(), *gate.stride(),
        *local.stride(), *valid.stride(), tokens, heads, dim, 128,
        num_warps=1, enable_fp_fusion=False)
    return out


@contextmanager
def patch(forward_globals=None):
    """Patch current bindings plus an optional frozen forward's globals.

    Nest inside ``layers.use('optimized')`` and pass
    ``layers.baseline_forward.__globals__`` for mamba_bench's frozen forward.
    FeatureReadout.features_prefill imports silu_rms at call time, so needs no patch.
    """
    from prefix_ttt.model import hybrid

    bindings = [(fused_features.__dict__, "silu_rms", silu_rms),
                (fused_features.__dict__, "readout", readout),
                (hybrid.__dict__, "fused_readout", readout)]
    if forward_globals is not None and forward_globals is not hybrid.__dict__:
        bindings.append((forward_globals, "fused_readout", readout))
    saved = [(namespace, key, namespace[key]) for namespace, key, _ in bindings]
    try:
        for namespace, key, value in bindings:
            namespace[key] = value
        yield
    finally:
        for namespace, key, value in reversed(saved):
            namespace[key] = value
