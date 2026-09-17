"""Trainable maps and signed readout; original LLM parameters live elsewhere."""

import math
import torch
from torch import nn
import torch.nn.functional as F


RMS_EPS = 1e-6


def rms_no_affine(x, eps=RMS_EPS):
    stats = x.double() if x.dtype == torch.float64 else x.float()
    return (stats * torch.rsqrt(stats.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


class FeatureReadout(nn.Module):
    def __init__(self, hidden_size=4096, heads=32, head_dim=128, seed=42):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        shape = (heads, head_dim, head_dim)
        self.a_phi = nn.Parameter(torch.randn(shape, generator=generator) / math.sqrt(head_dim))
        self.b_phi = nn.Parameter(torch.randn(shape, generator=generator) / math.sqrt(head_dim))
        self.gate_weight = nn.Parameter(torch.zeros(heads, hidden_size))

    @torch.no_grad()
    def prepare_inference(self):
        """Prepare serving copies after loading; training keeps the FP32 masters."""
        a, b = self.a_phi.to(torch.bfloat16), self.b_phi.to(torch.bfloat16)
        self.register_buffer('packed_phi_inference', torch.cat((a, b, a, b)), persistent=False)
        heads = a.shape[0]
        self.register_buffer('a_phi_inference', self.packed_phi_inference[:heads], persistent=False)
        self.register_buffer('b_phi_inference', self.packed_phi_inference[heads:2 * heads], persistent=False)
        self.register_buffer('ab_phi_inference', torch.cat((a, b), dim=-1), persistent=False)
        self.register_buffer('gate_weight_inference', self.gate_weight.to(torch.bfloat16), persistent=False)

    def inference_weight(self, name, x):
        weight = getattr(self, name)
        if not torch.is_grad_enabled():
            dtype = (torch.get_autocast_dtype(x.device.type)
                     if torch.is_autocast_enabled(x.device.type) else x.dtype)
            if dtype == torch.bfloat16:
                return getattr(self, name + '_inference', weight)
        return weight

    def features(self, x):
        a = torch.einsum("bthd,hdr->bthr", x, self.inference_weight('a_phi', x))
        b = torch.einsum("bthd,hdr->bthr", x, self.inference_weight('b_phi', x))
        if x.is_cuda and not torch.is_grad_enabled() and x.dtype in (torch.bfloat16, torch.float32):
            from prefix_ttt.ops.fused_features import silu_rms
            return silu_rms(a, b)
        return rms_no_affine(F.silu(a) * b)

    def features_pair(self, q, k):
        if (q.is_cuda and not torch.is_grad_enabled()
                and q.dtype == k.dtype == torch.bfloat16
                and q.shape == k.shape == (1, 1, 32, 128)
                and torch.is_autocast_enabled('cuda')
                and torch.get_autocast_dtype('cuda') == torch.bfloat16
                and hasattr(self, 'packed_phi_inference')):
            from prefix_ttt.ops.fused_features import silu_rms
            x = torch.stack((q, q, k, k)).reshape(128, 1, 128)
            projected = torch.bmm(x, self.packed_phi_inference)
            qa, qb, ka, kb = projected.reshape(4, 1, 1, 32, 128).unbind(0)
            return silu_rms(qa, qb), silu_rms(ka, kb)
        return self.features(q), self.features(k)

    def features_prefill(self, q, k):
        """Project Q/K into prepared A/B columns with the same rounding points.

        The serving copy is [H,D,2D]; FP32 masters and the fused SiLU/product/RMS
        stay unchanged. Unsupported inputs retain the autograd feature path.
        """
        if not (q.is_cuda and not torch.is_grad_enabled()
                and q.dtype == k.dtype == torch.bfloat16
                and q.shape == k.shape and q.ndim == 4 and q.shape[-1] == 128
                and (not torch.is_autocast_enabled('cuda')
                     or torch.get_autocast_dtype('cuda') == torch.bfloat16)
                and hasattr(self, 'ab_phi_inference')):
            return self.features(q), self.features(k)
        from prefix_ttt.ops.fused_features import silu_rms

        batch, tokens, heads, dim = q.shape
        outputs = []
        for x in (q, k):
            rows = x.permute(2, 0, 1, 3).reshape(heads, batch * tokens, dim)
            projected = torch.bmm(rows, self.ab_phi_inference)
            projected = projected.reshape(heads, batch, tokens, 2 * dim).permute(1, 2, 0, 3)
            a, b = projected.split(dim, dim=-1)
            outputs.append(silu_rms(a, b))
        return tuple(outputs)

    def readout(self, x, memory):
        gate = F.linear(x, self.inference_weight('gate_weight', x))
        return gate.unsqueeze(-1) * rms_no_affine(memory)
