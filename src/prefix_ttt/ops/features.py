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

    def features(self, x):
        a = torch.einsum("bthd,hdr->bthr", x, self.a_phi)
        b = torch.einsum("bthd,hdr->bthr", x, self.b_phi)
        return rms_no_affine(F.silu(a) * b)

    def readout(self, x, memory):
        gate = F.linear(x, self.gate_weight)
        return gate.unsqueeze(-1) * rms_no_affine(memory)
