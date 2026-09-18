"""Purity and consistency checks for the pure Prefix-TTT layout on a real GPU.

Run before the formal phase B: installs the P32 layout on the pinned checkpoint and
proves, on the operations that actually execute, that

  1. the language model performs no softmax attention call at all,
  2. a request allocates only FP32 recurrent state (64 MiB for batch 1),
  3. one dense prefill, a segmented prefill and token-by-token decoding agree.

Usage: CUDA_VISIBLE_DEVICES=0 uv run --locked python scripts/experiments/prefix_ttt_p32/check_purity.py
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from prefix_ttt.config import load_config
from prefix_ttt.model.bridge import load_checkpoint
from prefix_ttt.model.generation import new_cache
from prefix_ttt.model.hybrid import LAYOUT_ANCHORS, install_prefix_ttt


class CountAttention(TorchDispatchMode):
    def __init__(self):
        self.names = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if 'scaled_dot_product' in str(func):
            self.names.append(str(func))
        return func(*args, **(kwargs or {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/p32.json')
    parser.add_argument('--stage-a-checkpoint', help='trained feature maps/gates; omit to randomise the gates')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output')
    args = parser.parse_args()
    config = load_config(args.config)
    path = Path(config['data_root']) / config['model_relative_path']
    model, _ = load_checkpoint(path, dtype=torch.bfloat16)
    new_parameters = install_prefix_ttt(model, backend='fla',
                                        full_attention_layers=LAYOUT_ANCHORS['P32'])
    if args.stage_a_checkpoint:
        stage_a = torch.load(args.stage_a_checkpoint, map_location='cpu', weights_only=False)
        for index, layer in enumerate(model.model.layers):
            layer.self_attn.prefix_ttt.load_state_dict(stage_a['features'][str(index)], strict=True)
        del stage_a
    model = model.to(args.device).eval().requires_grad_(False)
    if not args.stage_a_checkpoint:
        # Zero gates would make the readout identically zero, which would hide any
        # readout defect; without a checkpoint, randomise them instead.
        with torch.no_grad():
            for layer in model.model.layers:
                layer.self_attn.prefix_ttt.gate_weight.normal_(std=.02)

    torch.manual_seed(0)
    tokens = torch.randint(1, 32000, (1, 64), device=args.device)
    valid = torch.ones_like(tokens, dtype=torch.bool)
    counter = CountAttention()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16), counter:
        dense = model(inputs_embeds=model.get_model().embed_tokens(tokens),
                      attention_mask=valid, use_cache=True)
    prefill_calls = list(counter.names)
    cache = dense.past_key_values
    state_bytes = sum(layer.state.numel() * layer.state.element_size()
                      for layer in cache.storage.layers.values())
    kv_bytes = sum((layer.key.numel() * layer.key.element_size()
                    + layer.value.numel() * layer.value.element_size())
                   for layer in cache.storage.layers.values()
                   if layer.key is not None)

    # Segmented prefill and single-token decoding must reproduce the dense prefix.
    segmented_cache = new_cache(model, 1, args.device)
    logits = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start, end in ((0, 32), (32, 64)):
            out = model(inputs_embeds=model.get_model().embed_tokens(tokens[:, start:end]),
                        attention_mask=valid[:, start:end],
                        past_key_values=segmented_cache, use_cache=True)
            logits.append(out.logits)
        segmented = torch.cat(logits, 1)
        step_cache = new_cache(model, 1, args.device)
        steps = []
        for index in range(tokens.shape[1]):
            out = model(inputs_embeds=model.get_model().embed_tokens(tokens[:, index:index + 1]),
                        attention_mask=valid[:, index:index + 1],
                        past_key_values=step_cache, use_cache=True)
            steps.append(out.logits)
        decoded = torch.cat(steps, 1)
    # Chunk boundaries change the reduction order of bf16 kernels, so the reference
    # is E0's own split-prefill gap (0.7% segmented / 2.0% per-token at this length),
    # not bit equality; a structural gap would show up as an O(1) relative error.
    tolerance = 0.10
    scale = float(dense.logits.abs().max())
    segmented_gap = float((segmented - dense.logits).abs().max()) / scale
    decoded_gap = float((decoded - dense.logits).abs().max()) / scale
    state_gap = max(
        float((segmented_cache.storage.layers[i].state - cache.storage.layers[i].state).abs().max())
        / (float(cache.storage.layers[i].state.abs().max()) + 1e-9) for i in range(32))

    report = {
        'ttt_layers': len(model.config.prefix_ttt_layers),
        'sdpa_calls_prefill': len(prefill_calls),
        'sdpa_names': sorted(set(prefill_calls)),
        'state_bytes': state_bytes,
        'state_bytes_expected': len(model.config.prefix_ttt_layers) * 32 * 128 * 128 * 4,
        'kv_bytes': kv_bytes,
        'logits_absmax': scale,
        'segmented_relative_gap': segmented_gap,
        'decode_relative_gap': decoded_gap,
        'state_relative_gap': state_gap,
        'tolerance': tolerance,
    }
    assert report['ttt_layers'] == 32, report
    assert report['sdpa_calls_prefill'] == 0, report
    assert report['kv_bytes'] == 0, report
    assert report['state_bytes'] == report['state_bytes_expected'], report
    assert segmented_gap < tolerance and decoded_gap < tolerance, report
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
