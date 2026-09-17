"""Eager complete-layer decode against growing and preallocated Flash KV caches.

Only consecutive single-token forwards are timed; prefix computation and cache
allocation precede each sample. No CUDA Graph or production code is installed.
"""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

import torch
from transformers.cache_utils import DynamicCache

from prefix_ttt.cache import HybridCache
from decode_local import patch as local_prepare
from mamba_bench import (BASELINE_COMMIT, DIM, HEADS, HEAD_DIM, Layers,
                         csv_ints, difference, distribution, file_sha256, measure)


VARIANTS = ('production', 'mha_dynamic', 'mha_static', 'cache_upper_bound', 'local_prepare')


class StaticKV(DynamicCache):
    """One-layer benchmark KV allocation; attention sees only initialized slots."""

    def __init__(self, batch, capacity, heads, dim, *, device, dtype):
        super().__init__()
        self.capacity = capacity
        self.length = 0
        self.key_cache = [torch.empty(batch, heads, capacity, dim, device=device, dtype=dtype)]
        self.value_cache = [torch.empty_like(self.key_cache[0])]

    def get_seq_length(self, layer_idx=0):
        if layer_idx != 0:
            raise ValueError('StaticKV benchmark contains exactly one layer')
        return self.length

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx != 0 or key_states.shape != value_states.shape:
            raise ValueError('StaticKV requires matching KV for layer zero')
        stop = self.length + key_states.shape[-2]
        if stop > self.capacity:
            raise ValueError('StaticKV capacity exceeded')
        self.key_cache[0][:, :, self.length:stop].copy_(key_states)
        self.value_cache[0][:, :, self.length:stop].copy_(value_states)
        self.length = stop
        return self.key_cache[0][:, :, :stop], self.value_cache[0][:, :, :stop]


@contextmanager
def cache_upper_bound():
    """Diagnostic only: caller guarantees all tokens valid and no finished rows.

    Skip all set_layer validation and finished merges, without mutating input
    state/KV. This measures an upper bound, not a safe general cache API.
    """
    original = HybridCache.set_layer
    def direct(storage, index, layer):
        storage.layers[index] = layer
    HybridCache.set_layer = direct
    try:
        yield
    finally:
        HybridCache.set_layer = original


def implementation(name):
    return 'mha' if name.startswith('mha_') else 'production'


def prepare(layers, prefix, name, steps):
    arm = implementation(name)
    cache = (StaticKV(prefix.shape[0], prefix.shape[1] + steps, HEADS, HEAD_DIM,
                      device=prefix.device, dtype=prefix.dtype)
             if name == 'mha_static' else None)
    cache = layers.forward(prefix, arm, 1, cache)[1]
    if name == 'cache_upper_bound':
        # Checks happen before timing; Layers.forward generates only valid tokens
        # and no sampling/EOS operation ever marks a row finished in this script.
        if (not bool(cache.valid_history.all()) or bool(cache.storage.finished.any())
                or cache.storage.full_attention_layers):
            raise ValueError('Cache upper bound requires all-valid, unfinished TTT-only input')
    return cache


def decode(layers, continuation, name, cache):
    context = (cache_upper_bound() if name == 'cache_upper_bound' else
               local_prepare() if name == 'local_prepare' else nullcontext())
    with context:
        for step in range(continuation.shape[1]):
            outcome = layers.forward(continuation[:, step:step + 1], implementation(name), 1, cache)
            cache = outcome[1]
    return outcome


def ttt_fields(cache):
    layer = cache.storage.layers[0]
    return dict(vars(layer), seen_tokens=cache.storage.seen_tokens,
                finished=cache.storage.finished, valid_history=cache.valid_history)


def check_pair(layers, prefix, continuation, candidate):
    """Compare every decoded output and cache; profiling/timing excludes this."""
    reference = 'mha_dynamic' if candidate == 'mha_static' else 'production'
    with layers.use(implementation(reference)):
        expected_cache = prepare(layers, prefix, reference, continuation.shape[1])
        actual_cache = prepare(layers, prefix, candidate, continuation.shape[1])
        rows = []
        for step in range(continuation.shape[1]):
            token = continuation[:, step:step + 1]
            old = actual_cache.storage.layers[0].state if candidate != 'mha_static' else None
            old_copy = None if old is None else old.clone()
            expected, expected_cache = decode(layers, token, reference, expected_cache)
            actual, actual_cache = decode(layers, token, candidate, actual_cache)
            output = difference(expected, actual)
            if candidate == 'mha_static':
                length = actual_cache.get_seq_length()
                fields = {name: torch.equal(getattr(expected_cache, name)[0],
                                            getattr(actual_cache, name)[0][:, :, :length])
                          for name in ('key_cache', 'value_cache')}
                fields['length'] = length == expected_cache.get_seq_length() == prefix.shape[1] + step + 1
            else:
                reference_fields, actual_fields = ttt_fields(expected_cache), ttt_fields(actual_cache)
                fields = {key: torch.equal(value, actual_fields[key])
                          for key, value in reference_fields.items()}
                fields['old_state_unchanged'] = torch.equal(old, old_copy)
                fields['state_fp32'] = actual_fields['state'].dtype == torch.float32
            passed = output['equal'] and output['finite'] and all(fields.values())
            rows.append({'step': step, 'output': output, 'cache': fields, 'passed': passed})
            if not passed:
                break
    return {'reference': reference, 'candidate': candidate, 'steps': rows,
            'passed': len(rows) == continuation.shape[1] and all(row['passed'] for row in rows)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path('artifacts/mamba-kernel/weights/layer1.pt'))
    parser.add_argument('--lengths', type=csv_ints, default=[640, 1536])
    parser.add_argument('--batches', type=csv_ints, default=[1])
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.rounds, args.repetitions, *args.lengths, *args.batches) < 1 or args.warmup < 0:
        parser.error('counts and dimensions must be positive; warmup may be zero')
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = [Path(__file__), Path(__file__).with_name('mamba_bench.py'),
               Path(__file__).with_name('decode_local.py')]
    sources.extend(sorted(Path('src/prefix_ttt').rglob('*.py')))
    result = {
        'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'environment': {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                        'cuda': torch.version.cuda, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                        'git_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', '--', 'src/prefix_ttt'])).hexdigest(),
                        'allow_tf32': False},
        'protocol': {'stage': 'complete_attention_layer_decode', 'cuda_graph': False,
                     'hidden_size': DIM, 'heads': HEADS, 'head_dim': HEAD_DIM, 'dtype': 'bfloat16',
                     'persistent_state_dtype': 'float32', 'depth': 1,
                     'input': 'same fixed prefix and teacher-forced hidden vectors; shared checkpoint QKV/O weights',
                     'timed': 'consecutive query-length-one forward calls including KV/state updates; CUDA events and synchronized wall',
                     'excluded': 'prefix forward, cache construction/allocation, correctness, backend context selection',
                     'static_capacity': 'prefix length + steps; only initialized prefix is visible to Flash',
                     'flash_backend': 'torch.nn.attention.sdpa_kernel(FLASH_ATTENTION), no fallback for both MHA arms',
                     'primary_baseline': 'mha_static', 'functions_differ': True,
                     'cache_upper_bound': 'diagnostic skips all set_layer checks/finished merge, all-valid unfinished single-layer only; not deployable',
                     'local_prepare': 'fuse Local KV copies/scatter, visible mask and counters; original SDPA unchanged',
                     'scope_limit': 'single-layer includes one cache.begin/finish per step; whole-model does that once per request, not per TTT layer'},
        'source_sha256': {str(path): file_sha256(path) for path in sources},
        'weights': {'path': str(args.weights), 'bytes': args.weights.stat().st_size,
                    'sha256': file_sha256(args.weights)}, 'measurements': {}}

    def save():
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(args.output)

    save()
    shape = 'initialization'
    try:
        layers = Layers(1, args.seed, 64, BASELINE_COMMIT, args.weights)
        for batch in args.batches:
            for length in args.lengths:
                shape = f'b{batch}-t{length}'
                torch.manual_seed(args.seed + batch * 100000 + length)
                prefix = torch.randn(batch, length, DIM, device='cuda', dtype=torch.bfloat16)
                continuation = torch.randn(batch, args.steps, DIM, device='cuda', dtype=torch.bfloat16)
                entry = result['measurements'][shape] = {'status': 'correctness'}
                entry['correctness'] = {candidate: check_pair(layers, prefix, continuation, candidate)
                                        for candidate in ('mha_static', 'cache_upper_bound', 'local_prepare')}
                save()
                if not all(check['passed'] for check in entry['correctness'].values()):
                    raise AssertionError('Decode output/cache equality failed; see per-step evidence')
                calls = {name: lambda cache, name=name: decode(layers, continuation, name, cache)
                         for name in VARIANTS}
                measured = measure(calls, lambda name: layers.use(implementation(name)),
                                   lambda name: prepare(layers, prefix, name, args.steps), args,
                                   batch * args.steps, args.steps)
                entry.update(measured)
                for name, rows in entry['samples'].items():
                    summary = entry['summary'][name]
                    summary['mean_cuda_ms_per_step'] = statistics.mean(row['cuda_ms'] for row in rows) / args.steps
                    summary['mean_wall_ms_per_step'] = statistics.mean(row['wall_ms'] for row in rows) / args.steps
                    summary['round_cuda_ms_per_step'] = [distribution([row['cuda_ms'] / args.steps for row in rows
                                                                      if row['round'] == index])
                                                         for index in range(args.rounds)]
                    baseline = entry['summary']['mha_static']
                    summary['speedup_vs_static_flash'] = baseline['cuda_ms_per_step'] / summary['cuda_ms_per_step']
                    summary['stable_lead_vs_static_flash'] = (name != 'mha_static' and args.rounds >= 5
                        and summary['speedup_vs_static_flash'] >= 1.05 and all(a < b for a, b in zip(
                            summary['round_cuda_medians'], baseline['round_cuda_medians'])))
                entry['status'] = 'succeeded'
                save()
                print(json.dumps({'shape': shape, 'summary': entry['summary']}), flush=True)
    except Exception as exc:
        result['failure'] = {'shape': shape, 'type': type(exc).__name__, 'message': str(exc)}
        save()
        raise


if __name__ == '__main__':
    main()
