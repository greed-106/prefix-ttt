"""Fixed-input Prefix-TTT/Flash SDPA inference benchmark; run through the queue.

Timing includes fresh cache construction, QKV, RoPE and O projection for complete
layers. Softmax and linear attention compute different functions. Profiling is a
separate stage, never part of the reported timing samples.
"""

import argparse
import ast
from collections import Counter
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

from prefix_ttt.model.generation import TransformersHybridCache
from prefix_ttt.model.hybrid import PrefixTTTAttention
from prefix_ttt.ops.features import FeatureReadout
from prefix_ttt.ops.fla import fla_prefix
from prefix_ttt.ops.fused_rotary import rotary
from prefix_ttt.ops.local import dense_local


BASELINE_COMMIT = 'eb3d2c61054686b596b2e407767f546ac99e9525'
DIM, HEADS, HEAD_DIM = 4096, 32, 128
VARIANTS = ('baseline', 'features', 'dense', 'optimized', 'tile128', 'production', 'mha')
BASELINE_DEPENDENCIES = ('ops/__init__.py', 'ops/fused_features.py', 'ops/fused_recurrent.py',
                         'ops/fused_rotary.py', 'ops/reference.py', 'cache.py')
BASELINE_OPERATORS = {'ops/fla.py': ('fla_prefix', 'recurrent_step'),
                      'ops/local.py': ('local_attention', 'local_attention_cached',
                                       'local_attention_decode', 'LocalCache')}


def committed_class(path, name, commit):
    source = subprocess.check_output(['git', 'show', f'{commit}:src/prefix_ttt/{path}'], text=True)
    namespace = {'__name__': 'mamba_bench_baseline'}
    exec(compile(source, f'{commit}/{path}', 'exec'), namespace)
    if path == 'model/hybrid.py':
        # Freeze whole operator namespaces, including their internal calls.
        # Remaining project imports are checked by validate_baseline().
        for relative, names in BASELINE_OPERATORS.items():
            source = subprocess.check_output(
                ['git', 'show', f'{commit}:src/prefix_ttt/{relative}'], text=True)
            operators = {'__name__': 'mamba_bench_baseline'}
            exec(compile(source, f'{commit}/{relative}', 'exec'), operators)
            namespace.update({key: operators[key] for key in names})
    return namespace[name]


def validate_baseline(commit):
    """Validate live dependencies of the Git-frozen forward and operators."""
    for relative in BASELINE_DEPENDENCIES:
        path = 'src/prefix_ttt/' + relative
        expected = subprocess.check_output(['git', 'show', f'{commit}:{path}'])
        if Path(path).read_bytes() != expected:
            raise RuntimeError(f'Baseline dependency changed: {path}; freeze it before comparing')
    path = 'src/prefix_ttt/ops/features.py'
    expected = subprocess.check_output(['git', 'show', f'{commit}:{path}'], text=True)
    def methods(source):
        cls = next(node for node in ast.parse(source).body
                   if isinstance(node, ast.ClassDef) and node.name == 'FeatureReadout')
        return {node.name: ast.dump(node) for node in cls.body if isinstance(node, ast.FunctionDef)}
    old, current = methods(expected), methods(Path(path).read_text())
    for method in ('__init__', 'features', 'readout', 'inference_weight'):
        if old[method] != current[method]:
            raise RuntimeError(f'Baseline FeatureReadout.{method} changed; freeze it before comparing')


@contextmanager
def implementation(name, baseline_forward, baseline_pair, profile=False):
    """Only this experiment swaps kernels; baseline forward is frozen in Git."""
    globals_ = (PrefixTTTAttention.forward if name == 'production' else baseline_forward).__globals__
    saved = {key: globals_[key] for key in ('fla_prefix', 'local_attention_cached',
                                           'fused_rotary', 'fused_readout',
                                           'dense_local') if key in globals_}
    old_pair = FeatureReadout.features_pair
    old_prefill = FeatureReadout.features_prefill
    if name != 'production':
        FeatureReadout.features_pair = baseline_pair
    try:
        if name in ('features', 'dense', 'optimized', 'tile128'):
            if name in ('features', 'optimized', 'tile128'):
                def pair(module, q, k):
                    return (module.features_prefill(q, k) if q.shape[1] > 1
                            else baseline_pair(module, q, k))
                FeatureReadout.features_pair = pair
            if name in ('dense', 'optimized', 'tile128'):
                def prefix(q, k, v, initial_state=None, need_final_state=True,
                           valid=None, tile_size=64):
                    return fla_prefix(q, k, v, initial_state, need_final_state,
                                      tile_size=128 if name == 'tile128' else tile_size)
                def local(q, k, v, valid=None, cache=None, finished=None):
                    # All benchmark prompts are valid and start at logical zero.
                    if cache is None:
                        return dense_local(q, k, v, need_cache=True)
                    return saved['local_attention_cached'](q, k, v, valid, cache, finished)
                globals_['fla_prefix'], globals_['local_attention_cached'] = prefix, local
        if profile:
            for key in saved:
                globals_[key] = record_call(key, globals_[key])
            FeatureReadout.features_pair = record_call('features_pair', FeatureReadout.features_pair)
            FeatureReadout.features_prefill = record_call('features_prefill', FeatureReadout.features_prefill)
        yield
    finally:
        globals_.update(saved)
        FeatureReadout.features_pair = old_pair
        FeatureReadout.features_prefill = old_prefill


def record_call(name, call):
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function('prefix_bench::' + name):
            return call(*args, **kwargs)
    return wrapped


class Layers:
    """Projection weights shared between comparison arms, independent by depth."""
    def __init__(self, depth, seed, tile_size, commit, weights=None):
        torch.manual_seed(seed)
        config = LlamaConfig(hidden_size=DIM, intermediate_size=11008,
                             num_attention_heads=HEADS, num_key_value_heads=HEADS,
                             num_hidden_layers=depth, max_position_embeddings=65536,
                             attention_dropout=0.0, attention_bias=False)
        state = None if weights is None else torch.load(weights, map_location='cpu', weights_only=True)
        weight_keys = {f'{name}_proj.weight' for name in ('q', 'k', 'v', 'o')}
        weight_keys.update('prefix_ttt.' + name for name in ('a_phi', 'b_phi', 'gate_weight'))
        if state is not None and set(state) != weight_keys:
            raise ValueError('weights must contain exactly the seven attention state_dict tensors')
        self.layers = []
        for index in range(depth):
            original = LlamaAttention(config, layer_idx=index).to(device='cuda', dtype=torch.bfloat16)
            layer = PrefixTTTAttention(original, backend='fla', seed=seed, tile_size=tile_size)
            with torch.no_grad():
                layer.prefix_ttt.gate_weight.normal_(std=DIM ** -0.5)
                if state is not None:
                    layer.load_state_dict(state, strict=True)
            layer.prefix_ttt.prepare_inference()
            # This nonpersistent candidate buffer keeps FP32 masters untouched.
            if not hasattr(layer.prefix_ttt, 'ab_phi_inference'):
                layer.prefix_ttt.register_buffer('ab_phi_inference', torch.cat((
                    layer.prefix_ttt.a_phi.to(torch.bfloat16),
                    layer.prefix_ttt.b_phi.to(torch.bfloat16)), dim=-1), persistent=False)
            layer.eval().requires_grad_(False)
            self.layers.append(layer)
        self.rope = LlamaRotaryEmbedding(config, device='cuda')
        self.baseline_forward = committed_class('model/hybrid.py', 'PrefixTTTAttention', commit).forward
        self.baseline_pair = committed_class('ops/features.py', 'FeatureReadout', commit).features_pair
        self.baseline_cache = committed_class('model/generation.py', 'TransformersHybridCache', commit)
        self.baseline_forward.__globals__['TransformersHybridCache'] = self.baseline_cache

    @contextmanager
    def use(self, name, profile=False):
        backend = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if name == 'mha' else nullcontext()
        with implementation(name, self.baseline_forward, self.baseline_pair, profile), backend:
            handles = []
            if profile:
                for layer in self.layers:
                    for label in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
                        module = getattr(layer, label)
                        original = module.forward
                        module.forward = record_call(label, original)
                        handles.append((module, original))
            try:
                yield
            finally:
                for module, original in handles:
                    module.forward = original

    def forward(self, x, name, depth, cache=None):
        batch, tokens = x.shape[:2]
        start = 0 if cache is None else cache.get_seq_length()
        valid = torch.ones((batch, tokens), device=x.device, dtype=torch.bool)
        positions = torch.arange(start, start + tokens, device=x.device)[None].expand(batch, -1)
        embeddings = self.rope(x, positions)
        if name == 'mha':
            cache = DynamicCache() if cache is None else cache
            for index, layer in enumerate(self.layers[:depth]):
                shape = (batch, tokens, HEADS, HEAD_DIM)
                q, k = (projection(x).view(shape).transpose(1, 2)
                        for projection in (layer.q_proj, layer.k_proj))
                v = layer.v_proj(x).view(shape).transpose(1, 2)
                q, k = rotary(q, k, *embeddings)
                k, v = cache.update(k, v, index)
                # Cached single-token queries can see the entire accumulated KV.
                if start and tokens != 1:
                    raise ValueError('MHA cached benchmark supports single-token decode only')
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                                     is_causal=start == 0)
                x = layer.o_proj(out.transpose(1, 2).reshape(batch, tokens, DIM))
        else:
            cache_class = TransformersHybridCache if name == 'production' else self.baseline_cache
            cache = cache_class(batch, [], x.device) if cache is None else cache
            cache.begin(valid)
            for layer in self.layers[:depth]:
                forward = PrefixTTTAttention.forward if name == 'production' else self.baseline_forward
                x, _ = forward(layer, x, embeddings, past_key_value=cache,
                               prefix_valid_mask=valid, use_cache=True)
            cache.finish()
        return x, cache


def cache_bytes(cache):
    if isinstance(cache, torch.Tensor):
        return cache.numel() * cache.element_size()
    if hasattr(cache, 'storage'):
        return cache.storage.tensor_bytes + cache.valid_history.numel() * cache.valid_history.element_size()
    if isinstance(cache, DynamicCache):
        return sum(x.numel() * x.element_size() for x in (*cache.key_cache, *cache.value_cache)
                   if isinstance(x, torch.Tensor))
    if isinstance(cache, (list, tuple)):
        return sum(cache_bytes(x) for x in cache)
    return 0


def distribution(values):
    ordered = sorted(values)
    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {'median': statistics.median(ordered), 'p10': quantile(.1), 'p90': quantile(.9)}


def measure(calls, contexts, prepare, args, token_count, steps=1):
    """Alternate variants for every sample; each sample has its own sync boundary."""
    names = list(calls)
    for name in names:
        with contexts(name):
            for _ in range(args.warmup):
                prepared = prepare(name)
                outcome = calls[name](prepared)
                del outcome, prepared
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    for round_index in range(args.rounds):
        for repetition in range(args.repetitions):
            order = names if (round_index * args.repetitions + repetition) % 2 == 0 else names[::-1]
            for name in order:
                with contexts(name):
                    prepared = prepare(name)
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    torch.cuda.synchronize()
                    wall_start = time.perf_counter()
                    start.record()
                    outcome = calls[name](prepared)
                    end.record()
                    torch.cuda.synchronize()
                    wall_ms = (time.perf_counter() - wall_start) * 1000
                    samples[name].append({'round': round_index, 'repetition': repetition,
                                          'cuda_ms': start.elapsed_time(end), 'wall_ms': wall_ms})
                    del outcome, prepared
    memory = {}
    for name in names:
        with contexts(name):
            gc.collect()
            torch.cuda.synchronize()
            prepared = prepare(name)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            outcome = calls[name](prepared)
            torch.cuda.synchronize()
            retained = torch.cuda.memory_allocated()
            memory[name] = {'allocated_before_bytes': allocated,
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'peak_increment_bytes': torch.cuda.max_memory_allocated() - allocated,
                            'retained_increment_bytes': retained - allocated,
                            'temporary_over_retained_bytes': torch.cuda.max_memory_allocated() - retained,
                            'retained_cache_bytes': cache_bytes(outcome[1]),
                            'output_bytes': outcome[0].numel() * outcome[0].element_size(),
                            'output_finite': bool(torch.isfinite(outcome[0]).all())}
            del outcome, prepared
    summary = {}
    for name, rows in samples.items():
        cuda = distribution([row['cuda_ms'] for row in rows])
        wall = distribution([row['wall_ms'] for row in rows])
        summary[name] = {'cuda_ms': cuda, 'wall_ms': wall,
                         'tokens_per_second': token_count * 1000 / cuda['median'],
                         'cuda_ms_per_step': cuda['median'] / steps,
                         'wall_ms_per_step': wall['median'] / steps,
                         'round_cuda_medians': [statistics.median(row['cuda_ms'] for row in rows
                                                               if row['round'] == index)
                                                for index in range(args.rounds)],
                         'memory': memory[name]}
    if 'mha' in summary:
        for name, entry in summary.items():
            if name == 'mha':
                continue
            entry['speedup_vs_mha'] = summary['mha']['cuda_ms']['median'] / entry['cuda_ms']['median']
            entry['stable_lead_vs_mha'] = (args.rounds >= 5 and entry['speedup_vs_mha'] >= 1.05 and all(
                left < right for left, right in zip(entry['round_cuda_medians'],
                                                    summary['mha']['round_cuda_medians'])))
    return {'samples': samples, 'summary': summary}


def difference(reference, candidate):
    expected, actual = reference.float(), candidate.float()
    delta = actual - expected
    return {'finite': bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()),
            'max_abs': delta.abs().max().item(),
            'relative_l2': (delta.norm() / expected.norm().clamp_min(1e-30)).item(),
            'equal': torch.equal(reference, candidate)}


def correctness(layers, x, variants):
    with layers.use('baseline'):
        expected, reference_cache = layers.forward(x, 'baseline', 1)
    reference = reference_cache.storage.layers[0]
    results = {}
    for name in variants:
        if name in ('baseline', 'mha'):
            continue
        with layers.use(name):
            actual, candidate_cache = layers.forward(x, name, 1)
        candidate = candidate_cache.storage.layers[0]
        entry = {'output': difference(expected, actual),
                 'state': difference(reference.state, candidate.state),
                 'local_key': difference(reference.key, candidate.key),
                 'local_value': difference(reference.value, candidate.value),
                 'local_position_equal': torch.equal(reference.local_position, candidate.local_position),
                 'seen_tokens_equal': torch.equal(reference_cache.storage.seen_tokens,
                                                   candidate_cache.storage.seen_tokens)}
        entry['passed'] = (all(entry[key]['finite'] for key in ('output', 'state', 'local_key', 'local_value'))
                           and entry['output']['relative_l2'] <= 0.02
                           and entry['state']['relative_l2'] <= 0.01
                           and entry['local_position_equal'] and entry['seen_tokens_equal'])
        results[name] = entry
        del actual, candidate_cache
    return {'limits': {'output_relative_l2': 0.02, 'state_relative_l2': 0.01},
            'variants': results, 'passed': all(entry['passed'] for entry in results.values())}


def profile_call(call, trace_path, require_flash=False):
    call()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True, profile_memory=True) as trace:
        outcome = call()
        torch.cuda.synchronize()
    del outcome
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace.export_chrome_trace(str(trace_path))
    kernels, transfers, device_us = Counter(), Counter(), Counter()
    # Chrome categories distinguish actual kernel launches from GPU annotations.
    # FunctionEvent CUDA entries can also contain prefix_bench user ranges.
    for event in json.loads(trace_path.read_text())['traceEvents']:
        if event.get('cat') == 'kernel' and event.get('ph') == 'X':
            kernels[event['name']] += 1
            device_us[event['name']] += event['dur']
        elif event.get('cat') in ('gpu_memcpy', 'gpu_memset') and event.get('ph') == 'X':
            transfers[event['name']] += 1
    if not kernels:
        raise RuntimeError('Profiler trace contains no CUDA kernel-category events')
    averages = trace.key_averages()
    operators = [event.key for event in averages]
    verified = 'aten::_scaled_dot_product_flash_attention' in operators
    if require_flash and not verified:
        raise RuntimeError('Profiler did not observe the required Flash SDPA operator')
    return {'kernel_count': sum(kernels.values()), 'kernels': dict(kernels),
            'kernel_device_us': dict(device_us), 'memory_activity_count': dict(transfers),
            'flash_operator_verified': verified, 'trace': str(trace_path),
            'components': [{'name': event.key, 'count': event.count,
                            'cpu_total_us': event.cpu_time_total,
                            'device_total_us': event.device_time_total,
                            'self_cpu_memory_bytes': event.self_cpu_memory_usage}
                           for event in averages if event.key.startswith('prefix_bench::')],
            'cpu_top': [{'name': event.key, 'count': event.count,
                         'self_cpu_us': event.self_cpu_time_total}
                        for event in sorted(averages, key=lambda x: x.self_cpu_time_total,
                                            reverse=True)[:25]]}


def csv_ints(value):
    return [int(item) for item in value.split(',')]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, help='torch.save of seven attention state_dict tensors')
    parser.add_argument('--stages', default='layer', help='comma-separated core,layer,stack,profile,decode,correctness')
    parser.add_argument('--variants', default='baseline,production,mha', help=','.join(VARIANTS))
    parser.add_argument('--lengths', type=csv_ints, default=csv_ints('512,640,1024,1536,2048,4096,8192,16384,32768'))
    parser.add_argument('--batches', type=csv_ints, default=[1, 4])
    parser.add_argument('--stack-lengths', type=csv_ints, default=[1536, 8192, 32768])
    parser.add_argument('--decode-lengths', type=csv_ints, default=[640, 1536])
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--decode-steps', type=int, default=128)
    parser.add_argument('--tiles', type=csv_ints, default=[16, 32, 64, 128], help='core tile ablation')
    parser.add_argument('--tile-size', type=int, choices=[16, 32, 64, 128], default=64)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--baseline-commit', default=BASELINE_COMMIT)
    args = parser.parse_args()
    args.stages, args.variants = args.stages.split(','), args.variants.split(',')
    if set(args.stages) - {'core', 'layer', 'stack', 'profile', 'decode', 'correctness'}:
        parser.error('unknown stage')
    if set(args.variants) - set(VARIANTS):
        parser.error('unknown variant')
    if min(args.rounds, args.repetitions, args.depth, args.decode_steps, *args.lengths,
           *args.batches, *args.stack_lengths, *args.decode_lengths) < 1 or args.warmup < 0:
        parser.error('counts and shapes must be positive; warmup may be zero')
    if set(args.tiles) - {16, 32, 64, 128}:
        parser.error('tiles must be 16,32,64,128')
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing evidence: {args.output}')
    validate_baseline(args.baseline_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    result = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
              'environment': {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                              'cuda': torch.version.cuda,
                              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                              'git_diff_sha256': hashlib.sha256(subprocess.check_output(
                                  ['git', 'diff', '--', 'src/prefix_ttt'])).hexdigest(),
                              'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                              'allow_tf32': False},
              'protocol': {'hidden_size': DIM, 'heads': HEADS, 'head_dim': HEAD_DIM,
                           'dtype': 'bfloat16', 'persistent_state_dtype': 'float32', 'dropout': 0,
                           'intermediate_chunk_state': 'locked FLA default states_in_fp32=False; BF16 h boundaries',
                           'gate': ('checkpoint signed gate' if args.weights else 'normal(std=1/sqrt(hidden_size)); nonzero'),
                           'stack_weights': ('single checkpoint layer repeated, distinct caches' if args.weights
                                             else 'independent random weights and caches by depth'),
                           'baseline_dependency_validation': 'Git-frozen forward, FLA and Local; remaining dependencies '
                           'use exact source equality except selected feature methods validated by AST',
                           'variant_tile_override': {'tile128': 128},
                           'mha_cache': 'Transformers DynamicCache, no artificial TTT bookkeeping',
                           'timer': 'per-call CUDA events and synchronized wall time; no CUDA graph',
                           'core': 'precomputed q/k/v; includes final state for TTT, retains k/v for MHA',
                           'claim': 'causal Torch FLASH_ATTENTION only; functions differ'},
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), Path('src/prefix_ttt/ops/fla.py'),
                                             Path('src/prefix_ttt/ops/local.py'),
                                             Path('src/prefix_ttt/model/hybrid.py'),
                                             Path('src/prefix_ttt/ops/features.py')) if path.exists()},
              'baseline_source_sha256': {path: hashlib.sha256(subprocess.check_output(
                  ['git', 'show', f'{args.baseline_commit}:src/prefix_ttt/{path}'])).hexdigest()
                  for path in ('model/hybrid.py', 'model/generation.py', 'ops/features.py', *BASELINE_OPERATORS)},
              'measurements': {}}
    if args.weights:
        result['weights'] = {'path': str(args.weights), 'bytes': args.weights.stat().st_size,
                             'sha256': file_sha256(args.weights)}

    def save():
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(args.output)

    save()
    layers = Layers(args.depth if 'stack' in args.stages else 1,
                    args.seed, args.tile_size, args.baseline_commit, args.weights)
    for stage in args.stages:
        lengths = (args.stack_lengths if stage == 'stack' else args.decode_lengths
                   if stage == 'decode' else args.lengths)
        for batch in args.batches:
            for length in lengths:
                key = f'{stage}-b{batch}-t{length}'
                try:
                    torch.manual_seed(args.seed + batch * 100000 + length)
                    if stage != 'core':
                        x = torch.randn(batch, length, DIM, device='cuda', dtype=torch.bfloat16)
                    if stage == 'core':
                        q, k, v = (torch.randn(batch, length, HEADS, HEAD_DIM, device='cuda',
                                              dtype=torch.bfloat16) for _ in range(3))
                        valid = torch.ones(batch, length, device='cuda', dtype=torch.bool)
                        calls = {}
                        for tile in args.tiles:
                            calls[f'baseline-tile{tile}'] = lambda _, tile=tile: layers.baseline_forward.__globals__['fla_prefix'](
                                q, k, v, valid=valid, tile_size=tile)
                            if set(args.variants) & {'optimized', 'tile128', 'dense', 'production'}:
                                calls[f'optimized-tile{tile}'] = lambda _, tile=tile: fla_prefix(
                                    q, k, v, tile_size=tile)
                        def flash(_):
                            out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                                                 v.transpose(1, 2), is_causal=True)
                            return out, (k, v)
                        calls['mha'] = flash
                        entry = measure(calls, lambda name: sdpa_kernel(SDPBackend.FLASH_ATTENTION)
                                        if name == 'mha' else nullcontext(), lambda name: None,
                                        args, batch * length)
                        del q, k, v, valid
                    elif stage == 'correctness':
                        entry = correctness(layers, x, args.variants)
                    elif stage == 'profile':
                        entry = {}
                        for name in args.variants:
                            with layers.use(name, profile=True):
                                for _ in range(args.warmup):
                                    layers.forward(x, name, 1)
                                trace_path = (Path('artifacts/mamba-kernel/traces') / args.output.stem
                                              / f'{key}-{name}.trace.json')
                                entry[name] = profile_call(lambda: layers.forward(x, name, 1), trace_path,
                                                           require_flash=name == 'mha')
                    else:
                        depth = args.depth if stage == 'stack' else 1
                        if stage == 'decode':
                            continuation = torch.randn(batch, args.decode_steps, DIM,
                                                       device='cuda', dtype=torch.bfloat16)
                            def prepare(name):
                                return layers.forward(x, name, 1)[1]
                            def decode(name, cache):
                                for step in range(args.decode_steps):
                                    outcome = layers.forward(continuation[:, step:step + 1], name, 1, cache)
                                    cache = outcome[1]
                                return outcome
                            calls = {name: lambda cache, name=name: decode(name, cache) for name in args.variants}
                            entry = measure(calls, layers.use, prepare, args,
                                            batch * args.decode_steps, args.decode_steps)
                            del continuation
                        else:
                            calls = {name: lambda _, name=name: layers.forward(x, name, depth)
                                     for name in args.variants}
                            entry = measure(calls, layers.use, lambda name: None, args, batch * length)
                    result['measurements'][key] = entry
                    save()
                    printed = ({name: {field: values[field] for field in
                                       ('kernel_count', 'flash_operator_verified', 'trace')}
                                for name, values in entry.items()} if stage == 'profile'
                               else entry.get('summary', entry))
                    print(json.dumps({'shape': key, 'summary': printed}), flush=True)
                    if stage == 'correctness' and not entry['passed']:
                        raise AssertionError('Complete-layer correctness thresholds failed; inspect saved evidence')
                    if stage != 'core':
                        del x
                    if stage not in ('profile', 'correctness'):
                        del calls
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as exc:
                    result['failure'] = {'shape': key, 'type': type(exc).__name__, 'message': str(exc)}
                    save()
                    raise


if __name__ == '__main__':
    main()
