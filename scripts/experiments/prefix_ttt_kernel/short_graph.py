"""Short-prefill experiment: fair eager/eager and CUDA Graph/Graph comparisons.

Run through the project GPU queue. Graphs specialize a fresh, all-valid prefill
to one shape and prepared inference weights. Replay includes device-side cache
initialization; Python cache construction, capture, warmup and input copies are
preparation costs, not replay latency. This is not a production graph wrapper.
"""

import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

import torch

from mamba_bench import (BASELINE_COMMIT, DIM, HEADS, HEAD_DIM, Layers, csv_ints,
                         difference, distribution, file_sha256, validate_baseline)


@contextmanager
def variant_context(layers, name):
    with layers.use('optimized' if name == 'warp1' else name):
        if name == 'warp1':
            from short_ops import patch
            with patch(layers.baseline_forward.__globals__):
                yield
        else:
            yield


def forward(layers, x, name):
    # Always construct a new cache. Reusing one here would benchmark decode or
    # accidentally accumulate earlier replay state instead of independent input.
    return layers.forward(x, 'optimized' if name == 'warp1' else name, 1)


def snapshot(outcome):
    output, cache = outcome
    tensors = {'output': output}
    if hasattr(cache, 'storage'):
        tensors.update(seen_tokens=cache.storage.seen_tokens,
                       finished=cache.storage.finished, valid_history=cache.valid_history)
        for index, layer in cache.storage.layers.items():
            tensors.update({f'layer{index}.{key}': value for key, value in vars(layer).items()
                            if value is not None})
        assert cache.current_valid is None
    else:
        for index, (key, value) in enumerate(zip(cache.key_cache, cache.value_cache)):
            tensors[f'layer{index}.key'] = key
            tensors[f'layer{index}.value'] = value
    return {'tensors': {key: value.clone() for key, value in tensors.items()},
            'sequence_length': cache.get_seq_length()}


def compare(expected, actual):
    if expected['tensors'].keys() != actual['tensors'].keys():
        return {'passed': False, 'error': 'output/cache fields changed'}
    fields = {}
    for key, reference in expected['tensors'].items():
        candidate = actual['tensors'][key]
        if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
            fields[key] = {'passed': False, 'error': 'shape or dtype changed'}
        elif reference.is_floating_point():
            fields[key] = difference(reference, candidate)
            limit = .02 if key == 'output' else .01
            fields[key]['relative_l2_limit'] = limit
            fields[key]['passed'] = fields[key]['finite'] and fields[key]['relative_l2'] <= limit
        else:
            fields[key] = {'equal': torch.equal(reference, candidate)}
            fields[key]['passed'] = fields[key]['equal']
    sequence_equal = expected['sequence_length'] == actual['sequence_length']
    return {'fields': fields, 'sequence_length_equal': sequence_equal,
            'passed': sequence_equal and all(field['passed'] for field in fields.values())}


def capture(layers, x, name, warmup):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    start = time.perf_counter()
    with variant_context(layers, name):
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                outcome = forward(layers, x, name)
                del outcome
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outcome = forward(layers, x, name)
        torch.cuda.synchronize()
    return graph, outcome, (time.perf_counter() - start) * 1000


def validate_graph(layers, x, original, changed, name, graph, outcome, reference):
    checks = {}
    for label, source in (('original', original), ('changed_input', changed),
                          ('repeat_changed_input', changed), ('restored_input', original)):
        x.copy_(source)
        with variant_context(layers, name):
            expected = snapshot(forward(layers, x, name))
        graph.replay()
        actual = snapshot(outcome)
        checks[label] = compare(expected, actual)
        if label == 'changed_input':
            checks['input_affects_output'] = not torch.equal(
                reference['tensors']['output'], actual['tensors']['output'])
            state_key = 'layer0.key' if name == 'mha' else 'layer0.state'
            checks['input_affects_cache'] = not torch.equal(
                reference['tensors'][state_key], actual['tensors'][state_key])
        del expected, actual
    checks['passed'] = (checks['input_affects_output'] and checks['input_affects_cache']
                        and all(checks[key]['passed'] for key in
                                ('original', 'changed_input', 'repeat_changed_input', 'restored_input')))
    return checks


def measure(layers, x, variants, graphs, args):
    names = [f'{mode}-{name}' for mode in ('eager', 'graph') for name in variants]
    samples = {key: [] for key in names}

    def call(mode, name):
        if mode == 'eager':
            return forward(layers, x, name)
        graphs[name][0].replay()
        return graphs[name][1]

    for key in names:
        mode, name = key.split('-', 1)
        with variant_context(layers, name) if mode == 'eager' else nullcontext():
            for _ in range(args.warmup):
                outcome = call(mode, name)
                del outcome
    torch.cuda.synchronize()
    for round_index in range(args.rounds):
        for repetition in range(args.repetitions):
            order = names if (round_index * args.repetitions + repetition) % 2 == 0 else names[::-1]
            for key in order:
                mode, name = key.split('-', 1)
                with variant_context(layers, name) if mode == 'eager' else nullcontext():
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    torch.cuda.synchronize()
                    wall_start = time.perf_counter()
                    start.record()
                    outcome = call(mode, name)
                    end.record()
                    torch.cuda.synchronize()
                    samples[key].append({'round': round_index, 'repetition': repetition,
                                         'cuda_ms': start.elapsed_time(end),
                                         'wall_ms': (time.perf_counter() - wall_start) * 1000})
                    del outcome
    summary = {}
    for key, rows in samples.items():
        summary[key] = {f'{clock}_ms': distribution([row[f'{clock}_ms'] for row in rows])
                        for clock in ('cuda', 'wall')}
        summary[key]['round_cuda_medians'] = [statistics.median(
            row['cuda_ms'] for row in rows if row['round'] == index) for index in range(args.rounds)]
        summary[key]['tokens_per_second'] = x.shape[0] * x.shape[1] * 1000 / summary[key]['cuda_ms']['median']
    for mode in ('eager', 'graph'):
        baseline = summary[f'{mode}-mha']
        for name in variants:
            if name == 'mha':
                continue
            entry = summary[f'{mode}-{name}']
            entry['speedup_vs_same_mode_mha'] = baseline['cuda_ms']['median'] / entry['cuda_ms']['median']
            entry['stable_lead_vs_same_mode_mha'] = (args.rounds >= 5
                and entry['speedup_vs_same_mode_mha'] >= 1.05
                and all(left < right for left, right in zip(
                    entry['round_cuda_medians'], baseline['round_cuda_medians'])))
    return {'samples': samples, 'summary': summary}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=Path('artifacts/mamba-kernel/weights/layer1.pt'))
    parser.add_argument('--lengths', type=csv_ints, default=[512, 640, 1024, 1536, 2048])
    parser.add_argument('--batches', type=csv_ints, default=[1, 4])
    parser.add_argument('--variants', default='optimized,mha', help='optimized,warp1,mha')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.variants = args.variants.split(',')
    if (set(args.variants) - {'optimized', 'warp1', 'mha'} or 'mha' not in args.variants
            or len(set(args.variants)) != len(args.variants)):
        parser.error('variants must be unique and include mha; choose optimized,warp1,mha')
    if not args.lengths or not args.batches or min(args.rounds, args.repetitions, args.warmup,
                                                *args.lengths, *args.batches) < 1:
        parser.error('all counts and shapes must be positive')
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite existing evidence: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}, 'measurements': {},
              'protocol': {'hidden_size': DIM, 'heads': HEADS, 'head_dim': HEAD_DIM,
                           'dtype': 'bfloat16', 'persistent_state_dtype': 'float32',
                           'full_layer': 'QKV, RoPE, features, Local-32, prefix state/readout, gate, O, fresh cache',
                           'ttt_path': 'optimized experiment; known all-valid mask, frozen cache without host validity sync',
                           'comparison': 'eager TTT vs eager Flash SDPA; graph TTT vs graph Flash SDPA',
                           'mha_backend': 'forced torch FLASH_ATTENTION; different mathematical function',
                           'graph_scope': 'fixed shape, prepared weights; capture includes device cache initialization',
                           'excluded_from_replay': 'input copy, Python object construction, warmup, compilation and capture',
                           'validation': 'all output/cache fields; changed input, repeated changed input, restored input',
                           'production_status': 'experiment only; no production graph integration'},
              'source_sha256': {str(path): file_sha256(path) for path in
                                (Path(__file__), Path(__file__).with_name('mamba_bench.py'),
                                 Path(__file__).with_name('short_ops.py'),
                                 Path('src/prefix_ttt/ops/prefill.py')) if path.exists()}}

    def save():
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(args.output)

    stage = 'initialization'
    save()
    try:
        validate_baseline(BASELINE_COMMIT)
        result['weights'] = {'path': str(args.weights), 'bytes': args.weights.stat().st_size,
                             'sha256': file_sha256(args.weights)}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                                 'gpu': torch.cuda.get_device_name(),
                                 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                 'git_diff_sha256': hashlib.sha256(subprocess.check_output(
                                     ['git', 'diff', '--', 'src/prefix_ttt'])).hexdigest(),
                                 'baseline_commit': BASELINE_COMMIT, 'allow_tf32': False}
        layers = Layers(1, args.seed, 64, BASELINE_COMMIT, args.weights)
        for batch in args.batches:
            for length in args.lengths:
                stage = f'b{batch}-t{length}'
                entry = {'correctness': {}, 'preparation_warmup_capture_ms': {}}
                result['measurements'][stage] = entry
                torch.manual_seed(args.seed + batch * 100000 + length)
                original = torch.randn(batch, length, DIM, device='cuda', dtype=torch.bfloat16)
                changed = torch.randn_like(original)
                x = original.clone()
                with variant_context(layers, 'optimized'):
                    optimized_reference = snapshot(forward(layers, x, 'optimized'))
                graphs = {}
                for name in args.variants:
                    with variant_context(layers, name):
                        reference = snapshot(forward(layers, x, name))
                    checks = {}
                    if name != 'mha':
                        checks['vs_optimized'] = compare(optimized_reference, reference)
                    graph, outcome, prep_ms = capture(layers, x, name, args.warmup)
                    graphs[name] = (graph, outcome)
                    entry['preparation_warmup_capture_ms'][name] = prep_ms
                    checks['graph_replay'] = validate_graph(layers, x, original, changed,
                                                           name, graph, outcome, reference)
                    checks['passed'] = all(value['passed'] for value in checks.values())
                    entry['correctness'][name] = checks
                    save()
                    if not checks['passed']:
                        raise AssertionError(f'{name} graph/cache correctness failed')
                    del graph, outcome, reference
                del optimized_reference
                entry.update(measure(layers, x, args.variants, graphs, args))
                save()
                print(json.dumps({'shape': stage, 'summary': entry['summary']}), flush=True)
                del graphs, x, original, changed
                gc.collect()
                torch.cuda.empty_cache()
        result['status'] = 'succeeded'
        save()
    except Exception as exc:
        result['status'] = 'failed'
        result['failure'] = {'stage': stage, 'type': type(exc).__name__, 'message': str(exc),
                             'traceback': traceback.format_exc()}
        save()
        raise


if __name__ == '__main__':
    main()
