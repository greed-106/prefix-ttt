"""Same-process current/committed E2 timing; profiling is a separate pass."""

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from prefix_ttt.instrument import cache_buckets
from prefix_ttt.lmms_model import PrefixTTTLlava
from prefix_ttt.model.hybrid import PrefixTTTAttention
import prefix_ttt.model.hybrid as hybrid
from prefix_ttt.ops.features import FeatureReadout


@contextmanager
def implementation(baseline, unpacked=False):
    """Load the committed oracle only in this experiment; no production switch."""
    saved = (PrefixTTTAttention.forward, FeatureReadout.features,
             FeatureReadout.readout, hybrid.fla_prefix, FeatureReadout.features_pair)
    if unpacked:
        FeatureReadout.features_pair = lambda module, q, k: (module.features(q), module.features(k))
    if baseline:
        classes = []
        for path, name in [('model/hybrid.py', 'PrefixTTTAttention'),
                           ('ops/features.py', 'FeatureReadout')]:
            source = subprocess.check_output(
                ['git', 'show', f'16d67fe:src/prefix_ttt/{path}'], text=True)
            namespace = {'__name__': 'committed_oracle'}
            exec(compile(source, f'16d67fe/{path}', 'exec'), namespace)
            classes.append(namespace[name])
        PrefixTTTAttention.forward = classes[0].forward
        FeatureReadout.features = classes[1].features
        FeatureReadout.readout = classes[1].readout
        source = subprocess.check_output(
            ['git', 'show', '16d67fe:src/prefix_ttt/ops/fla.py'], text=True)
        namespace = {'__name__': 'committed_fla'}
        exec(compile(source, '16d67fe/ops/fla.py', 'exec'), namespace)
        hybrid.fla_prefix = namespace['fla_prefix']
        classes[0].forward.__globals__['fla_prefix'] = namespace['fla_prefix']
    try:
        yield
    finally:
        (PrefixTTTAttention.forward, FeatureReadout.features,
         FeatureReadout.readout, hybrid.fla_prefix, FeatureReadout.features_pair) = saved


@torch.inference_mode()
def request(model, inputs, steps):
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(steps + 1)]
    torch.cuda.synchronize()
    start = time.perf_counter()
    events[0][0].record()
    output = model(input_ids=inputs, use_cache=True)
    events[0][1].record()
    logits = [output.logits[:, -1].clone()]
    # Identical teacher-forced continuation for both implementations.
    for step in range(steps):
        events[step + 1][0].record()
        output = model(input_ids=inputs[:, step:step + 1],
                       past_key_values=output.past_key_values, use_cache=True)
        events[step + 1][1].record()
        logits.append(output.logits[:, -1].clone())
    torch.cuda.synchronize()
    wall = (time.perf_counter() - start) * 1000
    result = {'prefill_ms': events[0][0].elapsed_time(events[0][1]),
              'decode_ms': [start.elapsed_time(end) for start, end in events[1:]],
              'forward_loop_wall_ms': wall, 'cache': cache_buckets(output.past_key_values)}
    return result, torch.stack(logits).cpu()


def profile_call(call):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as trace:
        call()
        torch.cuda.synchronize()
    kernels = Counter()
    transfers = Counter()
    device_us = Counter()
    for event in trace.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            if event.name.startswith(('Memcpy', 'Memset')):
                transfers[event.name] += 1
                continue
            kernels[event.name] += 1
            device_us[event.name] += event.device_time_total
    cpu = sorted(trace.key_averages(), key=lambda x: x.self_cpu_time_total, reverse=True)
    return {'kernel_count': sum(kernels.values()), 'device_us': sum(device_us.values()),
            'memory_activity_count': dict(transfers),
            'kernels': dict(kernels), 'kernel_device_us': dict(device_us),
            'cpu_top': [{'name': x.key, 'count': x.count, 'self_us': x.self_cpu_time_total}
                        for x in cpu[:25]]}


@torch.inference_mode()
def profile(model, inputs):
    prefill = profile_call(lambda: model(input_ids=inputs, use_cache=True))
    output = model(input_ids=inputs, use_cache=True)
    decode = profile_call(lambda: model(input_ids=inputs[:, :1],
                                        past_key_values=output.past_key_values, use_cache=True))
    return {'prefill': prefill, 'decode': decode}


def summarize(rows):
    return {'prefill_ms': statistics.median(x['prefill_ms'] for x in rows),
            'tpot_ms': statistics.median(t for x in rows for t in x['decode_ms']),
            'forward_loop_wall_ms': statistics.median(x['forward_loop_wall_ms'] for x in rows),
            'prefill_range_ms': [min(x['prefill_ms'] for x in rows),
                                 max(x['prefill_ms'] for x in rows)],
            'cache': rows[-1]['cache']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lengths', default='640')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=6)
    parser.add_argument('--with-e0', action='store_true')
    parser.add_argument('--with-unpacked', action='store_true')
    args = parser.parse_args()
    result = {'config': vars(args), 'gpu': torch.cuda.get_device_name(),
              'torch': torch.__version__, 'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
              'baseline_commit': '16d67fe', 'measurements': {}}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output_path.write_text(json.dumps(result, indent=2) + '\n')

    variants = ['e0', 'e2'] if args.with_e0 else ['e2']
    for model_name in variants:
        adapter = PrefixTTTLlava(args.pretrained,
                                 checkpoint=args.checkpoint if model_name == 'e2' else None)
        model = adapter.model
        text = ('A conversation about the objects, people and colors in a detailed image. ' * 300)
        tokens = adapter.tokenizer(text, return_tensors='pt', truncation=True,
                                   max_length=max(map(int, args.lengths.split(',')))).input_ids.cuda()
        for length in map(int, args.lengths.split(',')):
            inputs = tokens[:, :length].contiguous()
            names = ['e0'] if model_name == 'e0' else ['current', 'fused']
            if model_name == 'e2' and args.with_unpacked:
                names.insert(1, 'unpacked')
            rows = {name: [] for name in names}
            logits = {}
            for name in names:
                with implementation(name == 'current', name == 'unpacked'):
                    for _ in range(2):
                        request(model, inputs, args.steps)
            for repetition in range(args.repeats):
                for name in names if repetition % 2 == 0 else list(reversed(names)):
                    with implementation(name == 'current', name == 'unpacked'):
                        row, value = request(model, inputs, args.steps)
                    rows[name].append(row)
                    logits[name] = value
            entry = {'samples': rows, 'summary': {name: summarize(values)
                                                   for name, values in rows.items()}}
            for name in names:
                with implementation(name == 'current', name == 'unpacked'):
                    entry[name + '_profile'] = profile(model, inputs)
            if model_name == 'e2':
                delta = logits['fused'].float() - logits['current'].float()
                entry['correctness'] = {
                    'logits_equal': torch.equal(logits['fused'], logits['current']),
                    'logits_finite': bool(torch.isfinite(logits['fused']).all()),
                    'logits_max_abs': delta.abs().max().item(),
                    'logits_relative_l2': (delta.norm() / logits['current'].float().norm()).item(),
                    'argmax_equal': torch.equal(logits['fused'].argmax(-1), logits['current'].argmax(-1)),
                    'current_argmax': logits['current'].argmax(-1).tolist(),
                    'fused_argmax': logits['fused'].argmax(-1).tolist()}
                if 'unpacked' in logits:
                    entry['correctness']['packed_unpacked_equal'] = torch.equal(
                        logits['fused'], logits['unpacked'])
            result['measurements'][f'{model_name}-{length}'] = entry
            save()
            print(json.dumps({'model': model_name, 'length': length,
                              'summary': entry['summary'],
                              'correctness': entry.get('correctness')}), flush=True)
        del model, adapter
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
