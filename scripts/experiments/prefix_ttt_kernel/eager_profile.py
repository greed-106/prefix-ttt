"""Profile real E2 text prefill without CUDA Graph; submit through the GPU queue."""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from short_history import TEXT, asset_identity
from prefix_ttt.lmms_model import PrefixTTTLlava
from prefix_ttt.model import hybrid
from prefix_ttt.model.generation import TransformersHybridCache
from prefix_ttt.ops.features import FeatureReadout


@contextmanager
def ranges(model):
    """Add nested labels only; restore instance and class bindings exactly."""
    saved = []

    def wrap(owner, name, label):
        original = getattr(owner, name)
        saved.append((owner, name, original, name in vars(owner)))

        def measured(*args, **kwargs):
            with torch.profiler.record_function('eager_profile::' + label):
                return original(*args, **kwargs)
        setattr(owner, name, measured)

    try:
        for index, layer in enumerate(model.get_model().layers):
            kind = 'ttt' if isinstance(layer.self_attn, hybrid.PrefixTTTAttention) else 'full'
            wrap(layer, 'forward', f'decoder:{index}:{kind}')
        for name in ('begin', 'finish'):
            wrap(TransformersHybridCache, name, 'cache.' + name)
        wrap(FeatureReadout, 'features_prefill', 'stage:features')
        for name, label in [('dense_local', 'local'), ('fla_prefix', 'prefix'),
                            ('fused_readout', 'readout')]:
            wrap(hybrid, name, 'stage:' + label)
        yield
    finally:
        for owner, name, original, had_local in reversed(saved):
            if had_local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def trace_summary(profiler, path):
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text())['traceEvents']
    kernels, counts = Counter(), Counter()
    for event in events:
        if event.get('cat') == 'kernel' and event.get('ph') == 'X':
            kernels[event['name']] += event['dur']
            counts[event['name']] += 1
    labels = [{key: event.get(key) for key in ('name', 'ts', 'dur', 'pid', 'tid')}
              for event in events if event.get('cat') == 'user_annotation'
              and event.get('ph') == 'X' and event['name'].startswith('eager_profile::')]
    cpu = sorted(profiler.key_averages(), key=lambda item: item.self_cpu_time_total, reverse=True)
    return {'trace': str(path), 'kernel_count': sum(counts.values()),
            'kernel_device_us': dict(kernels), 'kernel_counts': dict(counts),
            'kernel_device_total_us': sum(kernels.values()), 'nested_cpu_ranges': labels,
            'cpu_top': [{'name': event.key, 'count': event.count, 'self_cpu_us': event.self_cpu_time_total}
                        for event in cpu if event.key != 'Activity Buffer Request'][:40],
            'collector_activity_buffer_us': sum(event.self_cpu_time_total for event in cpu
                                               if event.key == 'Activity Buffer Request')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-manifest')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', default='640,1536')
    args = parser.parse_args()
    args.lengths = [int(value) for value in args.lengths.split(',')]
    if min(args.lengths) < 1 or len(set(args.lengths)) != len(args.lengths):
        parser.error('lengths must be distinct positive integers')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x'):
        pass
    trace_root = Path('artifacts/eager-profile') / args.output.parent.name
    paths = list(Path('src/prefix_ttt').rglob('*.py')) + [Path(__file__)]
    paths += [Path(__file__).with_name(name) for name in ('short_history.py', 'measure.py', 'mamba_bench.py')]
    result = {'status': 'running', 'config': {key: str(value) if isinstance(value, Path) else value
                                             for key, value in vars(args).items()},
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'protocol': {'batch': 1, 'external_warmup_calls': 2, 'schedule_warmup_calls': 2,
                           'active_calls': 3, 'schedule_repeats': 1, 'cuda_graph': False,
                           'record_shapes': False, 'profile_memory': False, 'with_stack': False,
                           'input_source': 'same repeated text as measure.py and short_history.py',
                           'text': TEXT, 'text_sha256': hashlib.sha256(TEXT.encode()).hexdigest(),
                           'scope': 'production full language-model prefill, fresh cache; no vision or decode',
                           'limitations': 'Profiling and nested labels perturb host timing. CPU ranges are '
                           'inclusive and must not be added together. Kernel sums exclude gaps and transfers. '
                           'End-of-step synchronization is measurement plumbing, not model work; '
                           'these profiled durations are not a new latency benchmark.'}, 'measurements': {}}

    def save():
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)

    current = 'initialization'
    save()
    try:
        result['assets'] = asset_identity(args)
        result['environment'] = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                                 'cuda': torch.version.cuda, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 'allow_tf32': torch.backends.cuda.matmul.allow_tf32}
        adapter = PrefixTTTLlava(args.pretrained, checkpoint=args.checkpoint)
        model = adapter.model
        if len(model.config.prefix_ttt_layers) != 23:
            raise ValueError('Expected the production E2 layout with 23 TTT layers')
        tokens = adapter.tokenizer(TEXT, return_tensors='pt', truncation=True,
                                   max_length=max(args.lengths)).input_ids.cuda()
        if tokens.shape != (1, max(args.lengths)):
            raise ValueError('Text does not provide the requested input length')
        trace_root.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for length in args.lengths:
                current = f'b1-t{length}'
                path = trace_root / f'{args.output.stem}-{current}.trace.json'
                if path.exists():
                    raise FileExistsError(f'Refusing to overwrite trace: {path}')
                inputs = tokens[:, :length].contiguous()
                ids = inputs.cpu().tolist()
                entry = {'input_shape': list(inputs.shape), 'input_ids': ids,
                         'input_ids_json_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
                result['measurements'][current] = entry
                for _ in range(2):
                    model(input_ids=inputs, use_cache=True)
                torch.cuda.synchronize()

                def ready(profiler):
                    entry.update(trace_summary(profiler, path))
                    save()

                with ranges(model), torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=0, warmup=2, active=3, repeat=1),
                        record_shapes=False, profile_memory=False, with_stack=False,
                        on_trace_ready=ready) as profiler:
                    for step in range(5):
                        with torch.profiler.record_function(f'eager_profile::prefill:{step}'):
                            output = model(input_ids=inputs, use_cache=True)
                        torch.cuda.synchronize()
                        profiler.step()
                        if step != 4:
                            del output
                entry['logits_finite'] = bool(torch.isfinite(output.logits).all())
                del output
                entry['recorded_prefill_calls'] = sum(row['name'].startswith('eager_profile::prefill:')
                                                      for row in entry['nested_cpu_ranges'])
                if not entry['logits_finite'] or entry['kernel_count'] == 0:
                    raise RuntimeError('Profile has no kernels or produced nonfinite logits')
                if entry['recorded_prefill_calls'] != 3:
                    raise RuntimeError('Profiler did not capture exactly three active prefill calls')
                save()
                print(json.dumps({'shape': current, 'kernel_count': entry['kernel_count'],
                                  'active_calls': 3, 'trace': str(path)}), flush=True)
        result['status'] = 'succeeded'
        save()
    except Exception as exc:
        result['status'] = 'failed'
        result['failure'] = {'stage': current, 'type': type(exc).__name__, 'message': str(exc)}
        save()
        raise


if __name__ == '__main__':
    main()
