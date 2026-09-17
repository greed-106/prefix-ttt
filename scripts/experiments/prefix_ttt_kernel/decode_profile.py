"""Attribute three consecutive real E2 eager decode steps; use the GPU queue.

This is a profiler trace, not a latency benchmark. Prefill is outside the trace;
each step consumes its predecessor's cache and a teacher-forced input token.
"""

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from eager_profile import trace_summary
from short_history import TEXT, asset_identity
from prefix_ttt.cache import HybridCache
from prefix_ttt.lmms_model import PrefixTTTLlava
from prefix_ttt.model import hybrid
from prefix_ttt.model.generation import TransformersHybridCache
from prefix_ttt.ops.features import FeatureReadout


STEPS, BEFORE_PROFILE, WARMUP, ACTIVE = 128, 15, 2, 3


@contextmanager
def ranges(model):
    """Label the actual decode call sites and restore every binding on exit."""
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
        wrap(HybridCache, 'set_layer', 'cache.set_layer')
        wrap(FeatureReadout, 'features_pair', 'stage:features')
        for name, label in [('local_attention_decode', 'local'),
                            ('recurrent_decode', 'recurrent'), ('fused_readout', 'readout')]:
            wrap(hybrid, name, 'stage:' + label)
        yield
    finally:
        for owner, name, original, had_local in reversed(saved):
            if had_local:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


def continuation(model, inputs, output, start, stop, logits, lengths, profiler=None):
    """Advance one request, retaining logits for checks outside the trace."""
    for step in range(start, stop):
        cache = output.past_key_values
        label = (torch.profiler.record_function(f'eager_profile::decode:{step + 1}')
                 if profiler is not None else nullcontext())
        with label:
            output = model(input_ids=inputs[:, step:step + 1], past_key_values=cache, use_cache=True)
        if output.past_key_values is not cache:
            raise RuntimeError('Decode unexpectedly replaced the request cache')
        length = cache.get_seq_length()
        if length != inputs.shape[1] + step + 1:
            raise RuntimeError('Decode cache did not advance by exactly one token')
        lengths.append(length)
        logits.append(output.logits[:, -1])
        if profiler is not None:
            profiler.step()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-manifest')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lengths', default='640,1536')
    args = parser.parse_args()
    args.lengths = [int(value) for value in args.lengths.split(',')]
    if min(args.lengths) < STEPS or len(set(args.lengths)) != len(args.lengths):
        parser.error(f'lengths must be distinct and at least {STEPS} for teacher forcing')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x'):
        pass
    trace_root = Path('artifacts/decode-profile') / args.output.parent.name
    paths = list(Path('src/prefix_ttt').rglob('*.py')) + [Path(__file__)]
    paths += [Path(__file__).with_name(name) for name in
              ('eager_profile.py', 'short_history.py', 'measure.py', 'mamba_bench.py')]
    result = {'status': 'running', 'config': {key: str(value) if isinstance(value, Path) else value
                                             for key, value in vars(args).items()},
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'protocol': {'batch': 1, 'external_warmup_requests': 1, 'teacher_forced_steps': STEPS,
                           'decode_steps_before_profiler': BEFORE_PROFILE,
                           'schedule_warmup_steps': WARMUP, 'active_steps': ACTIVE,
                           'active_step_numbers_1based': list(range(BEFORE_PROFILE + WARMUP + 1,
                                                                  BEFORE_PROFILE + WARMUP + ACTIVE + 1)),
                           'cuda_graph': False, 'record_shapes': False,
                           'profile_memory': False, 'with_stack': False,
                           'text': TEXT, 'text_sha256': hashlib.sha256(TEXT.encode()).hexdigest(),
                           'scope': 'real continuous language-model decode; prefill excluded; no vision',
                           'limitations': 'Profiling and nested labels perturb host timing. CPU ranges are '
                           'inclusive and must not be added. Kernel sums exclude gaps and transfers. '
                           'No manual per-step synchronization. These durations are attribution evidence, '
                           'not a latency benchmark.'}, 'measurements': {}}

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
            raise ValueError('Expected production E2 with 23 TTT layers')
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
                output = model(input_ids=inputs, use_cache=True)
                output = continuation(model, inputs, output, 0, STEPS, [], [])
                torch.cuda.synchronize()
                del output

                output = model(input_ids=inputs, use_cache=True)
                logits, lengths = [output.logits[:, -1]], []
                output = continuation(model, inputs, output, 0, BEFORE_PROFILE, logits, lengths)
                torch.cuda.synchronize()

                def ready(profiler):
                    entry.update(trace_summary(profiler, path))
                    entry['trace_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    save()

                with ranges(model), torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                        schedule=torch.profiler.schedule(wait=0, warmup=WARMUP, active=ACTIVE, repeat=1),
                        record_shapes=False, profile_memory=False, with_stack=False,
                        on_trace_ready=ready) as profiler:
                    output = continuation(model, inputs, output, BEFORE_PROFILE,
                                          BEFORE_PROFILE + WARMUP + ACTIVE, logits, lengths, profiler)
                output = continuation(model, inputs, output, BEFORE_PROFILE + WARMUP + ACTIVE,
                                      STEPS, logits, lengths)
                torch.cuda.synchronize()
                entry['logits_finite_by_position'] = torch.isfinite(torch.stack(logits)).flatten(1).all(1).tolist()
                entry['logits_finite'] = all(entry['logits_finite_by_position'])
                entry['cache_lengths_after_decode'] = lengths
                entry['final_seen_tokens'] = output.past_key_values.storage.seen_tokens.cpu().tolist()
                entry['recorded_decode_steps_1based'] = [int(row['name'].rsplit(':', 1)[1])
                    for row in entry['nested_cpu_ranges'] if row['name'].startswith('eager_profile::decode:')]
                if not entry['logits_finite'] or entry['kernel_count'] == 0:
                    raise RuntimeError('Profile has no kernels or produced nonfinite logits')
                if sorted(entry['recorded_decode_steps_1based']) != result['protocol']['active_step_numbers_1based']:
                    raise RuntimeError('Profiler did not capture exactly the intended three decode steps')
                if entry['final_seen_tokens'] != [length + STEPS]:
                    raise RuntimeError('Final effective cache position differs from continuous decode')
                del output, logits
                save()
                print(json.dumps({'shape': current, 'kernel_count': entry['kernel_count'],
                                  'active_steps': entry['recorded_decode_steps_1based'], 'trace': str(path)}), flush=True)
        result['status'] = 'succeeded'
        save()
    except Exception as exc:
        result['status'] = 'failed'
        result['failure'] = {'stage': current, 'type': type(exc).__name__, 'message': str(exc)}
        save()
        raise


if __name__ == '__main__':
    main()
