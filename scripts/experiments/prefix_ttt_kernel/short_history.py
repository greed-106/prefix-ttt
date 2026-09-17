"""Historical whole-model protocol: packed E2 versus current E2, optionally E0.

Reuses measure.py request/summarize, including CPU copies of all 129 sampled
logit vectors. Forward-loop wall time includes logit collection, not a full
greedy request. Run GPU work through the project scheduler.
"""

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch

from measure import request, summarize
from mamba_bench import BASELINE_COMMIT, committed_class, validate_baseline
from prefix_ttt.lmms_model import PrefixTTTLlava
from prefix_ttt.model.generation import TransformersHybridCache
from prefix_ttt.model.hybrid import PrefixTTTAttention
from prefix_ttt.ops.features import FeatureReadout


TEXT = 'A conversation about the objects, people and colors in a detailed image. ' * 300


def frozen_methods():
    attention = committed_class('model/hybrid.py', 'PrefixTTTAttention', BASELINE_COMMIT)
    features = committed_class('ops/features.py', 'FeatureReadout', BASELINE_COMMIT)
    cache = committed_class('model/generation.py', 'TransformersHybridCache', BASELINE_COMMIT)
    # Keep the live class, including its current __init__. Old begin/finish do
    # not close over __class__; replacing __init__ would break zero-arg super.
    attention.forward.__globals__['TransformersHybridCache'] = TransformersHybridCache
    return attention.forward, features.features_pair, cache.begin, cache.finish


@contextmanager
def implementation(name, frozen):
    saved = (PrefixTTTAttention.forward, FeatureReadout.features_pair,
             TransformersHybridCache.begin, TransformersHybridCache.finish)
    if name == 'old_packed':
        (PrefixTTTAttention.forward, FeatureReadout.features_pair,
         TransformersHybridCache.begin, TransformersHybridCache.finish) = frozen
    try:
        yield
    finally:
        (PrefixTTTAttention.forward, FeatureReadout.features_pair,
         TransformersHybridCache.begin, TransformersHybridCache.finish) = saved


def file_record(path, checksum=False):
    path = Path(path).resolve()
    stat = path.stat()
    record = {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if checksum:
        record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


def asset_identity(args):
    pretrained = Path(args.pretrained).resolve()
    checkpoint = file_record(args.checkpoint)
    index = pretrained / 'model.safetensors.index.json'
    identity = {'checkpoint': checkpoint,
                'pretrained': str(pretrained),
                'config': file_record(pretrained / 'config.json', checksum=True),
                'index': file_record(index, checksum=True),
                'shards': [file_record(pretrained / name) for name in sorted(set(
                    json.loads(index.read_text())['weight_map'].values()))]}
    if args.checkpoint_manifest:
        manifest = Path(args.checkpoint_manifest)
        previous = json.loads(manifest.read_text())
        old = previous['checkpoint']
        if (str(Path(old['path']).resolve()) != checkpoint['path']
                or any(old[key] != checkpoint[key] for key in ('bytes', 'mtime_ns'))
                or str(Path(previous['pretrained']).resolve()) != str(pretrained)
                or previous['pretrained_index_sha256'] != identity['index']['sha256']):
            raise ValueError('Asset identity differs from the supplied checkpoint manifest')
        identity['existing_manifest'] = file_record(manifest, checksum=True)
        identity['checkpoint_metadata'] = previous['checkpoint_metadata']
        identity['verification'] = 'manifest path/size/mtime and index hash match; large assets not rehashed'
    else:
        identity['verification'] = 'path/size/mtime only for large assets; no prior manifest supplied'
    return identity


def compare_logits(reference, candidate):
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    old_argmax, new_argmax = reference.argmax(-1), candidate.argmax(-1)
    result = {'shape': list(reference.shape), 'reference_finite': bool(torch.isfinite(reference).all()),
              'candidate_finite': bool(torch.isfinite(candidate).all()), 'all_finite': finite,
              'logits_equal': torch.equal(reference, candidate),
              'argmax_equal': torch.equal(old_argmax, new_argmax),
              'argmax_changes': int((old_argmax != new_argmax).sum()),
              'reference_argmax': old_argmax.tolist(), 'candidate_argmax': new_argmax.tolist()}
    delta = candidate.float() - reference.float()
    result.update({'max_abs': delta.abs().max().item() if finite else None,
                   'relative_l2': (delta.norm() / reference.float().norm().clamp_min(1e-30)).item()
                                  if finite else None,
                   'per_position_max_abs': delta.abs().flatten(1).amax(1).tolist() if finite else None})
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-manifest', help='existing mamba_weights.py provenance JSON')
    parser.add_argument('--lengths', default='640,1536')
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--with-e0', action='store_true')
    args = parser.parse_args()
    args.lengths = [int(value) for value in args.lengths.split(',')]
    if min(args.steps, args.rounds, *args.lengths) < 1 or args.warmups < 0:
        parser.error('steps, rounds and lengths must be positive; warmups may be zero')
    if args.steps > min(args.lengths):
        parser.error('teacher-forced steps cannot exceed the shortest input')
    return args


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite evidence: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_paths = [Path(__file__), Path(__file__).with_name('measure.py'),
                    Path(__file__).with_name('mamba_bench.py')]
    source_paths += list(Path('src/prefix_ttt').rglob('*.py'))
    result = {'config': {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
              'baseline_commit': BASELINE_COMMIT,
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
              'protocol': {'batch': 1, 'warmups': args.warmups, 'rounds': args.rounds,
                           'teacher_forced_steps': args.steps, 'text': TEXT,
                           'weights': 'one loaded E2; shared prepared AB copies for both implementations',
                           'prefill': 'fresh request cache, CUDA Events',
                           'tpot': 'median of all individual decode CUDA Event samples',
                           'wall': 'forward-loop wall includes sampled-logit collection; not greedy end-to-end',
                           'logits': 'all vocabulary entries at prompt last position and every decode step',
                           'e0': 'optional contemporary measurement, loaded separately in the same process'},
              'measurements': {}}

    def save():
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)

    current = 'initialization'
    save()
    try:
        validate_baseline(BASELINE_COMMIT)
        frozen = frozen_methods()
        result['assets'] = asset_identity(args)
        result['environment'] = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                                 'cuda': torch.version.cuda, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 'allow_tf32': torch.backends.cuda.matmul.allow_tf32}
        save()
        for model_name in (['e0', 'e2'] if args.with_e0 else ['e2']):
            current = f'{model_name}-load'
            adapter = PrefixTTTLlava(args.pretrained, checkpoint=args.checkpoint if model_name == 'e2' else None)
            model = adapter.model
            maps = [module for module in model.modules() if isinstance(module, FeatureReadout)]
            if model_name == 'e2' and (not maps or any(not hasattr(module, 'ab_phi_inference') for module in maps)):
                raise ValueError('Expected E2 with prepared AB buffers before either timed implementation')
            result.setdefault('models', {})[model_name] = {
                'ttt_layers': len(maps),
                'shared_extra_ab_buffer_bytes': sum(module.ab_phi_inference.numel()
                                                   * module.ab_phi_inference.element_size() for module in maps)}
            tokens = adapter.tokenizer(TEXT, return_tensors='pt', truncation=True,
                                       max_length=max(args.lengths)).input_ids.cuda()
            if tokens.shape != (1, max(args.lengths)):
                raise ValueError('Historical text did not provide the requested B1 input length')
            for length in args.lengths:
                current = f'{model_name}-{length}'
                inputs = tokens[:, :length].contiguous()
                names = ['e0'] if model_name == 'e0' else ['old_packed', 'production']
                rows = {name: [] for name in names}
                entry = {'samples': rows, 'round_comparisons': [], 'repeat_drift': {name: [] for name in names},
                         'input_ids': inputs.cpu().tolist()}
                result['measurements'][current] = entry
                first = {}
                for name in names:
                    with implementation(name, frozen):
                        for _ in range(args.warmups):
                            request(model, inputs, args.steps)
                for round_index in range(args.rounds):
                    logits = {}
                    for name in names if round_index % 2 == 0 else names[::-1]:
                        with implementation(name, frozen):
                            row, value = request(model, inputs, args.steps)
                        row['round'] = round_index
                        rows[name].append(row)
                        logits[name] = value
                        first.setdefault(name, value)
                        drift = compare_logits(first[name], value)
                        drift['round'] = round_index
                        entry['repeat_drift'][name].append(drift)
                    if model_name == 'e2':
                        comparison = compare_logits(logits['old_packed'], logits['production'])
                        comparison['round'] = round_index
                        entry['round_comparisons'].append(comparison)
                    entry['summary'] = {name: summarize(values) for name, values in rows.items()}
                    save()
                    if not all(entry['repeat_drift'][name][-1]['all_finite'] for name in names):
                        raise FloatingPointError('Nonfinite sampled logits; saved completed samples and comparisons')
                    print(json.dumps({'measurement': current, 'round': round_index,
                                      'summary': entry['summary'],
                                      'paired_logits_equal': None if model_name == 'e0'
                                      else entry['round_comparisons'][-1]['logits_equal']}), flush=True)
                del inputs, first, logits, value
            del model, adapter, tokens, maps
            gc.collect()
            torch.cuda.empty_cache()
        result['status'] = 'succeeded'
        save()
    except Exception as exc:
        result['status'] = 'failed'
        result['failure'] = {'stage': current, 'type': type(exc).__name__, 'message': str(exc)}
        save()
        raise


if __name__ == '__main__':
    main()
