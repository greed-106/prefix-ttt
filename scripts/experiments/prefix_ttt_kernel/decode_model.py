"""Paired real-E2 continuous eager decode: production versus Local preparation.

Run through the GPU queue. Prefill is outside timing and outside the patch;
one CUDA event pair measures all 128 steps, with no manual per-step sync/copy.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

import decode_local
from mamba_bench import distribution
from short_history import TEXT, asset_identity, compare_logits
from prefix_ttt.lmms_model import PrefixTTTLlava


STEPS, WARMUPS, ROUNDS = 128, 2, 5


def request(model, inputs, name):
    output = model(input_ids=inputs, use_cache=True)
    cache = output.past_key_values
    # Release the large prefill-logits allocation after the first decode step.
    logits = [output.logits[:, -1].clone()]
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    with decode_local.patch() if name == 'local_prepare' else nullcontext():
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        for step in range(STEPS):
            output = model(input_ids=inputs[:, step:step + 1],
                           past_key_values=output.past_key_values, use_cache=True)
            logits.append(output.logits[:, -1])
        end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000
    if output.past_key_values is not cache:
        raise RuntimeError('Continuous decode unexpectedly replaced its request cache')
    counts = {'physical_length': cache.get_seq_length(),
              'seen_tokens': cache.storage.seen_tokens.cpu().tolist(),
              'valid_tokens': cache.valid_history.sum(1).cpu().tolist(),
              'finished': cache.storage.finished.cpu().tolist(),
              'layers': len(cache.storage.layers),
              'state_dtypes': sorted({str(layer.state.dtype) for layer in cache.storage.layers.values()
                                      if layer.state is not None})}
    expected_length = inputs.shape[1] + STEPS
    if (counts['physical_length'] != expected_length or counts['seen_tokens'] != [expected_length]
            or counts['valid_tokens'] != [expected_length] or counts['finished'] != [False]
            or counts['state_dtypes'] != ['torch.float32']):
        raise RuntimeError(f'Continuous decode cache invariants failed: {counts}')
    cuda_ms = start.elapsed_time(end)
    return {'cuda_ms': cuda_ms, 'wall_ms': wall_ms,
            'sequence_average_cuda_ms_per_step': cuda_ms / STEPS,
            'sequence_average_wall_ms_per_step': wall_ms / STEPS,
            'cache': counts}, torch.stack(logits).cpu()


def summarize(samples):
    return {field: distribution([row[field] for row in samples]) for field in
            ('cuda_ms', 'wall_ms', 'sequence_average_cuda_ms_per_step',
             'sequence_average_wall_ms_per_step')}


@torch.inference_mode()
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
        parser.error(f'lengths must be distinct and at least {STEPS}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x'):
        pass
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = sorted(Path('src/prefix_ttt').rglob('*.py')) + [Path(__file__)]
    paths += [Path(__file__).with_name(name) for name in
              ('decode_local.py', 'short_history.py', 'measure.py', 'mamba_bench.py')]
    result = {'status': 'running', 'config': {key: str(value) if isinstance(value, Path) else value
                                             for key, value in vars(args).items()},
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
              'protocol': {'batch': 1, 'teacher_forced_steps': STEPS, 'warmups_per_arm': WARMUPS,
                           'paired_rounds': ROUNDS, 'cuda_graph': False,
                           'scope': 'one shared real E2; ordinary eager language-model decode only; no vision',
                           'input': TEXT, 'text_sha256': hashlib.sha256(TEXT.encode()).hexdigest(),
                           'fresh_cache': 'each warmup/sample starts with unpatched production prefill',
                           'patch': 'Local preparation only; context entered before decode timing',
                           'timing': 'one CUDA event pair around 128 continuous steps; wall ends after final '
                           'synchronize; no manual per-step synchronize, event recording, or CPU copy',
                           'distribution': 'five request averages per arm, not individual-token quantiles',
                           'logits': 'all vocabulary entries at prompt last position and all 128 decode '
                           'positions; GPU views retained during decode and copied after the timed interval',
                           'history_comparability': 'decode-only sequence-average protocol; not the older '
                           'prefill-inclusive forward-loop wall or median of individual-step CUDA events',
                           'order': 'production/local_prepare in even rounds, reverse in odd rounds'},
              'measurements': {}}

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
        names = ('production', 'local_prepare')
        for length in args.lengths:
            current = f'b1-t{length}'
            inputs = tokens[:, :length].contiguous()
            ids = inputs.cpu().tolist()
            entry = result['measurements'][current] = {
                'input_ids': ids, 'input_ids_json_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                'samples': {name: [] for name in names}, 'round_comparisons': []}
            for name in names:
                for _ in range(WARMUPS):
                    request(model, inputs, name)
            for round_index in range(ROUNDS):
                values = {}
                for name in names if round_index % 2 == 0 else names[::-1]:
                    row, values[name] = request(model, inputs, name)
                    row['round'] = round_index
                    entry['samples'][name].append(row)
                comparison = compare_logits(values['production'], values['local_prepare'])
                comparison['round'] = round_index
                comparison['cache_counts_equal'] = (entry['samples']['production'][-1]['cache']
                                                     == entry['samples']['local_prepare'][-1]['cache'])
                entry['round_comparisons'].append(comparison)
                entry['summary'] = {name: summarize(rows) for name, rows in entry['samples'].items()}
                reference = entry['summary']['production']['cuda_ms']['median']
                candidate = entry['summary']['local_prepare']['cuda_ms']['median']
                entry['speedup'] = reference / candidate
                entry['all_completed_rounds_faster'] = all(left['cuda_ms'] > right['cuda_ms'] for left, right in
                    zip(entry['samples']['production'], entry['samples']['local_prepare']))
                save()
                if not (comparison['logits_equal'] and comparison['all_finite'] and comparison['cache_counts_equal']):
                    raise AssertionError('Paired decode logits/cache checks failed; raw comparisons saved')
                print(json.dumps({'shape': current, 'round': round_index, 'speedup': entry['speedup'],
                                  'logits_equal': comparison['logits_equal'], 'summary': entry['summary']}), flush=True)
        result['status'] = 'succeeded'
        save()
    except Exception as exc:
        result['status'] = 'failed'
        result['failure'] = {'stage': current, 'type': type(exc).__name__, 'message': str(exc)}
        save()
        raise


if __name__ == '__main__':
    main()
