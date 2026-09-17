#!/usr/bin/env python3
"""Compare an optimization run against a recorded baseline run.

Both roots keep the same layout: ``<root>/<label>/<task>/..._results.json`` with the
matching ``*_samples_<task>.jsonl``, plus ``<root>/cost/<label>-<task>.jsonl`` written
by ``prefix_ttt.instrument``. The comparison reports the official score, how many
requests changed their greedy response, and the per-request cost medians, so one
kernel optimization can be accepted or rejected with a single command.
"""
import argparse
import json
import statistics
from pathlib import Path

BUCKETS = ('full_kv_bytes', 'ttt_state_bytes', 'local_window_bytes')
ROWS = (('requests', 'count', 1),
        ('prefill_ms', 'prefill (ms)', 1),
        ('tpot_ms', 'TPOT (ms/token)', 1),
        ('peak_allocated_bytes', 'peak allocated (GiB)', 2 ** 30),
        ('cache_bytes', 'cache (MiB)', 2 ** 20),
        ('cache_full_kv_bytes', 'cache full KV (MiB)', 2 ** 20))


def scores(root, label, task):
    files = sorted(Path(root, label, task).rglob('*_results.json'))
    if not files:
        raise SystemExit(f'no results under {root}/{label}/{task}')
    block = json.loads(files[-1].read_text())['results'][task]
    return {key.split(',')[0]: value for key, value in block.items()
            if key.endswith(',none') and 'stderr' not in key
            and isinstance(value, (int, float))}


def responses(root, label, task):
    files = sorted(Path(root, label, task).rglob(f'*_samples_{task}.jsonl'))
    if not files:
        raise SystemExit(f'no sample log under {root}/{label}/{task}')
    rows = {}
    for line in files[-1].read_text().splitlines():
        record = json.loads(line)
        rows[str(record['doc_id'])] = record['filtered_resps']
    return rows


def cost(root, label, task):
    path = Path(root, 'cost', f'{label}-{task}.jsonl')
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {
        'requests': len(records),
        'prefill_ms': statistics.median([r['prefill_ms'] for r in records]),
        'tpot_ms': statistics.median([r['tpot_ms'] for r in records]),
        'peak_allocated_bytes': statistics.median([r['peak_allocated_bytes'] for r in records]),
        'cache_bytes': statistics.median([sum(r['cache'][n] for n in BUCKETS) for r in records]),
        'cache_full_kv_bytes': statistics.median([r['cache']['full_kv_bytes'] for r in records]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--label', default='e2')
    parser.add_argument('--task', default='mme')
    args = parser.parse_args()

    base_scores = scores(args.baseline, args.label, args.task)
    new_scores = scores(args.candidate, args.label, args.task)
    print(f'== official {args.label}/{args.task} score')
    for key in sorted(set(base_scores) | set(new_scores)):
        before, after = base_scores.get(key), new_scores.get(key)
        delta = '' if before is None or after is None else f'   ({after - before:+.4f})'
        print(f'   {key:28s} {before}  ->  {after}{delta}')

    base_responses = responses(args.baseline, args.label, args.task)
    new_responses = responses(args.candidate, args.label, args.task)
    changed = [doc for doc in base_responses
               if doc in new_responses and base_responses[doc] != new_responses[doc]]
    missing = [doc for doc in base_responses if doc not in new_responses]
    print(f'== responses: {len(new_responses)} recorded, {len(changed)} changed, '
          f'{len(missing)} missing')
    for doc in changed[:5]:
        print(f'   doc {doc}: {base_responses[doc]!r} -> {new_responses[doc]!r}')

    base_cost = cost(args.baseline, args.label, args.task)
    new_cost = cost(args.candidate, args.label, args.task)
    print('== per-request cost (median)')
    for key, title, scale in ROWS:
        before, after = base_cost[key], new_cost[key]
        speedup = f'   x{before / after:.2f}' if key in ('prefill_ms', 'tpot_ms') else ''
        print(f'   {title:22s} {before / scale:10.3f} -> {after / scale:10.3f}'
              f'   ({after - before:+.3f}){speedup}')


if __name__ == '__main__':
    main()
