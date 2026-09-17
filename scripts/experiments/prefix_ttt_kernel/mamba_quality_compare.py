"""Compare pinned lmms-eval MME/POPE answers and official result metrics.

This reads completed evaluations; it never regenerates answers or recomputes
benchmark scores. A valid comparison exits successfully even when its reported
exact_answers or score_nonregression verdict is false.
"""

import argparse
import json
import math
from pathlib import Path


METRICS = {'mme': ('mme_perception_score', 'mme_cognition_score'),
           'pope': ('pope_accuracy', 'pope_f1_score')}
IDENTITY_FIELDS = ('doc_hash', 'input', 'target', 'input_media')


def load_task(root, task):
    result_files = []
    for path in sorted(root.rglob('*_results.json')):
        document = json.loads(path.read_text())
        if task in document['results']:
            result_files.append((path, document['results'][task]))
    if len(result_files) != 1:
        raise ValueError(f'{root}: expected one unambiguous result file for {task}, found {len(result_files)}')
    sample_files = sorted(root.rglob(f'*_samples_{task}.jsonl'))
    if not sample_files:
        raise ValueError(f'{root}: no sample file for {task}')
    samples = {}
    for path in sample_files:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                sample = json.loads(line)
                doc_id = sample['doc_id']
                if isinstance(doc_id, bool) or not isinstance(doc_id, (str, int)):
                    raise ValueError(f'{path}:{line_number}: invalid doc_id')
                key = str(doc_id)
                if key in samples:
                    raise ValueError(f'{path}:{line_number}: duplicate {task} doc_id {doc_id}')
                if 'filtered_resps' not in sample:
                    raise ValueError(f'{path}:{line_number}: missing filtered_resps')
                samples[key] = sample
    if not samples:
        raise ValueError(f'{root}: empty samples for {task}')
    result_path, metrics = result_files[0]
    return samples, metrics, {'results': str(result_path.resolve()),
                              'samples': [str(path.resolve()) for path in sample_files]}


def metric_value(metrics, name):
    # Select the actual metric key, preserving its lmms-eval filter suffix.
    keys = [key for key in metrics if key.partition(',')[0] == name]
    if len(keys) != 1:
        raise ValueError(f'Expected one official metric {name}, found {keys}')
    value = metrics[keys[0]]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Official metric {keys[0]} is not a finite number')
    return keys[0], value


def compare_task(baseline, candidate, task):
    before, before_metrics, before_files = load_task(baseline, task)
    after, after_metrics, after_files = load_task(candidate, task)
    missing = sorted(before.keys() - after.keys())
    added = sorted(after.keys() - before.keys())
    changed, identity_changes = [], []
    hash_compared = 0
    for key in sorted(before.keys() & after.keys()):
        left, right = before[key], after[key]
        if left['filtered_resps'] != right['filtered_resps']:
            changed.append({'doc_id': left['doc_id'],
                            'baseline': left['filtered_resps'], 'candidate': right['filtered_resps']})
        if 'doc_hash' in left and 'doc_hash' in right:
            hash_compared += 1
        differences = {}
        for field in IDENTITY_FIELDS:
            if ((field in left) != (field in right)
                    or (field in left and left[field] != right[field])):
                differences[field] = {'baseline_present': field in left, 'candidate_present': field in right,
                                      'baseline': left.get(field), 'candidate': right.get(field)}
        if differences:
            identity_changes.append({'doc_id': left['doc_id'], 'fields': differences})
    scores = []
    for name in METRICS[task]:
        before_key, old = metric_value(before_metrics, name)
        after_key, new = metric_value(after_metrics, name)
        if before_key != after_key:
            raise ValueError(f'Cannot compare different metric filters: {before_key} vs {after_key}')
        scores.append({'metric': before_key, 'baseline': old, 'candidate': new,
                       'delta': new - old, 'higher_is_better': True, 'nonregression': new >= old})
    same_samples = not (missing or added or identity_changes)
    return {
        'baseline_files': before_files, 'candidate_files': after_files,
        'baseline_count': len(before), 'candidate_count': len(after),
        'shared_count': len(before.keys() & after.keys()),
        'missing_count': len(missing), 'added_count': len(added), 'changed_count': len(changed),
        'missing': [{'doc_id': before[key]['doc_id'], 'baseline': before[key]['filtered_resps']}
                    for key in missing],
        'added': [{'doc_id': after[key]['doc_id'], 'candidate': after[key]['filtered_resps']}
                  for key in added],
        'changed': changed, 'doc_hash_compared_count': hash_compared,
        'identity_changed_count': len(identity_changes), 'identity_changes': identity_changes,
        'sample_identity_matches': same_samples,
        'exact_answers': same_samples and not changed,
        'official_results': {'baseline': before_metrics, 'candidate': after_metrics},
        'score_comparisons': scores,
        'score_nonregression': all(score['nonregression'] for score in scores),
    }


def compare(baseline, candidate):
    tasks = {task: compare_task(baseline, candidate, task) for task in METRICS}
    return {
        'baseline': str(baseline.resolve()), 'candidate': str(candidate.resolve()),
        'response_field': 'filtered_resps',
        'exact_answers_policy': 'Same task/doc_id coverage, matching available identity fields, identical unnormalized responses',
        'score_policy': 'Official MME Perception/Cognition and POPE accuracy/F1; candidate >= baseline, no tolerance',
        'tasks': tasks,
        'exact_answers': all(task['exact_answers'] for task in tasks.values()),
        'score_nonregression': all(task['score_nonregression'] for task in tasks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Comparison output exists; choose a new --output')
    result = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('exact_answers', 'score_nonregression')}, indent=2))


if __name__ == '__main__':
    main()
