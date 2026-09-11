"""Build fixed index manifests only after complete, successful offline audits."""

import argparse
from array import array
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath

import ijson

from prefix_ttt.digests import digest_file, digest_json


SCHEMA_VERSION = 1
A_STAGE_SAMPLES = 50_000   # phase-A quota inside the fixed manifest
PREPROCESS_FILES = (
    'src/prefix_ttt/audit_data.py', 'src/prefix_ttt/model/labels.py',
    'src/prefix_ttt/model/supervision.py', 'src/prefix_ttt/model/bridge.py',
    'third_party/llava/llava/train/train.py', 'third_party/llava/llava/conversation.py',
    'third_party/llava/llava/mm_utils.py',
)
EXCLUSIONS = {'data_invalid_conversations', 'data_invalid_turn', 'data_no_assistant_content',
              'unsupported_image_schema', 'protocol_image_truncated', 'protocol_answer_truncated'}


def stable_key(seed, namespace, value):
    return digest_json([seed, namespace, value])


def largest_remainder(counts, total):
    size = sum(counts.values())
    if total < 0 or total > size:
        raise ValueError('Requested A sample count exceeds available training records')
    if not size:
        return {key: 0 for key in counts}
    quotas = {key: total * count // size for key, count in counts.items()}
    ranking = sorted(counts, key=lambda key: (-(total * counts[key] % size), key))
    for key in ranking[:total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def length_bucket(length):
    for bound in (512, 1024, 1536, 2048):
        if length <= bound:
            return f'le{bound}'
    return 'gt2048'


def build(annotation, label_rows, label_summary, image_summary, *, dev_size=2048,
          a_size=A_STAGE_SAMPLES, seed=42, preprocess_files=None, execution_source_files=None):
    if type(dev_size) is not int or dev_size < 1 or type(a_size) is not int or a_size < 1:
        raise ValueError('Positive integer dev and A sizes required')
    annotation, label_rows = Path(annotation), Path(label_rows)
    label_summary, image_summary = Path(label_summary), Path(image_summary)
    summary = json.loads(label_summary.read_text())
    images = json.loads(image_summary.read_text())
    count = summary['counts']['records']
    labels = summary['labels']
    if (not isinstance(labels, dict) or labels.get('complete') is not True
            or labels.get('checked') != count or labels.get('preprocessing_verified') is not True
            or labels.get('statuses', {}).get('preprocessing_error', 0)):
        raise ValueError('Complete full label audit without preprocessing errors required')
    decode = images['decode']
    if (images['counts']['records'] != count or images.get('missing_images') != []
            or summary.get('missing_images') != [] or not isinstance(decode, dict)
            or decode.get('checked') != images['unique_images'] or decode.get('failures') != []
            or images['unique_images'] != summary['unique_images']):
        raise ValueError('Full matching image decode audit with zero missing/failures required')
    project = Path(__file__).resolve().parents[2]
    files = [Path(p) for p in preprocess_files] if preprocess_files is not None else [project / p for p in PREPROCESS_FILES]
    preprocess_hashes = {str(path.resolve()): digest_file(path) for path in files}
    if not files:
        raise ValueError('Preprocessing source files must be recorded')
    snapshot = project / 'artifacts/cpu/audit_data.executed.py'
    execution_files = ([Path(p) for p in execution_source_files] if execution_source_files is not None
                       else ([snapshot] if snapshot.is_file() else []))
    execution_hashes = {str(path.resolve()): digest_file(path) for path in execution_files}

    # Union all original rows, including excluded bridge rows, so a transitive
    # image/conversation/content group can never be split across train and dev.
    parents = array('i', range(count))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    owners = {}
    valid, excluded, strata = [], [], {}
    statuses = Counter()
    unique_images = set()
    with annotation.open('rb') as raw_handle, label_rows.open() as audit_handle:
        for index, row in enumerate(ijson.items(raw_handle, 'item')):
            if index >= count:
                raise ValueError('Annotation record count exceeds audited count')
            line = audit_handle.readline()
            if not line:
                raise ValueError('Label audit is truncated')
            audited = json.loads(line)
            if audited.get('index') != index or audited.get('id') != str(row.get('id', index)):
                raise ValueError('Audit indices/identifiers do not match original annotation')
            status = audited['status']
            if status != 'valid' and status not in EXCLUSIONS:
                raise ValueError(f'Unresolved preprocessing or unknown status: {status}')
            statuses[status] += 1
            image = row.get('image')
            source = image.split('/')[0] if isinstance(image, str) and image else 'text_only'
            raw = row.get('conversations')
            content = hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            keys = []
            if isinstance(image, str) and image:
                normalized = PurePosixPath(image)
                if normalized.is_absolute() or '..' in normalized.parts:
                    raise ValueError('Image paths must be relative to the fixed data root')
                keys.append(('image', str(normalized)))
                unique_images.add(image)
            identifier = row.get('id')
            if identifier is not None and str(identifier).strip():
                keys.append(('id', source, str(identifier)))
            if isinstance(raw, list) and raw:
                keys.append(('content', content))
            for key in keys:
                if key in owners:
                    left, right = find(index), find(owners[key])
                    parents[max(left, right)] = min(left, right)
                else:
                    owners[key] = index
            if status == 'valid':
                if audited.get('image') != image or audited.get('content_hash') != content:
                    raise ValueError('Audited image/content differs from original annotation')
                if audited.get('targets_after', 0) <= 0 or audited.get('targets_before', 0) <= 0:
                    raise ValueError('Valid label record has no shifted targets')
                length = audited.get('expanded_length')
                if type(length) is not int or length < 2:
                    raise ValueError('Valid expanded length is missing')
                modality = 'image' if image else 'text'
                strata[index] = f'{source}|{modality}|{length_bucket(length)}'
                valid.append(index)
            else:
                excluded.append({'index': index, 'reason': status,
                                 'category': 'protocol' if status.startswith('protocol_') or status == 'unsupported_image_schema' else 'data'})
        if sum(statuses.values()) != count or audit_handle.read().strip():
            raise ValueError('Annotation/JSONL total count mismatch')
    if dict(statuses) != {k: v for k, v in labels['statuses'].items() if v}:
        raise ValueError('Label status counts differ from audit summary')
    if len(unique_images) != images['unique_images']:
        raise ValueError('Original images differ from decode audit inventory')
    del owners
    groups = defaultdict(list)
    for index in valid:
        groups[find(index)].append(index)
    dev = []
    for root in sorted(groups, key=lambda root: stable_key(seed, 'dev-group', root)):
        if len(dev) >= dev_size:
            break
        dev.extend(groups[root])
    if len(dev) < dev_size:
        raise ValueError('Insufficient valid records for dev')
    dev_set = set(dev)
    train = sorted((i for i in valid if i not in dev_set), key=lambda i: stable_key(seed, 'B-order', i))
    dev.sort(key=lambda i: stable_key(seed, 'dev-order', i))
    bins = defaultdict(list)
    for index in train:
        bins[strata[index]].append(index)
    quotas = largest_remainder({key: len(indices) for key, indices in bins.items()}, a_size)
    stage_a = []
    for key, indices in bins.items():
        stage_a.extend(sorted(indices, key=lambda i: stable_key(seed, 'A-select', i))[:quotas[key]])
    stage_a.sort(key=lambda i: stable_key(seed, 'A-order', i))
    return {
        'schema_version': SCHEMA_VERSION, 'seed': seed,
        'annotation': str(annotation.resolve()),
        'inputs': {str(p.resolve()): digest_file(p) for p in (annotation, label_rows, label_summary, image_summary)},
        'preprocessing_source_sha256': preprocess_hashes,
        'preprocessing_source_scope': 'current files at manifest creation; not a claim these exact bytes executed the audit',
        'audit_execution_source_sha256': execution_hashes,
        'audit_execution_source_scope': 'execution-time snapshots only; empty means unavailable, not inferred from current files',
        'grouping': 'transitive image path / (source, nonempty original id) / conversation content sha256; excluded rows bridge groups',
        'dev_requested': dev_size, 'dev_actual': len(dev), 'A_requested': a_size,
        'counts': {'original': count, 'valid': len(valid), 'train': len(train), 'dev': len(dev), 'A': len(stage_a),
                   'excluded': len(excluded), 'valid_groups': len(groups)},
        'strata_counts': dict(sorted(Counter(strata[i] for i in train).items())),
        'A_strata_quotas': dict(sorted(quotas.items())),
        'order_sha256': {name: digest_json(indices) for name, indices in [('train', train), ('dev', dev), ('A', stage_a)]},
        'train': train, 'dev': dev, 'A': stage_a, 'exclusions': excluded,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation', default='data/llava-v1.5-assets-v1/datasets/llava-665k/llava_v1_5_mix665k.json')
    parser.add_argument('--label-rows', default='artifacts/cpu/labels-full/label-audit.jsonl')
    parser.add_argument('--label-summary', default='artifacts/cpu/labels-full/data-audit.json')
    parser.add_argument('--image-summary', default='artifacts/cpu/data/data-audit.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite a fixed manifest')
    result = build(args.annotation, args.label_rows, args.label_summary, args.image_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
