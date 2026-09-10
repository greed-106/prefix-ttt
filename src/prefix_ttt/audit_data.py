"""Offline structural/image/label audit, reusing LLaVA's actual preprocessor.

Artifacts are audits, not a final train/dev split. Preprocessing errors block
the audit and are NEVER classified as data exclusions.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import copy
import hashlib
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import ijson

_TOKENIZER = None
_PATCH_COUNT = None


def init_labels(model_path):
    global _TOKENIZER, _PATCH_COUNT
    import torch
    from prefix_ttt.model.bridge import load_tokenizer
    from llava import conversation
    torch.set_num_threads(1)
    _TOKENIZER = load_tokenizer(model_path)
    _TOKENIZER.model_max_length = 1_000_000  # audit BEFORE truncation
    vision = json.loads((Path(model_path) / 'config.json').read_text())['vision_config']
    _PATCH_COUNT = (vision['image_size'] // vision['patch_size']) ** 2
    conversation.default_conversation = conversation.conv_templates['v1']


def audit_labels(item):
    index, row = item
    from llava.train.train import preprocess_v1, preprocess_multimodal
    from prefix_ttt.model.labels import classify_supervision
    identifier = str(row.get('id', index))
    result = {'index': index, 'id': identifier}
    raw = row.get('conversations')
    if not isinstance(raw, list) or not raw:
        return {**result, 'status': 'data_invalid_conversations'}
    if any(not isinstance(turn, dict) or not isinstance(turn.get('value'), str)
           or turn.get('from') not in ('human', 'gpt') for turn in raw):
        return {**result, 'status': 'data_invalid_turn'}
    image = row.get('image')
    if image is not None and not isinstance(image, str):
        return {**result, 'status': 'unsupported_image_schema'}
    has_answer = any(turn['from'] == 'gpt' and turn['value'].strip() for turn in raw)
    if not has_answer:
        return {**result, 'status': 'data_no_assistant_content'}
    try:
        conversations = [copy.deepcopy(raw)]
        if image:
            conversations = preprocess_multimodal(conversations, SimpleNamespace(
                is_multimodal=True, mm_use_im_start_end=False))
        tokens = preprocess_v1(conversations, _TOKENIZER, has_image=bool(image))
        ids, labels = tokens['input_ids'][0].tolist(), tokens['labels'][0].tolist()
        sentinels = ids.count(-200)
        if sentinels != int(bool(image)):
            raise ValueError(f'image metadata / sentinel mismatch: {bool(image)} vs {sentinels}')
        expanded = []
        visual_positions = []
        for token, label in zip(ids, labels):
            if token == -200:
                visual_positions.extend(range(len(expanded), len(expanded) + _PATCH_COUNT))
                expanded.extend([-100] * _PATCH_COUNT)
            else:
                expanded.append(label)
        before = sum(label != -100 for label in expanded[1:])
        after = sum(label != -100 for label in expanded[1:2048])
        truncated_image = bool(visual_positions and max(visual_positions) >= 2048)
        status = classify_supervision(has_assistant_content=has_answer,
            targets_before_truncation=before, targets_after_truncation=after,
            image_truncated=truncated_image)
        return {**result, 'status': status, 'text_token_count': len(ids),
                'expanded_length': len(expanded), 'targets_before': before, 'targets_after': after,
                'image': image, 'content_hash': hashlib.sha256(json.dumps(raw,
                ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    except Exception as error:
        return {**result, 'status': 'preprocessing_error', 'error_type': type(error).__name__,
                'error': str(error), 'image': image, 'roles': [t['from'] for t in raw]}


def decode_image(path):
    from PIL import Image
    try:
        with Image.open(path) as image:
            image.convert('RGB').load()
        return None
    except Exception as error:
        return {'path': str(path), 'error_type': type(error).__name__, 'error': str(error)}


def rows(annotation):
    with Path(annotation).open('rb') as handle:
        yield from enumerate(ijson.items(handle, 'item'))


def audit(root, output, workers=8, label_limit=0, labels=False, decode=False):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    annotation = root / 'datasets/llava-665k/llava_v1_5_mix665k.json'
    image_root = root / 'datasets/llava-665k/images'
    model = root / 'models/llava-1.5-7b-hf-b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
    vision = json.loads((model / 'config.json').read_text())['vision_config']
    patch_count = (vision['image_size'] // vision['patch_size']) ** 2
    counts, sources, image_paths = Counter(), Counter(), set()
    for index, row in rows(annotation):
        counts['records'] += 1
        image = row.get('image')
        if not image:
            counts['text_only'] += 1
        elif isinstance(image, str):
            counts['single_image'] += 1
            sources[image.split('/')[0]] += 1
            image_paths.add(image)
        else:
            counts['unsupported_image_schema'] += 1
    missing = sorted(path for path in image_paths if not (image_root / path).is_file())
    summary = {'schema_version': 1, 'counts': dict(counts), 'sources': dict(sources),
               'unique_images': len(image_paths), 'missing_images': missing,
               'decode': 'not_run', 'labels': 'not_run',
               'patch_count_from_config': patch_count, 'full_vision_forward': 'not_run',
               'final_train_dev_manifest': 'not_created'}
    (output / 'data-audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    if decode:
        failures = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            ordered = sorted(image_paths)
            # Bounded batches avoid enqueuing hundreds of thousands of futures.
            for start in range(0, len(ordered), 1024):
                failures.extend(x for x in pool.map(decode_image,
                    (image_root / p for p in ordered[start:start + 1024])) if x is not None)
        summary['decode'] = {'checked': len(image_paths), 'failures': failures}
        (output / 'data-audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    if labels:
        model = root / 'models/llava-1.5-7b-hf-b234b804b114d9e37bb655e11cbbb5f5e971b7a9'
        statuses = Counter()
        length_bins = Counter()
        errors = []
        iterator = rows(annotation)
        if label_limit:
            iterator = itertools.islice(iterator, label_limit)
        with ProcessPoolExecutor(max_workers=workers, initializer=init_labels,
                                 initargs=(str(model),)) as pool, \
                (output / 'label-audit.jsonl').open('w') as handle:
            while batch := list(itertools.islice(iterator, 512)):
                for result in pool.map(audit_labels, batch, chunksize=16):
                    statuses[result['status']] += 1
                    length = result.get('expanded_length')
                    if length is not None:
                        length_bins['<=2048' if length <= 2048 else '>2048'] += 1
                    if result['status'] == 'preprocessing_error' and len(errors) < 100:
                        errors.append(result)
                    handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                handle.flush()
                # Sparse progress: one line per 8192 samples.
                if sum(statuses.values()) % 8192 == 0:
                    print(json.dumps({'labels_checked': sum(statuses.values()),
                                      'statuses': dict(statuses)}), flush=True)
        summary['labels'] = {'checked': sum(statuses.values()), 'statuses': dict(statuses),
                             'length_bins': dict(length_bins), 'first_errors': errors,
                             'complete': sum(statuses.values()) == counts['records'],
                             'preprocessing_verified': not statuses['preprocessing_error']}
        (output / 'data-audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    blocked = bool(missing or (isinstance(summary['decode'], dict) and summary['decode']['failures'])
                   or (isinstance(summary['labels'], dict)
                   and not summary['labels']['preprocessing_verified']))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if blocked else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/llava-v1.5-assets-v1')
    parser.add_argument('--output', default='artifacts/cpu/data')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--labels', action='store_true')
    parser.add_argument('--decode', action='store_true')
    parser.add_argument('--label-limit', type=int, default=0)
    args = parser.parse_args()
    raise SystemExit(audit(args.data_root, args.output, args.workers,
                           args.label_limit, args.labels, args.decode))


if __name__ == '__main__':
    main()
