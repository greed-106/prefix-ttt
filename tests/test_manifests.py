import hashlib
import json
from collections import Counter

import pytest

from prefix_ttt.manifests import build, digest_json, largest_remainder


def fixture(tmp_path, special=None):
    rows = [{'id': f'id-{i}', 'image': f'coco/{i}.jpg' if i % 2 else None,
             'conversations': [{'from': 'human', 'value': f'Question {i}'},
                               {'from': 'gpt', 'value': f'Answer {i}'}]} for i in range(30)]
    if special:
        special(rows)
    audits = []
    for index, row in enumerate(rows):
        audits.append({'index': index, 'id': str(row.get('id', index)), 'image': row.get('image'),
                       'content_hash': hashlib.sha256(json.dumps(row['conversations'], ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                       'status': 'valid', 'expanded_length': 300 + index * 90,
                       'targets_before': 10, 'targets_after': 5})
    count = len(rows)
    summary = {'counts': {'records': count}, 'unique_images': len({r['image'] for r in rows if r.get('image')}),
               'missing_images': [], 'labels': {'checked': count, 'complete': True,
                'preprocessing_verified': True, 'statuses': {'valid': count}}}
    images = {'counts': {'records': count}, 'unique_images': summary['unique_images'], 'missing_images': [],
              'decode': {'checked': summary['unique_images'], 'failures': []}}
    paths = [tmp_path / n for n in ['annotation.json', 'labels.jsonl', 'summary.json', 'images.json']]

    def save():
        paths[0].write_text(json.dumps(rows))
        paths[1].write_text(''.join(json.dumps(a) + '\n' for a in audits))
        paths[2].write_text(json.dumps(summary))
        paths[3].write_text(json.dumps(images))

    save()
    def run(**kwargs):
        save()
        return build(*paths, dev_size=4, a_size=12, preprocess_files=[__file__], **kwargs)
    return rows, audits, summary, images, run


def test_reproducible_stratification_and_hashes(tmp_path):
    _, _, _, _, run = fixture(tmp_path)
    result = run()
    assert result == run()
    assert result['order_sha256']['train'] == digest_json(result['train'])
    assert set(result['A']) <= set(result['train'])
    assert not set(result['dev']) & set(result['train'])
    assert result['counts']['A'] == sum(result['A_strata_quotas'].values()) == 12
    assert result['counts']['valid'] == result['counts']['train'] + result['counts']['dev']
    assert result['preprocessing_source_sha256'] and len(result['inputs']) == 4
    assert result['order_sha256'] != run(seed=43)['order_sha256']


def test_transitive_groups_and_empty_identifiers(tmp_path):
    def special(rows):
        rows[0]['image'] = rows[1]['image']  # shared image
        rows[2]['image'] = 'coco/unique-2.jpg'
        rows[1]['id'] = rows[2]['id']       # transitive original id
        rows[2]['conversations'] = rows[3]['conversations']  # transitive content
        rows[4]['id'] = rows[5]['id'] = ''  # not a grouping key
        rows[6].pop('id')                  # no fabricated audit-index group
    _, _, _, _, run = fixture(tmp_path, special)
    for seed in (42, 43, 44):
        result = run(seed=seed)
        dev = set(result['dev'])
        assert len({i in dev for i in range(4)}) == 1
    assert result['counts']['valid_groups'] == 27


def test_excluded_rows_still_bridge_groups(tmp_path):
    def special(rows):
        rows[0]['image'] = rows[1]['image']
        rows[2]['image'] = 'coco/unique-2.jpg'
        rows[1]['id'] = rows[2]['id']
    _, audits, summary, _, run = fixture(tmp_path, special)
    audits[1]['status'] = 'protocol_answer_truncated'
    summary['labels']['statuses'] = {'valid': 29, 'protocol_answer_truncated': 1}
    result = run()
    assert (0 in result['dev']) == (2 in result['dev'])
    assert 1 not in result['train'] + result['dev'] + result['A']
    assert result['exclusions'] == [{'index': 1, 'reason': 'protocol_answer_truncated', 'category': 'protocol'}]


@pytest.mark.parametrize('change', ['incomplete', 'count', 'preproc_summary', 'preproc_row', 'decode_missing', 'decode_failure', 'decode_short', 'decode_not_run'])
def test_refuse_unverified_audits(tmp_path, change):
    _, audits, summary, images, run = fixture(tmp_path)
    if change == 'incomplete':
        summary['labels']['complete'] = False
    elif change == 'count':
        summary['labels']['checked'] -= 1
    elif change == 'preproc_summary':
        summary['labels']['statuses']['preprocessing_error'] = 1
    elif change == 'preproc_row':
        audits[3]['status'] = 'preprocessing_error'
    elif change == 'decode_missing':
        images['missing_images'] = ['missing.jpg']
    elif change == 'decode_failure':
        images['decode']['failures'] = [{'path': 'bad.jpg'}]
    elif change == 'decode_short':
        images['decode']['checked'] -= 1
    else:
        images['decode'] = 'not_run'
    with pytest.raises(ValueError):
        run()


@pytest.mark.parametrize('change', ['row_index', 'row_content', 'row_count', 'summary_status', 'empty_targets'])
def test_refuse_mismatched_inputs(tmp_path, change):
    _, audits, summary, _, run = fixture(tmp_path)
    if change == 'row_index':
        audits[3]['index'] = 0
    elif change == 'row_content':
        audits[3]['content_hash'] = 'bad'
    elif change == 'row_count':
        audits.pop()
    elif change == 'summary_status':
        summary['labels']['statuses'] = {'valid': 29, 'data_no_assistant_content': 1}
    else:
        audits[3]['targets_after'] = 0
    with pytest.raises(ValueError):
        run()


def test_largest_remainder_exact_and_tie_deterministic():
    assert largest_remainder({'a': 7, 'b': 2, 'c': 1}, 6) == {'a': 4, 'b': 1, 'c': 1}
    assert largest_remainder({'c': 1, 'b': 1, 'a': 1}, 2) == {'c': 0, 'b': 1, 'a': 1}
    with pytest.raises(ValueError):
        largest_remainder({'a': 2}, 3)


def test_actual_A_matches_quotas(tmp_path):
    _, audits, _, _, run = fixture(tmp_path)
    from prefix_ttt.manifests import length_bucket
    result = run()
    actual = Counter()
    for i in result['A']:
        row = audits[i]
        key = f"{'coco|image' if row['image'] else 'text_only|text'}|{length_bucket(row['expanded_length'])}"
        actual[key] += 1
    assert actual == Counter(result['A_strata_quotas'])


def test_id_namespace_avoids_source_collision_but_content_groups_cross_source(tmp_path):
    def special(rows):
        rows[0]['image'] = 'coco/a.jpg'
        rows[1]['image'] = 'vg/a.jpg'
        rows[2]['image'] = 'gqa/a.jpg'
        rows[0]['id'] = rows[1]['id'] = rows[2]['id'] = '42'
        rows[2]['conversations'] = rows[3]['conversations']
    _, _, _, _, run = fixture(tmp_path, special)
    result = run()
    # Only identical content joins rows 2 and 3; bare id collision joins none.
    assert result['counts']['valid_groups'] == 29
    for seed in range(4):
        result = run(seed=seed)
        assert (2 in result['dev']) == (3 in result['dev'])


def test_execution_snapshot_hash_is_distinct_from_current_source(tmp_path):
    _, _, _, _, run = fixture(tmp_path)
    snapshot = tmp_path / 'audit_data.executed.py'
    snapshot.write_text('# exact historical execution source\n')
    result = run(execution_source_files=[snapshot])
    assert result['audit_execution_source_sha256'] == {
        str(snapshot): hashlib.sha256(snapshot.read_bytes()).hexdigest()}
    assert result['audit_execution_source_sha256'] != result['preprocessing_source_sha256']
    assert run(execution_source_files=[])['audit_execution_source_sha256'] == {}
