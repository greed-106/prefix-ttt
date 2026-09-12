import json

import pytest

from prefix_ttt.data_pipeline import load_manifest, micro_batches
from prefix_ttt.digests import digest_file, digest_json


def test_micro_batches_partition_one_group_without_repeats():
    order = list(range(133))
    for cursor in (0, 128):
        for world_size in (1, 4, 8):
            batches = [batch for rank in range(world_size)
                       for batch in micro_batches(order, cursor, rank, world_size, 8)]
            flat = sum(batches, [])
            assert sorted(flat) == order[cursor:cursor + 128]
            assert len(flat) == len(set(flat))
            assert all(len(batch) <= 8 for batch in batches)


def test_micro_batches_keep_manifest_order():
    order = list(range(256))
    assert micro_batches(order, 128, 0, 2, 4)[0] == [128, 130, 132, 134]
    assert micro_batches(order, 128, 1, 2, 4)[0] == [129, 131, 133, 135]


def write_manifest(tmp_path, train=(0, 1, 2, 3), dev=(4,), a=(0,)):
    """A manifest shaped like artifacts/cpu/fixed_manifest.json, but tiny.

    The real file is 8 MB and its annotation is far larger, so the entry point is
    exercised here on a synthetic one: every check in load_manifest still runs.
    """
    annotation = tmp_path / 'annotation.json'
    annotation.write_text('[]')
    manifest = {'train': list(train), 'dev': list(dev), 'A': list(a),
                'annotation': str(annotation), 'inputs': {str(annotation): digest_file(annotation)},
                'order_sha256': {split: digest_json(list(values))
                                 for split, values in (('train', train), ('dev', dev), ('A', a))}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_load_manifest_verifies_every_split_order(tmp_path):
    path, manifest = write_manifest(tmp_path)
    loaded, sha = load_manifest(path, tmp_path)
    assert sha == digest_file(path)
    assert loaded['train'] == manifest['train']
    assert loaded['annotation'] == str(tmp_path / 'annotation.json')


def test_load_manifest_rejects_a_tampered_order_or_annotation(tmp_path):
    path, manifest = write_manifest(tmp_path)
    tampered = dict(manifest, order_sha256=dict(manifest['order_sha256'], train='0' * 64))
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match='Invalid fixed train order'):
        load_manifest(path, tmp_path)

    path, manifest = write_manifest(tmp_path)
    (tmp_path / 'annotation.json').write_text('[1]')
    with pytest.raises(ValueError, match='Original annotation changed after audit'):
        load_manifest(path, tmp_path)

    path, manifest = write_manifest(tmp_path, dev=(0,))
    with pytest.raises(ValueError, match='split separation'):
        load_manifest(path, tmp_path)

