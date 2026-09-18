import json
from pathlib import Path

import pytest

from prefix_ttt.config import load_config


def test_shipped_recipe_matches_the_code():
    config = load_config('configs/base.json')
    assert config['eta'] * 128 == 1
    assert config['training']['micro_batch_size'] > 0


@pytest.mark.parametrize('key', ['eta', 'local_block_size', 'max_expanded_length', 'grad_clip'])
def test_mirrored_value_drift_is_rejected(tmp_path, key):
    config = json.loads(Path('configs/base.json').read_text())
    if key in config:
        config[key] = 12345
    else:
        config['training'][key.split('.')[-1]] = -1
    path = tmp_path / 'base.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='mismatch|differ'):
        load_config(path)


def test_anchor_drift_is_rejected(tmp_path):
    config = json.loads(Path('configs/base.json').read_text())
    config['full_attention_layers'] = config['full_attention_layers'][:-1]
    path = tmp_path / 'base.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='anchor'):
        load_config(path)


def test_pure_ttt_recipe_declares_every_layer_and_distillation():
    config = load_config('configs/p32.json')
    assert config['full_attention_layers'] == []
    assert config['candidate_ttt_layers'] == list(range(32))
    assert config['training']['kd_weight'] == 1.0
    assert config['training']['kd_temperature'] == 1.0


def test_pure_ttt_recipe_is_checked_against_the_code(tmp_path):
    config = json.loads(Path('configs/p32.json').read_text())
    config['training']['kd_weight'] = 0.5
    path = tmp_path / 'p32.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='kd_weight'):
        load_config(path)
