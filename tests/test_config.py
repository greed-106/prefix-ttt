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
