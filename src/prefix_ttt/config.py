"""Load the fixed recipe and prove that the file still describes this code.

``configs/base.json`` records the whole recipe, but a run only reads a few keys from
it; the rest are mirrored by constants in the code. Rather than let the two drift
apart silently, every entry point loads the file through :func:`load_config`, which
compares the mirrored values and fails at start-up instead of hours later.

Keys that are pure documentation (``project``, ``schema_version``, ``use_cache``,
``checkpoint_reentrant``) are listed at the end and deliberately not compared.
"""
import json
from pathlib import Path

from prefix_ttt.model.bridge import (IMAGE_ASPECT_RATIO, MAX_EXPANDED_LENGTH, PINNED_HEAD_DIM,
                                     PINNED_HEADS, PINNED_LAYERS)
from prefix_ttt.model.generation import DO_SAMPLE, MAX_NEW_TOKENS, NUM_BEAMS
from prefix_ttt.model.hybrid import LAYOUT_ANCHORS
from prefix_ttt.model.trainability import (LORA_ALPHA, LORA_BIAS, LORA_DROPOUT, LORA_RANK,
                                           LORA_SEED)
from prefix_ttt.ops import ETA, LOCAL_BLOCK_SIZE, TILE_SIZE
from prefix_ttt.ops.features import RMS_EPS
from prefix_ttt.runtime import SEED
from prefix_ttt.training import (ADAM_BETAS, ADAM_EPS, A_STAGE_PASSES, BASE_LR, EFFECTIVE_BATCH_SIZE,
                                 GRAD_CLIP, KD_TEMPERATURE, KD_WEIGHT, LORA_LR,
                                 MATRIX_WEIGHT_DECAY, NEW_MODULE_LR, PILOT_MIN_SAMPLES,
                                 WARMUP_FRACTION)
from prefix_ttt.data_pipeline import CONV_TEMPLATE
from prefix_ttt.manifests import A_STAGE_SAMPLES


def _mirrored(config):
    """Every decorative key that the code also states, as (path, file value, code value)."""
    generation = config['generation']
    training = config['training']
    rows = [
        ('seed', config['seed'], SEED),
        ('lora_seed', config['lora_seed'], LORA_SEED),
        ('template', config['template'], CONV_TEMPLATE),
        ('image_aspect_ratio', config['image_aspect_ratio'], IMAGE_ASPECT_RATIO),
        ('max_expanded_length', config['max_expanded_length'], MAX_EXPANDED_LENGTH),
        ('num_heads', config['num_heads'], PINNED_HEADS),
        ('head_dim', config['head_dim'], PINNED_HEAD_DIM),
        ('feature_dim', config['feature_dim'], PINNED_HEAD_DIM),
        ('local_block_size', config['local_block_size'], LOCAL_BLOCK_SIZE),
        ('tile_size', config['tile_size'], TILE_SIZE),
        ('eta', config['eta'], ETA),
        ('rms_eps', config['rms_eps'], RMS_EPS),
        ('lora.rank', config['lora']['rank'], LORA_RANK),
        ('lora.alpha', config['lora']['alpha'], LORA_ALPHA),
        ('lora.dropout', config['lora']['dropout'], LORA_DROPOUT),
        ('lora.bias', config['lora']['bias'], LORA_BIAS),
        ('training.effective_batch_size', training['effective_batch_size'], EFFECTIVE_BATCH_SIZE),
        ('training.stage_a_samples', training['stage_a_samples'], A_STAGE_SAMPLES),
        ('training.pilot_min_samples', training['pilot_min_samples'], PILOT_MIN_SAMPLES),
        ('training.new_module_lr', training['new_module_lr'], NEW_MODULE_LR),
        ('training.lora_lr', training['lora_lr'], LORA_LR),
        ('training.betas', training['betas'], list(ADAM_BETAS)),
        ('training.eps', training['eps'], ADAM_EPS),
        ('training.matrix_weight_decay', training['matrix_weight_decay'], MATRIX_WEIGHT_DECAY),
        ('training.warmup_fraction', training['warmup_fraction'], WARMUP_FRACTION),
        ('training.grad_clip', training['grad_clip'], GRAD_CLIP),
        ('generation.do_sample', generation['do_sample'], DO_SAMPLE),
        ('generation.num_beams', generation['num_beams'], NUM_BEAMS),
        ('generation.max_new_tokens', generation['max_new_tokens'], MAX_NEW_TOKENS),
    ]
    if 'base_lr' in training:
        rows.append(('training.base_lr', training['base_lr'], BASE_LR))
    if 'stage_a_passes' in training:
        rows.append(('training.stage_a_passes', training['stage_a_passes'], A_STAGE_PASSES))
    if 'kd_weight' in training:
        rows.append(('training.kd_weight', training['kd_weight'], KD_WEIGHT))
    if 'kd_temperature' in training:
        rows.append(('training.kd_temperature', training['kd_temperature'], KD_TEMPERATURE))
    return rows


def load_config(path, *, require_base_lr=False):
    """Read the recipe, check every mirrored value, and return it.

    ``require_base_lr`` is set by the full fine-tuning entry point: only that arm
    reads the base learning rate, so only that arm insists the file states it.
    """
    config = json.loads(Path(path).read_text())
    if require_base_lr and 'base_lr' not in config['training']:
        raise ValueError('the full fine-tuning recipe must state training.base_lr')
    for key, recorded, implemented in _mirrored(config):
        if recorded != implemented:
            raise ValueError(f'configs recipe mismatch at {key}: file={recorded!r} '
                             f'code={implemented!r}; update the file or the constant')
    anchors = tuple(config['full_attention_layers'])
    if anchors not in LAYOUT_ANCHORS.values():
        raise ValueError('configs anchor layers are not a supported layout: '
                         f'{sorted(LAYOUT_ANCHORS)}')
    if set(config['candidate_ttt_layers']) | set(anchors) != set(range(PINNED_LAYERS)):
        raise ValueError('candidate_ttt_layers must cover every non-anchor layer')
    return config
