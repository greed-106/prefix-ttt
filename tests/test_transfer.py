import json

import pytest
import torch
from torch import nn

from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from prefix_ttt.ops.features import FeatureReadout
from prefix_ttt.transfer import TransferHooks, check_baselines


def test_transfer_preserves_teacher_and_only_trains_branches():
    torch.manual_seed(42)
    config = LlavaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=4, max_position_embeddings=128)
    teacher = LlavaLlamaForCausalLM(config).eval().requires_grad_(False)
    branches = nn.ModuleDict({str(i): FeatureReadout(32, 4, 8, seed=42+i) for i in range(2)})
    # Nonzero gates expose feature gradients and avoid hiding broken dependence.
    with torch.no_grad():
        for branch in branches.values():
            branch.gate_weight.normal_(std=.01)
    inputs = torch.randint(0, 48, (1, 35))
    with torch.no_grad():
        expected = teacher.get_model()(inputs, use_cache=False).last_hidden_state
    hooks = TransferHooks(teacher, branches, backend='reference')
    hooks.valid = torch.ones_like(inputs, dtype=torch.bool)
    hooks.visual = hooks.valid.clone()
    hooks.visual[:, 12:] = False
    hooks.denominator = 1
    try:
        with torch.no_grad():
            actual = teacher.get_model()(inputs, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in branches.parameters())
        assert not hooks.captured
        assert set(hooks.diagnostics) == {'0', '1'}
    finally:
        hooks.close()


@pytest.fixture
def native_baselines(tmp_path):
    paths = []
    for benchmark, count, metrics in (
        ('mme', 2374, ('mme_perception_score', 'mme_cognition_score')),
        ('pope', 9000, ('pope_accuracy', 'pope_precision', 'pope_recall',
                        'pope_f1_score', 'pope_yes_ratio')),
        ('gqa', 12578, ('exact_match',)),
    ):
        path = tmp_path / benchmark
        model_dir = path / 'local-model'
        model_dir.mkdir(parents=True)
        (model_dir / 'timestamp_results.json').write_text(json.dumps({
            'config': {'limit': None},
            'n-samples': {benchmark: {'original': count, 'effective': count}},
            'results': {benchmark: {f'{metric},none': 0.0 for metric in metrics}},
        }))
        (model_dir / f'timestamp_samples_{benchmark}.jsonl').write_text(
            ''.join(json.dumps({'doc_id': i}) + '\n' for i in range(count)))
        paths.append(path)
    return paths


def test_baseline_gate_accepts_complete_native_results(native_baselines):
    check_baselines(native_baselines)


@pytest.mark.parametrize('defect', ['limit', 'partial', 'nan', 'missing_metric',
                                   'missing_samples', 'duplicate_samples', 'ambiguous'])
def test_baseline_gate_rejects_invalid_native_results(native_baselines, defect):
    paths = native_baselines
    check_baselines(paths)
    path = next(paths[0].rglob('*_results.json'))
    value = json.loads(path.read_text())
    samples = path.with_name('timestamp_samples_mme.jsonl')
    if defect == 'limit':
        value['config']['limit'] = 2
    elif defect == 'partial':
        value['n-samples']['mme']['effective'] -= 1
    elif defect == 'nan':
        value['results']['mme']['mme_perception_score,none'] = float('nan')
    elif defect == 'missing_metric':
        del value['results']['mme']['mme_perception_score,none']
    elif defect == 'missing_samples':
        samples.unlink()
    elif defect == 'duplicate_samples':
        samples.write_text(samples.read_text().replace('"doc_id": 1}', '"doc_id": 0}', 1))
    elif defect == 'ambiguous':
        path.with_name('other_results.json').write_text(path.read_text())
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        check_baselines(paths)
