import torch
from torch import nn

from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from prefix_ttt.ops.features import FeatureReadout
from prefix_ttt.training import visual_queries
from prefix_ttt.transfer import TransferHooks


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
    hooks.positions = visual_queries(hooks.valid, hooks.visual)
    hooks.denominator = 1
    try:
        with torch.no_grad():
            actual = teacher.get_model()(inputs, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert all(p.grad is None for p in teacher.parameters())
        # The re-derived attention call must reproduce the teacher's own output, or
        # the vision-only target would use a different softmax than the teacher did.
        assert float(hooks.consistency) == 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in branches.parameters())
        assert not hooks.captured
        assert set(hooks.diagnostics) == {'0', '1'}
    finally:
        hooks.close()
