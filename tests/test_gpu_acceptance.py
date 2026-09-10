"""Bounded GPU acceptance fixtures; never substitute a CPU backend."""

import pytest
import torch

from prefix_ttt.ops.fla import fla_prefix
from prefix_ttt.ops.reference import sequential_prefix


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel acceptance NOT RUN"),
]


def _inputs(length, seed=42):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = [torch.randn(1, length, 2, 128, generator=generator, device="cuda",
                          dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    state = torch.randn(1, 2, 128, 128, generator=generator, device="cuda",
                        requires_grad=True)
    return (*values, state)


def _check(actual, expected, threshold, name, record_property):
    difference = actual.float() - expected.float()
    absolute = difference.abs().max().item()
    relative = (difference.norm() / expected.float().norm().clamp_min(1e-8)).item()
    record_property(name + "_max_absolute", absolute)
    record_property(name + "_relative_frobenius", relative)
    assert torch.isfinite(actual).all(), name
    assert relative <= threshold, (name, relative, absolute)


@pytest.mark.parametrize("length", [1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129])
def test_fla_lengths_nonzero_initial_state(length, record_property):
    q, k, v, state = _inputs(length)
    actual = fla_prefix(q, k, v, state)
    expected = sequential_prefix(q.float(), k.float(), v.float(), state)
    assert actual[1].dtype == torch.float32
    for name, a, e, threshold in zip(("output", "state"), actual, expected, (0.02, 0.01)):
        _check(a, e, threshold, name, record_property)


def test_fla_future_causality_and_past_influence(record_property):
    q, k, v, state = _inputs(65)
    with torch.no_grad():
        original, _ = fla_prefix(q, k, v, state)
        future_k, future_v = k.clone(), v.clone()
        # A perturbation inside the first tile must not affect earlier reads.
        future_k[:, 17:] += 2
        future_v[:, 17:] -= 3
        future, _ = fla_prefix(q, future_k, future_v, state)
        torch.testing.assert_close(future[:, :17], original[:, :17], rtol=0, atol=0)
        past_v = v.clone()
        past_v[:, :1] += 8
        past, _ = fla_prefix(q, k, past_v, state)
        effect = (past[:, 1:].float() - original[:, 1:].float()).norm().item()
        record_property("past_kv_effect_norm", effect)
        assert effect > 0
        # The first read must include the current write, not just the old S0.
        expected = sequential_prefix(q.float(), k.float(), past_v.float(), state)[0]
        _check(past[:, :1], expected[:, :1], 0.02, "inclusive_first_read", record_property)


def test_fla_segmented_tail_and_cross_state_gradients(record_property):
    inputs = _inputs(129)
    q, k, v, state = inputs
    outputs = []
    carried = state
    # Half tile, Local-32 boundary, full tile and a one-token tail.
    for start, end in zip((0, 17, 33, 97, 128), (17, 33, 97, 128, 129)):
        out, carried = fla_prefix(q[:, start:end], k[:, start:end], v[:, start:end], carried)
        outputs.append(out)
    segmented = (torch.cat(outputs, dim=1), carried)
    whole = fla_prefix(q, k, v, state)
    reference = sequential_prefix(q.float(), k.float(), v.float(), state)
    for label, result in (("segmented", segmented), ("whole", whole)):
        for name, a, e, threshold in zip(("output", "state"), result, reference, (0.02, 0.01)):
            _check(a, e, threshold, label + "_" + name, record_property)

    # A loss only on the last segment and final state must reach early K/V
    # through intermediate final-state backward, with no detach at boundaries.
    generator = torch.Generator(device="cuda").manual_seed(73)
    output_weight = torch.randn(segmented[0][:, -1:].shape, generator=generator, device="cuda")
    state_weight = torch.randn(state.shape, generator=generator, device="cuda")
    gradients = []
    for output, final_state in (segmented, reference):
        loss = (output[:, -1:].float() * output_weight).sum() + (final_state * state_weight).sum()
        gradients.append(torch.autograd.grad(loss, inputs, retain_graph=True))
    for name, actual, expected in zip(("q", "k", "v", "initial_state"), *gradients):
        _check(actual, expected, 0.05, "gradient_" + name, record_property)
    assert gradients[0][1][:, :17].float().norm() > 0
    assert gradients[0][2][:, :17].float().norm() > 0
