"""Correctness of the production all-valid prefill operators."""

import pytest
import torch

from prefix_ttt.ops.features import FeatureReadout
from prefix_ttt.ops.fla import fla_prefix, recurrent_step
from prefix_ttt.ops.local import dense_local, local_attention_cached, local_attention_decode
from prefix_ttt.ops.reference import sequential_prefix


LENGTHS = (0, 1, 31, 32, 33, 63, 64, 65, 129)
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU prefill acceptance NOT RUN")


def _inputs(length, *, device="cpu", dtype=torch.float64, dim=4, strided=False):
    generator = torch.Generator(device=device).manual_seed(412 + length)
    values = [torch.randn(2, length * (2 if strided else 1), 2, dim,
                          device=device, dtype=dtype, generator=generator) for _ in range(3)]
    return [x[:, ::2] for x in values] if strided else values


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("strided", [False, True])
def test_dense_local_fp64_output_and_cache(length, strided):
    q, k, v = _inputs(length, strided=strided)
    actual, cache = dense_local(q, k, v)
    expected, reference_cache = local_attention_cached(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    for field in ("key", "value", "lengths", "seen_tokens"):
        torch.testing.assert_close(getattr(cache, field), getattr(reference_cache, field), rtol=0, atol=0)
    assert cache.lengths.tolist() == [length % 32] * 2
    assert cache.seen_tokens.tolist() == [length] * 2
    uncached, absent = dense_local(q, k, v, need_cache=False)
    torch.testing.assert_close(uncached, expected, rtol=1e-9, atol=1e-10)
    assert absent is None


@pytest.mark.parametrize("length", [31, 32, 33, 63, 64, 65])
def test_dense_local_cache_continues_decode_and_segmented_prefill(length):
    q, k, v = _inputs(length + 35)
    first, cache = dense_local(q[:, :length], k[:, :length], v[:, :length])
    before = {field: getattr(cache, field).clone() for field in ("key", "value", "lengths", "seen_tokens")}
    next_qkv = [x[:, length:length + 1] for x in (q, k, v)]
    one, after_one = local_attention_decode(*next_qkv, torch.ones(2, 1, dtype=torch.bool), cache)
    rest, final = local_attention_cached(q[:, length + 1:], k[:, length + 1:], v[:, length + 1:],
                                         cache=after_one)
    expected, expected_cache = local_attention_cached(q, k, v)
    torch.testing.assert_close(torch.cat((first, one, rest), dim=1), expected, rtol=1e-9, atol=1e-10)
    for field in before:
        torch.testing.assert_close(getattr(cache, field), before[field], rtol=0, atol=0)
        torch.testing.assert_close(getattr(final, field), getattr(expected_cache, field), rtol=0, atol=0)


def test_features_cpu_autograd_keeps_original_path():
    module = FeatureReadout(hidden_size=8, heads=2, head_dim=4).double()
    q, k, _ = [x.requires_grad_() for x in _inputs(33, strided=True)]
    actual = module.features_prefill(q, k)
    expected = (module.features(q), module.features(k))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)
    inputs = (q, k, module.a_phi, module.b_phi)
    gradients = [torch.autograd.grad(sum(x.square().sum() for x in pair), inputs, retain_graph=True)
                 for pair in (actual, expected)]
    for a, e in zip(*gradients):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("length", [31, 32, 33, 63, 64, 65, 129])
@pytest.mark.parametrize("strided", [False, True])
@torch.no_grad()
def test_dense_local_bf16_output_and_cache(length, strided):
    q, k, v = _inputs(length, device="cuda", dtype=torch.bfloat16, dim=128, strided=strided)
    actual, cache = dense_local(q, k, v)
    expected, reference_cache = local_attention_cached(q, k, v)
    torch.testing.assert_close(actual, expected)
    for field in ("key", "value", "lengths", "seen_tokens"):
        torch.testing.assert_close(getattr(cache, field), getattr(reference_cache, field), rtol=0, atol=0)


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("length", [1, 31, 32, 33, 63, 64, 65, 129])
@pytest.mark.parametrize("strided", [False, True])
@torch.no_grad()
def test_features_prefill_bf16_preserves_rounding(length, strided):
    module = FeatureReadout(hidden_size=256, heads=2, head_dim=128).cuda()
    module.prepare_inference()
    q, k, _ = _inputs(length, device="cuda", dtype=torch.bfloat16, dim=128, strided=strided)
    actual = module.features_prefill(q, k)
    expected = (module.features(q), module.features(k))
    assert hasattr(module, "ab_phi_inference"), "prefill weights must be prepared before acceptance"
    for a, e in zip(actual, expected):
        assert a.dtype == torch.bfloat16
        torch.testing.assert_close(a, e)
    assert module.a_phi.dtype == module.b_phi.dtype == torch.float32
    assert "ab_phi_inference" not in module.state_dict()


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("length", [31, 32, 33, 63, 64, 65])
@pytest.mark.parametrize("tile", [16, 32, 64, 128])
@torch.no_grad()
def test_dense_prefix_matches_fp64_oracle_nonzero_state(length, tile, record_property):
    q, k, v = _inputs(length, device="cuda", dtype=torch.bfloat16, dim=128, strided=True)
    generator = torch.Generator(device="cuda").manual_seed(19)
    state = torch.randn(2, 2, 128, 128, device="cuda", generator=generator)
    before = state.clone()
    actual = fla_prefix(q, k, v, state, tile_size=tile)
    reference = sequential_prefix(*(x.cpu().double() for x in (q, k, v)), state.cpu().double())
    baseline = fla_prefix(q, k, v, state, valid=torch.ones(q.shape[:2], device=q.device,
                                                       dtype=torch.bool), tile_size=tile)
    for name, a, e, old, threshold in zip(("output", "state"), actual, reference, baseline, (0.02, 0.01)):
        difference = a.cpu().double() - e
        relative = (difference.norm() / e.norm().clamp_min(1e-8)).item()
        absolute = difference.abs().max().item()
        record_property(name + "_relative_l2", relative)
        record_property(name + "_max_absolute", absolute)
        assert torch.isfinite(a).all()
        assert relative < threshold, (tile, length, name, relative, absolute)
        torch.testing.assert_close(a, old, rtol=0, atol=0)
    assert actual[1].dtype == torch.float32
    torch.testing.assert_close(state, before, rtol=0, atol=0)
    without_state, absent = fla_prefix(q, k, v, state, need_final_state=False, tile_size=tile)
    torch.testing.assert_close(without_state, actual[0], rtol=0, atol=0)
    assert absent is None


@pytest.mark.gpu
@gpu
@pytest.mark.parametrize("length", [0, 31, 32, 33, 63, 64, 65])
@torch.no_grad()
def test_dense_prefix_empty_initial_state_and_decode(length):
    q, k, v = _inputs(length + 1, device="cuda", dtype=torch.bfloat16, dim=128)
    out, state = fla_prefix(q[:, :length], k[:, :length], v[:, :length])
    before = state.clone()
    next_out, final = recurrent_step(q[:, length:], k[:, length:], v[:, length:], state)
    reference = sequential_prefix(*(x.float() for x in (q, k, v)))
    for a, e, threshold in zip((torch.cat((out, next_out), dim=1), final), reference, (0.02, 0.01)):
        assert (a.float() - e).norm() / e.norm().clamp_min(1e-8) < threshold
    torch.testing.assert_close(state, before, rtol=0, atol=0)
    if length == 0:
        _, carried = fla_prefix(q[:, :0], k[:, :0], v[:, :0], final)
        torch.testing.assert_close(carried, final, rtol=0, atol=0)
