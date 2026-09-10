import pytest
import torch
import torch.nn.functional as F

from prefix_ttt.ops.reference import chunk_prefix, sequential_prefix
from prefix_ttt.ops.local import local_attention, local_attention_cached
from prefix_ttt.ops.features import FeatureReadout
from prefix_ttt.ops.fla import fla_prefix, recurrent_step, _pack


def fixture(length, dtype=torch.float64):
    generator = torch.Generator().manual_seed(42 + length)
    return [torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
            for shape in [(2, length, 2, 4)] * 3 + [(2, 2, 4, 4)]]


@pytest.mark.parametrize("length", [1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize("tile", [16, 32, 64, 128])
def test_reference_outputs_state_and_all_gradients(length, tile):
    args = fixture(length)
    expected = sequential_prefix(*args)
    actual = chunk_prefix(*args, tile_size=tile)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)
    # Nontrivial output AND tail-state upstream gradients.
    weights = [torch.randn_like(x) for x in actual]
    losses = [sum((x * w).sum() for x, w in zip(pair, weights)) for pair in (actual, expected)]
    gradients = [torch.autograd.grad(loss, args, retain_graph=True) for loss in losses]
    for a, e in zip(*gradients):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)


def test_causal_and_history_response():
    q, k, v, state = fixture(65)
    before, _ = chunk_prefix(q, k, v, state)
    future_k, future_v = k.detach().clone(), v.detach().clone()
    future_k[:, 18:] += 3
    future_v[:, 18:] -= 2
    after, _ = chunk_prefix(q, future_k, future_v, state)
    torch.testing.assert_close(before[:, :18], after[:, :18])
    past = v.detach().clone()
    past[:, 0] += 1
    changed, _ = chunk_prefix(q, k, past, state)
    assert (changed[:, -1] - before[:, -1]).norm() > 1e-6


@pytest.mark.parametrize("split", [1, 17, 31, 32, 33, 64])
def test_segments_and_state_backward(split):
    q, k, v, state = fixture(65)
    whole, tail = chunk_prefix(q, k, v, state)
    a, middle = chunk_prefix(q[:, :split], k[:, :split], v[:, :split], state)
    b, final = chunk_prefix(q[:, split:], k[:, split:], v[:, split:], middle)
    torch.testing.assert_close(torch.cat([a, b], 1), whole, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(final, tail, rtol=1e-9, atol=1e-10)
    g1 = torch.autograd.grad(b.square().sum() + final.square().sum(), (k, v, state), retain_graph=True)
    g2 = torch.autograd.grad(whole[:, split:].square().sum() + tail.square().sum(), (k, v, state))
    for x, y in zip(g1, g2):
        torch.testing.assert_close(x, y, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("left", [False, True])
def test_padding_outputs_state_gradients(left):
    q, k, v, state = fixture(65)
    valid = torch.arange(65)[None, :] < torch.tensor([33, 51])[:, None]
    if left:
        valid = valid.flip(1)
    out, final = chunk_prefix(q, k, v, state, valid)
    assert out[~valid].count_nonzero() == 0
    for row in range(2):
        args = [x[row:row + 1, valid[row]] for x in (q, k, v)]
        independent, tail = sequential_prefix(*args, state[row:row + 1])
        torch.testing.assert_close(out[row, valid[row]], independent[0])
        torch.testing.assert_close(final[row], tail[0])
    grads = torch.autograd.grad(out.sum() + final.sum(), (q, k, v))
    for grad in grads:
        assert grad[~valid].count_nonzero() == 0


@pytest.mark.parametrize("length", [1, 31, 32, 33, 65, 129])
def test_local_batched_matches_explicit_blocks(length):
    q, k, v, _ = fixture(length)
    valid = torch.ones(2, length, dtype=torch.bool)
    if length > 2:
        valid[0, :2] = False
        valid[1, -2:] = False
    actual = local_attention(q, k, v, valid)
    expected = torch.zeros_like(actual)
    for row in range(2):
        indices = valid[row].nonzero().flatten()
        for start in range(0, len(indices), 32):
            ix = indices[start:start + 32]
            projections = [x[row, ix].transpose(0, 1) for x in (q, k, v)]
            expected[row, ix] = F.scaled_dot_product_attention(*projections, is_causal=True).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    ga = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)


def test_zero_gate_then_feature_gradients():
    module = FeatureReadout(hidden_size=8, heads=2, head_dim=4).double()
    q, k, v, _ = fixture(5)
    x = torch.randn(2, 5, 8, dtype=torch.float64)
    memory, _ = chunk_prefix(module.features(q), module.features(k), v)
    result = module.readout(x, memory)
    assert result.count_nonzero() == 0
    result.sum().backward()
    assert module.gate_weight.grad.norm() > 0
    assert module.a_phi.grad.count_nonzero() == 0
    with torch.no_grad():
        module.gate_weight.fill_(0.1)
    module.zero_grad()
    memory, _ = chunk_prefix(module.features(q), module.features(k), v)
    module.readout(x, memory).square().sum().backward()
    assert module.a_phi.grad.norm() > 0
    assert module.b_phi.grad.norm() > 0


def test_cpu_fla_cannot_silently_fallback():
    q, k, v, _ = fixture(2)
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        fla_prefix(q, k, v)


def test_local_uses_one_block_batched_call(monkeypatch):
    original = F.scaled_dot_product_attention
    calls = []

    def record(q, k, v, **kwargs):
        calls.append((q.shape, k.shape))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", record)
    q, k, v, _ = fixture(129)
    local_attention(q, k, v)
    assert calls == [(torch.Size([10, 2, 32, 4]), torch.Size([10, 2, 32, 4]))]


def test_checkpoint_gradient_equivalence():
    from torch.utils.checkpoint import checkpoint

    q, k, v, state = fixture(65)
    direct = chunk_prefix(q, k, v, state)
    recomputed = checkpoint(chunk_prefix, q, k, v, state, use_reentrant=False)
    gradients = [torch.autograd.grad(sum(x.square().sum() for x in output), (q, k, v, state),
                                     retain_graph=True) for output in (direct, recomputed)]
    for a, e in zip(*gradients):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)


def test_fp32_and_bf16_state_accumulation():
    args = fixture(129, torch.float32)
    for a, e in zip(chunk_prefix(*args), sequential_prefix(*args)):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)
    q, k, v = [x.detach().bfloat16() for x in args[:3]]
    output, state = chunk_prefix(q, k, v)
    expected, expected_state = sequential_prefix(q.float(), k.float(), v.float())
    assert output.dtype == torch.bfloat16
    assert state.dtype == torch.float32
    assert (output.float() - expected).norm() / expected.norm() < 0.02
    torch.testing.assert_close(state, expected_state, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("splits", [[1, 31, 32, 33, 64, 65], [17, 47, 65], list(range(1, 66))])
@pytest.mark.parametrize("left", [False, True])
def test_cached_local_segments_outputs_and_gradients(splits, left):
    q, k, v, _ = fixture(65)
    valid = torch.arange(65)[None] < torch.tensor([51, 65])[:, None]
    if left:
        valid = valid.flip(1)
    expected = local_attention(q, k, v, valid)
    cache, start, pieces = None, 0, []
    for end in splits:
        output, cache = local_attention_cached(q[:, start:end], k[:, start:end], v[:, start:end],
                                                valid[:, start:end], cache)
        pieces.append(output)
        start = end
        assert cache.key.shape[1] == 32
        assert torch.equal(cache.lengths, cache.seen_tokens % 32)
    actual = torch.cat(pieces, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    ga = torch.autograd.grad(actual.square().sum(), (q, k, v), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (q, k, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-10)
    for row in range(2):
        length = int(cache.lengths[row])
        selected = k[row, valid[row]]
        torch.testing.assert_close(cache.key[row, :length], selected[-length:] if length else selected[:0])
        assert cache.key[row, length:].count_nonzero() == 0


def test_cached_local_finished_freezes_kv_and_position():
    q, k, v, _ = fixture(33)
    _, cache = local_attention_cached(q[:, :17], k[:, :17], v[:, :17])
    output, after = local_attention_cached(q[:, 17:], k[:, 17:], v[:, 17:], cache=cache,
                                           finished=torch.tensor([True, False]))
    assert output[0].count_nonzero() == 0
    assert after.seen_tokens.tolist() == [17, 33]
    torch.testing.assert_close(after.key[0], cache.key[0])
    torch.testing.assert_close(after.value[0], cache.value[0])
    expected = local_attention(q, k, v)
    torch.testing.assert_close(output[1], expected[1, 17:])


def test_varlen_packing_boundaries_empty_rows_and_gradients():
    q, k, v, _ = fixture(5)
    valid = torch.tensor([[False, False, False, False, False], [False, True, True, False, True]])
    packed, active, boundaries = _pack(q, k, v, valid)
    assert active.tolist() == [1]
    assert boundaries.tolist() == [0, 3]
    for actual, original in zip(packed, (q, k, v)):
        torch.testing.assert_close(actual[0], original[valid])
    grads = torch.autograd.grad(sum(x.sum() for x in packed), (q, k, v))
    for g in grads:
        assert g[~valid].count_nonzero() == 0
        assert g[valid].eq(1).all()
    _, active, boundaries = _pack(q, k, v, torch.zeros_like(valid))
    assert active.numel() == 0
    assert boundaries.tolist() == [0]


def test_single_step_fp32_state_and_gradients_under_autocast():
    q, k, v, state = fixture(1, torch.float32)
    valid = torch.tensor([[True], [False]])
    expected = sequential_prefix(q, k, v, state, valid)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = recurrent_step(q, k, v, state, valid)
    assert actual[1].dtype == torch.float32
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual[1][1], state[1])
    gradients = [torch.autograd.grad(sum(x.square().sum() for x in outputs), (q, k, v, state),
                                     retain_graph=True) for outputs in (actual, expected)]
    for a, e in zip(*gradients):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("operator", [sequential_prefix, chunk_prefix])
def test_reference_autocast_does_not_reduce_accumulation_precision(operator):
    args = fixture(65, torch.float32)
    expected = operator(*args)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = operator(*args)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not mounted: FLA kernel acceptance NOT RUN")
@pytest.mark.parametrize("tile", [16, 32, 64, 128])
def test_fla_gpu_varlen_output_state_and_gradients(tile):
    generator = torch.Generator(device="cuda").manual_seed(42)
    tensors = [torch.randn(3, 65, 2, 128, generator=generator, device="cuda", dtype=torch.bfloat16,
                           requires_grad=True) for _ in range(3)]
    state = torch.randn(3, 2, 128, 128, generator=generator, device="cuda", requires_grad=True)
    valid = torch.arange(65, device="cuda")[None] < torch.tensor([0, 33, 65], device="cuda")[:, None]
    actual = fla_prefix(*tensors, state, valid=valid, tile_size=tile)
    # Compare against the same quantized input, without changing its values.
    expected = sequential_prefix(*(x.float() for x in tensors), state, valid)
    for a, e, threshold in zip(actual, expected, (0.02, 0.01)):
        error = (a.float() - e).norm() / e.norm().clamp_min(1e-8)
        assert error < threshold, (tile, error.item(), (a.float() - e).abs().max().item())
    weights = [torch.randn_like(x.float()) for x in actual]
    gradients = [torch.autograd.grad(sum((x.float() * w).sum() for x, w in zip(outputs, weights)),
                                     (*tensors, state), retain_graph=True) for outputs in (actual, expected)]
    for a, e in zip(*gradients):
        error = (a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-8)
        assert error < 0.05, (tile, error.item(), (a.float() - e.float()).abs().max().item())
