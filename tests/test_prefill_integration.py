"""Production dispatch, cache semantics and whole/segmented prefill agreement."""

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding

from prefix_ttt.model.generation import TransformersHybridCache
from prefix_ttt.model.hybrid import PrefixTTTAttention


pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA integration NOT RUN')]


def relative(actual, expected, threshold):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).norm()
    assert error / expected.float().norm().clamp_min(1e-8) <= threshold


@pytest.mark.parametrize('kind', ['dense', 'padding', 'holes', 'empty_row', 'finished'])
@torch.inference_mode()
def test_production_prefill_dispatch_and_cache(kind, monkeypatch):
    import prefix_ttt.model.hybrid as hybrid
    torch.manual_seed(71)
    config = LlamaConfig(hidden_size=256, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=256)
    layer = PrefixTTTAttention(LlamaAttention(config, 0).cuda().bfloat16(), backend='fla')
    layer.prefix_ttt.gate_weight.normal_(std=.02)
    layer.prefix_ttt.prepare_inference()
    rope = LlamaRotaryEmbedding(config, device='cuda')
    x = torch.randn(2, 65, 256, device='cuda', dtype=torch.bfloat16)
    valid = torch.ones(2, 65, device='cuda', dtype=torch.bool)
    if kind == 'padding':
        valid[1, -4:] = False
    elif kind == 'holes':
        valid[1, 5::7] = False
    elif kind == 'empty_row':
        valid[1] = False
    calls = []
    dense = hybrid.dense_local
    def record(*args, **kwargs):
        calls.append(1)
        return dense(*args, **kwargs)
    monkeypatch.setattr(hybrid, 'dense_local', record)

    def run(force_original=False):
        cache = TransformersHybridCache(2, [], 'cuda')
        if kind == 'finished':
            cache.storage.mark_finished(torch.tensor([False, True], device='cuda'))
        _, positions = cache.begin(valid)
        expected_flag = kind == 'dense'
        assert cache._dense_prefill == expected_flag
        if force_original:
            cache._dense_prefill = False
        out = layer(x, rope(x, positions), past_key_value=cache, use_cache=True)[0]
        cache.finish()
        assert not cache._dense_prefill
        return out, cache

    actual, cache = run()
    expected, reference = run(force_original=True)
    assert len(calls) == (1 if kind == 'dense' else 0)
    relative(actual, expected, .02)
    for field in ('state', 'key', 'value', 'local_position'):
        a = getattr(cache.storage.layers[0], field)
        e = getattr(reference.storage.layers[0], field)
        if field == 'state':
            relative(a, e, .01)
        else:
            torch.testing.assert_close(a, e, rtol=0, atol=0)
    torch.testing.assert_close(cache.valid_history, reference.valid_history)
    torch.testing.assert_close(cache.storage.seen_tokens, reference.storage.seen_tokens)
    invalid = ~cache.valid_history
    assert torch.count_nonzero(actual[invalid]) == 0


@torch.inference_mode()
def test_production_segmented_prefill_and_decode_keep_old_state():
    torch.manual_seed(81)
    config = LlamaConfig(hidden_size=256, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=256)
    layer = PrefixTTTAttention(LlamaAttention(config, 0).cuda().bfloat16(), backend='fla')
    layer.prefix_ttt.gate_weight.normal_(std=.02)
    layer.prefix_ttt.prepare_inference()
    rope = LlamaRotaryEmbedding(config, device='cuda')
    x = torch.randn(2, 66, 256, device='cuda', dtype=torch.bfloat16)

    def forward(part, cache):
        valid = torch.ones(part.shape[:2], device='cuda', dtype=torch.bool)
        _, positions = cache.begin(valid)
        out = layer(part, rope(part, positions), past_key_value=cache, use_cache=True)[0]
        cache.finish()
        return out

    whole_cache = TransformersHybridCache(2, [], 'cuda')
    whole = forward(x[:, :65], whole_cache)
    old = whole_cache.storage.layers[0].state
    saved = old.clone()
    last = forward(x[:, 65:], whole_cache)
    torch.testing.assert_close(old, saved, rtol=0, atol=0)
    split_cache = TransformersHybridCache(2, [], 'cuda')
    split = [forward(x[:, start:end], split_cache)
             for start, end in ((0, 17), (17, 33), (33, 65), (65, 66))]
    relative(torch.cat(split, dim=1), torch.cat((whole, last), dim=1), .02)
    relative(split_cache.storage.layers[0].state, whole_cache.storage.layers[0].state, .01)
    torch.testing.assert_close(split_cache.storage.layers[0].key, whole_cache.storage.layers[0].key,
                               rtol=0, atol=0)
