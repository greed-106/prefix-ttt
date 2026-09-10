import pytest
import torch

from prefix_ttt.cache import HybridCache, LayerState


def test_counters_finished_and_selection_independent_of_layer_zero():
    cache = HybridCache(2, [3, 7])
    cache.advance(torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool))
    cache.set_layer(0, LayerState(state=torch.randn(2, 2, 4, 4),
                                 key=torch.randn(2, 4, 2, 4), value=torch.randn(2, 4, 2, 4),
                                 local_position=torch.tensor([2, 4])))
    cache.mark_finished(torch.tensor([True, False]))
    cache.advance(torch.ones(2, 1, dtype=torch.bool))
    assert cache.next_position.tolist() == [2, 4]
    chosen = cache.select(torch.tensor([1, 0, 1]))
    assert chosen.seen_tokens.tolist() == [4, 2, 4]
    assert chosen.finished.tolist() == [False, True, False]
    assert chosen.layers[0].local_position.tolist() == [4, 2, 4]
    torch.testing.assert_close(chosen.layers[0].state[0], cache.layers[0].state[1])
    assert chosen.tensor_bytes > cache.tensor_bytes
    chosen.layers[0].state.zero_()
    assert cache.layers[0].state.count_nonzero() > 0
    assert HybridCache(2, [3, 7]).seen_tokens.tolist() == [0, 0]


def test_storage_invariants():
    cache = HybridCache(1, [3])
    with pytest.raises(ValueError, match="FP32"):
        cache.set_layer(0, LayerState(state=torch.zeros(1, 2, 4, 4).bfloat16()))
    with pytest.raises(ValueError, match="capacity"):
        cache.set_layer(0, LayerState(key=torch.zeros(1, 33, 2, 4), value=torch.zeros(1, 33, 2, 4)))
    with pytest.raises(ValueError, match="MSA"):
        cache.set_layer(3, LayerState(state=torch.zeros(1, 2, 4, 4)))


def test_finished_protects_all_layer_fields_and_full_kv_growth():
    cache = HybridCache(2, [3])
    cache.set_layer(0, LayerState(state=torch.ones(2, 2, 4, 4),
                                 key=torch.ones(2, 32, 2, 4), value=torch.ones(2, 32, 2, 4),
                                 local_position=torch.tensor([3, 4])))
    cache.set_layer(3, LayerState(key=torch.ones(2, 5, 2, 4), value=torch.ones(2, 5, 2, 4)))
    cache.mark_finished(torch.tensor([True, False]))
    cache.set_layer(0, LayerState(state=torch.full((2, 2, 4, 4), 9.),
                                 key=torch.full((2, 32, 2, 4), 9.), value=torch.full((2, 32, 2, 4), 9.),
                                 local_position=torch.tensor([8, 8])))
    assert cache.layers[0].state[0].eq(1).all()
    assert cache.layers[0].key[0].eq(1).all()
    assert cache.layers[0].value[0].eq(1).all()
    assert cache.layers[0].state[1].eq(9).all()
    assert cache.layers[0].local_position.tolist() == [3, 8]
    cache.set_layer(3, LayerState(key=torch.full((2, 7, 2, 4), 9.), value=torch.full((2, 7, 2, 4), 9.)))
    assert cache.layers[3].key[0, :5].eq(1).all()
    assert cache.layers[3].key[0, 5:].eq(0).all()
    assert cache.layers[3].key[1].eq(9).all()
