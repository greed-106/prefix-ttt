import torch

from prefix_ttt.gpu_regression import compare, expanded_ids


def test_regression_comparison_fails_closed():
    value = torch.tensor([1., 2.])
    assert compare(value, value, max_abs=0, relative=0)['passed']
    assert not compare(value + 1, value, max_abs=0.1, relative=0.1)['passed']
    assert not compare(value * float('nan'), value, max_abs=1, relative=1)['passed']


def test_shared_token_ids_only_expand_image_sentinel():
    ids = torch.tensor([[1, -200, 7, 8]])
    assert expanded_ids(ids, 63, 4).tolist() == [[1, 63, 63, 63, 63, 7, 8]]
    assert expanded_ids(torch.tensor([[1, 2]]), 63, 4).tolist() == [[1, 2]]
