import torch
import pytest


from prefix_ttt.runtime import load_trainable, restore_rng, rng_state
from prefix_ttt.training import cosine_factor, token_normalized_ce


def test_summed_rank_gradients_equal_global_token_mean():
    torch.manual_seed(42)
    logits = torch.randn(3, 5, 7, requires_grad=True)
    labels = torch.randint(0, 7, (3, 5))
    labels[0, 2:] = -100
    targets = int(labels[:, 1:].ne(-100).sum())
    expected = torch.autograd.grad(token_normalized_ce(logits, labels, targets), logits)[0]
    actual = torch.autograd.grad(sum(token_normalized_ce(logits[i:i+1], labels[i:i+1], targets)
                                     for i in range(3)), logits)[0]
    torch.testing.assert_close(actual, expected)


def test_resume_rng_optimizer_and_full_schedule():
    import copy
    import random
    import numpy as np
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_factor(step, 5182))
    def update():
        optimizer.zero_grad()
        model(torch.randn(3, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        return random.random(), np.random.random()
    update()
    state = copy.deepcopy({'trainable': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(), 'rng': rng_state()})
    randoms = update()
    expected = copy.deepcopy(model.state_dict())
    load_trainable(model, state['trainable'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'])
    assert update() == randoms
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    with pytest.raises(ValueError):
        load_trainable(model, {})
