import copy

import pytest
import torch

from prefix_ttt.training import (accumulation_steps, shifted_target_count,
    token_normalized_ce, full_attention_transfer_loss, visual_transfer_loss, visual_queries,
    trajectory, cosine_factor, optimizer_groups)


def test_accumulation_matches_large_batch_and_ddp_mean():
    torch.manual_seed(4)
    layer = torch.nn.Linear(5, 11)
    inputs = torch.randn(5, 9, 5)
    labels = torch.randint(0, 11, (5, 9))
    labels[0, 2:] = -100
    labels[1, 4:] = -100
    labels[3] = -100  # empty microbatch, but not empty group
    targets = shifted_target_count(labels)
    oracle = copy.deepcopy(layer)
    token_normalized_ce(oracle(inputs), labels, targets).backward()
    for i in range(5):
        token_normalized_ce(layer(inputs[i:i+1]), labels[i:i+1], targets).backward()
    for a, b in zip(layer.parameters(), oracle.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    ranks = [copy.deepcopy(oracle), copy.deepcopy(oracle)]
    for rank, indices in zip(ranks, ([0, 1, 2], [3, 4])):
        rank.zero_grad()
        token_normalized_ce(rank(inputs[indices]), labels[indices], targets, world_size=2).backward()
    for a, b, target in zip(ranks[0].parameters(), ranks[1].parameters(), oracle.parameters()):
        torch.testing.assert_close((a.grad + b.grad) / 2, target.grad)


def test_stage_a_balances_modalities_and_detaches_teacher():
    full = torch.ones(2, 4, 2, 3, requires_grad=True)
    readout = torch.zeros_like(full, requires_grad=True)
    visual = torch.tensor([[True, False, False, False], [False, False, False, False]])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    loss = full_attention_transfer_loss(readout, full, visual, valid)
    torch.testing.assert_close(loss, torch.full((2,), 1 / (1 + 1e-6)))
    loss.sum().backward()
    assert full.grad is None
    assert readout.grad[0, 0].abs().sum() == pytest.approx(readout.grad[0, 1:].abs().sum().item())
    assert readout.grad[1, 2:].count_nonzero() == 0


def test_stage_a_vision_loss_uses_the_full_output_energy():
    target = torch.ones(2, 4, 2, 3, requires_grad=True)
    readout = torch.zeros_like(target, requires_grad=True)
    positions = torch.tensor([[True, True, False, False], [False, False, False, False]])
    energy = torch.tensor([0.5, 0.5])
    loss = visual_transfer_loss(readout, target, positions, energy)
    # The error is divided by the teacher's complete-output energy, never by the
    # (smaller) vision-only energy; a sample without supervised positions is zero.
    torch.testing.assert_close(loss[0], torch.tensor(1.0 / (0.5 + 1e-6)))
    assert loss[1] == 0
    loss.sum().backward()
    assert target.grad is None
    assert readout.grad[1].count_nonzero() == 0


def test_visual_queries_are_the_first_text_positions_after_the_image():
    valid = torch.ones(3, 6, dtype=torch.bool)
    valid[1, 4:] = False
    image = torch.tensor([[False, True, True, False, False, False],
                          [False, False, False, False, False, False],
                          [False, True, True, False, False, False]])
    assert visual_queries(valid, image, limit=2).tolist() == [
        [False, False, False, True, True, False],
        [False, False, False, False, False, False],
        [False, False, False, True, True, False]]
    with pytest.raises(ValueError):
        visual_queries(valid, image, limit=0)


def test_trajectory_and_resume_do_not_restart_schedule():
    assert accumulation_steps(2, 1) == 64
    with pytest.raises(ValueError):
        accumulation_steps(3, 1)
    plan = trajectory(660000)
    assert plan['pilot_step'] == 391 and plan['pilot_samples'] == 50048
    assert plan['total_steps'] == 5157
    before = [cosine_factor(i, plan['total_steps']) for i in range(400)]
    resumed = [cosine_factor(i, plan['total_steps']) for i in range(391, 400)]
    assert resumed == before[391:]
    assert cosine_factor(plan['total_steps'], plan['total_steps']) == 0
    with pytest.raises(ValueError):
        token_normalized_ce(torch.randn(1, 2, 4), torch.full((1, 2), -100), 0)
    with pytest.raises(ValueError):
        trajectory(49999)


def test_optimizer_groups_rates_decay_and_unexpected_parameters():
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
            self.features = torch.nn.Parameter(torch.ones(2, 3, 3))
            self.lora_A = torch.nn.Parameter(torch.ones(2, 3))
    model = Fixture()
    groups = optimizer_groups(model, [model.features])
    assert {group['name']: (group['lr'], group['weight_decay']) for group in groups} == {
        'new_module': (1e-4, .01), 'lora': (2e-5, .01)}
    assert sum(len(group['params']) for group in groups) == 2
    model.base.requires_grad_(True)
    with pytest.raises(ValueError, match='Unexpected trainable'):
        optimizer_groups(model, [model.features])
