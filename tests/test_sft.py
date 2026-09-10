import torch
import pytest


def test_pilot_diagnostic_gate(tmp_path):
    import json
    from prefix_ttt.sft import check_pilot_diagnostics
    from prefix_ttt.manifests import digest_file
    checkpoint = tmp_path / 'pilot.pt'
    checkpoint.write_bytes(b'fixture')
    identity = dict(layout='E2', manifest_sha256='m', config_sha256='c')
    paths = []
    for layout in ('E1', 'E2'):
        path = tmp_path / (layout + '.json')
        path.write_text(json.dumps({**identity, 'layout': layout, 'status': 'passed',
            'global_step': 391, 'samples_seen': 50048, 'checkpoint_sha256': digest_file(checkpoint)}))
        paths.append(path)
    check_pilot_diagnostics(paths, identity, checkpoint)
    check_pilot_diagnostics([paths[1]], identity, checkpoint)
    with pytest.raises(ValueError):
        check_pilot_diagnostics([paths[0]], identity, checkpoint)
    checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError):
        check_pilot_diagnostics(paths, identity, checkpoint)
    value = json.loads(paths[0].read_text())
    paths[0].write_text(json.dumps({**value, 'status': 'failed'}))
    with pytest.raises(ValueError):
        check_pilot_diagnostics(paths, identity, checkpoint)

from prefix_ttt.sft import sample_group, rng_state, restore_rng, load_trainable
from prefix_ttt.training import cosine_factor, token_normalized_ce


def test_switch_gate(tmp_path):
    import json
    from prefix_ttt.sft import check_switch_diagnostic
    path = tmp_path / 'switch.json'
    report = dict(status='passed', manifest_sha256='manifest', config_sha256='config', stage_a_sha256='A')
    path.write_text(json.dumps(report))
    check_switch_diagnostic(path, 'manifest', 'config', 'A')
    for key, value in [('status', 'failed'), ('manifest_sha256', 'other'),
                       ('config_sha256', 'other'), ('stage_a_sha256', 'other')]:
        path.write_text(json.dumps({**report, key: value}))
        with pytest.raises(ValueError):
            check_switch_diagnostic(path, 'manifest', 'config', 'A')


def test_order_partial_rank_partition():
    order = list(range(133))
    for cursor in (0, 128):
        partitions = [sample_group(order, cursor, rank, 4) for rank in range(4)]
        assert sorted(sum(partitions, [])) == order[cursor:cursor + 128]
        assert sum(map(len, partitions)) == len(set(sum(partitions, [])))


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
