"""Full fine-tuning: the trainable whitelist and the ZeRO-1 optimizer wiring."""
import socket

import pytest
import torch

from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from prefix_ttt.model.hybrid import install_prefix_ttt
from prefix_ttt.model.trainability import (VISION_TOWER_PREFIX, audit_parameters,
                                           enable_full_finetuning, install_lora)
from prefix_ttt.training import BASE_LR, LORA_LR, NEW_MODULE_LR, optimizer_groups


def tiny_model(with_extra_modules=True):
    """A miniature LLaVA with the module names the real whitelist must match."""
    torch.manual_seed(123)
    config = LlavaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=4, max_position_embeddings=128)
    model = LlavaLlamaForCausalLM(config).eval()
    if with_extra_modules:
        model.model.vision_tower = torch.nn.Linear(8, 8)   # frozen: not in any group
        model.model.mm_projector = torch.nn.Linear(8, 32)  # trained, like the real one
    return model


def test_whitelist_trains_the_llm_and_projector_but_never_the_vision_tower():
    model = tiny_model()
    new = install_prefix_ttt(model, backend='reference', full_attention_layers=[0])
    enabled = enable_full_finetuning(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, 'the whitelist matched nothing'
    assert not any(name.startswith(VISION_TOWER_PREFIX) for name in trainable)
    assert 'model.mm_projector.weight' in trainable
    assert 'model.embed_tokens.weight' in trainable
    assert 'lm_head.weight' in trainable
    assert any(name.endswith('prefix_ttt.gate_weight') for name in trainable)
    assert enabled == sum(p.numel() for p in model.parameters() if p.requires_grad)
    # the Prefix-TTT parameters keep the FP32 dtype the recurrence needs
    assert all(p.dtype == torch.float32 for p in new)
    assert all(p.requires_grad for p in new)


def test_audit_rejects_trainable_base_unless_full_finetuning_is_declared():
    model = tiny_model()
    enable_full_finetuning(model)
    with pytest.raises(ValueError, match='Unexpected trainable base parameter'):
        audit_parameters(model)
    records = audit_parameters(model, allow_base=True)
    kinds = {record['optimizer_group'] for record in records if record['requires_grad']}
    assert kinds == {'base'}, 'a two-layer model has no Prefix-TTT modules installed'
    frozen = {record['name'] for record in records if not record['requires_grad']}
    assert frozen and all(name.startswith(VISION_TOWER_PREFIX) for name in frozen)
    # the LoRA arm still refuses a trainable base, and still labels frozen weights
    lora_model = tiny_model()
    install_lora(lora_model)
    assert all(record['optimizer_group'] == 'lora'
               for record in audit_parameters(lora_model, allow_base=False)
               if record['requires_grad'])


def test_full_finetuning_groups_use_the_base_learning_rate():
    model = tiny_model()
    new = install_prefix_ttt(model, backend='reference', full_attention_layers=[0])
    enable_full_finetuning(model)
    groups = optimizer_groups(model, new_parameters=new, allow_base=True)
    by_name = {group['name']: group for group in groups}
    assert set(by_name) == {'base', 'new_module'}
    assert by_name['base']['lr'] == BASE_LR
    assert by_name['new_module']['lr'] == NEW_MODULE_LR
    assert BASE_LR == LORA_LR and NEW_MODULE_LR > BASE_LR
    # every trainable tensor is in exactly one group
    grouped = [id(p) for group in groups for p in group['params']]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(p) for p in model.parameters() if p.requires_grad}
    with pytest.raises(ValueError, match='Unexpected trainable base parameter'):
        optimizer_groups(model, new_parameters=new, allow_base=False)


def _free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def _zero_worker(rank, world, port):
    """One gloo rank: the ZeRO-1 behaviour this project depends on."""
    import torch.distributed as dist
    from torch.distributed.optim import ZeroRedundancyOptimizer

    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}',
                            rank=rank, world_size=world)
    try:
        model = tiny_model(with_extra_modules=False)
        new = install_prefix_ttt(model, backend='reference', full_attention_layers=[0])
        enable_full_finetuning(model)
        groups = optimizer_groups(model, new_parameters=new, allow_base=True)

        # a mixed-dtype parameter set is refused outright, which is why the full
        # arm loads the model in FP32: the parameters ARE the master weights.
        mixed = [{'params': [next(model.parameters())],
                  'lr': 1e-5, 'weight_decay': 0.0},
                 {'params': [torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))],
                  'lr': 1e-5, 'weight_decay': 0.0}]
        with pytest.raises(ValueError, match='same dense type'):
            ZeroRedundancyOptimizer(mixed, optimizer_class=torch.optim.AdamW)

        optimizer = ZeroRedundancyOptimizer(groups, optimizer_class=torch.optim.AdamW,
                                            betas=(0.9, 0.95), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5 ** step)
        parameters = [p for p in model.parameters() if p.requires_grad]
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            generator = torch.Generator().manual_seed(step)
            # input_ids, not inputs_embeds: every parameter must receive a gradient,
            # which is also what sft.py's missing-gradient check enforces.
            ids = torch.randint(0, 48, (2, 5), generator=generator)
            model(input_ids=ids, use_cache=False).logits.square().mean().backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            # the schedule must reach the sharded inner optimizer
            assert all(group['lr'] == BASE_LR * 0.5 ** (step + 1)
                       for group in optimizer.param_groups if group['name'] == 'base')

        # ZeRO broadcasts every rank's shard back, so all ranks must agree exactly
        for name, parameter in model.named_parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            assert torch.equal(parameter.detach(), reference), name

        # the local shard round-trips, which is how the full arm checkpoints Adam
        before = optimizer.optim.state_dict()
        path = f'/tmp/prefix-ttt-zero-shard-{rank}.pt'
        torch.save(before, path)
        optimizer.optim.load_state_dict(torch.load(path, weights_only=False))
        after = optimizer.optim.state_dict()
        assert [len(state) for state in after['state'].values()] == \
               [len(state) for state in before['state'].values()]

        # and the shards partition the parameter set instead of overlapping
        owned = sum(len(group['params']) for group in optimizer.optim.param_groups)
        assert owned == len(before['state']), 'every owned parameter keeps Adam state'
        counts = torch.tensor([owned])
        gathered = [torch.zeros_like(counts) for _ in range(world)]
        dist.all_gather(gathered, counts)
        assert int(sum(int(count) for count in gathered)) == len(parameters)
    finally:
        dist.destroy_process_group()


def test_zero_redundancy_optimizer_shards_agrees_and_round_trips():
    world = 2
    torch.multiprocessing.spawn(_zero_worker, args=(world, _free_port()),
                                nprocs=world, join=True)


def test_master_weight_adamw_matches_torch_adamw_on_fp32_parameters():
    """The master-weight update must be exactly torch.optim.AdamW's."""
    from prefix_ttt.optim import MasterWeightAdamW

    torch.manual_seed(7)
    ours = torch.nn.Parameter((torch.randn(64, 32) * 0.02).float(), requires_grad=True)
    reference = torch.nn.Parameter(ours.detach().clone(), requires_grad=True)
    state = {'lr': 2e-5, 'betas': (0.9, 0.95), 'eps': 1e-8, 'weight_decay': 0.01}
    mine = MasterWeightAdamW([ours], **state)
    theirs = torch.optim.AdamW([reference], **state)
    for step in range(20):
        grad = torch.randn(64, 32, generator=torch.Generator().manual_seed(100 + step))
        ours.grad = grad.clone()
        reference.grad = grad.clone()
        mine.step()
        theirs.step()
    assert torch.equal(ours.detach(), reference.detach())
    assert torch.equal(mine.state[ours]['master'], ours.detach())


def test_master_weight_adamw_accumulates_into_a_bf16_parameter():
    """BF16 storage alone would drop these updates; the FP32 master must not."""
    from prefix_ttt.optim import MasterWeightAdamW

    torch.manual_seed(11)
    store = torch.nn.Parameter((torch.randn(4096) * 0.02).to(torch.bfloat16), requires_grad=True)
    naive = torch.nn.Parameter(store.detach().clone(), requires_grad=True)
    optimized = MasterWeightAdamW([store], lr=2e-5, betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=0.01)
    plain = torch.optim.AdamW([naive], lr=2e-5, betas=(0.9, 0.95), eps=1e-8,
                              weight_decay=0.01)
    before = store.detach().float().clone()
    for step in range(10):
        grad = torch.randn(4096, generator=torch.Generator().manual_seed(200 + step)) * 1e-3
        store.grad = grad.to(torch.bfloat16)
        naive.grad = grad.to(torch.bfloat16)
        optimized.step()
        plain.step()
    # The BF16 parameter is only a projection of the master, so compare what
    # actually accumulates: the FP32 master against the naive BF16 weight.
    master = optimized.state[store]['master']
    moved = (master - before).abs()
    ignored = (naive.detach().float() - before).abs()
    # Measured: master mean|d| 6.27e-05 with nothing frozen; naive BF16 mean|d|
    # 1.91e-05 with 70.9% of the elements never leaving their starting value.
    assert (moved == 0).float().mean() < 0.01, 'the master must move nearly every element'
    assert (ignored == 0).float().mean() > 0.5, 'BF16 storage alone freezes most elements'
    assert moved.sum() > 2 * ignored.sum(), 'the master must recover most of the update'
