import copy

import pytest
import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from prefix_ttt.model.hybrid import PrefixTTTAttention, install_prefix_ttt
from prefix_ttt.model.trainability import install_lora, audit_parameters
from prefix_ttt.ops.local import local_attention


def tiny_model():
    torch.manual_seed(123)
    config = LlavaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=4, max_position_embeddings=128)
    return LlavaLlamaForCausalLM(config).eval()


def test_zero_gate_is_local_and_original_objects_survive():
    model = tiny_model()
    original = model.model.layers[1].self_attn
    anchor = model.model.layers[0].self_attn
    new = install_prefix_ttt(model, backend='reference', full_attention_layers=[0])
    attention = model.model.layers[1].self_attn
    assert attention.q_proj is original.q_proj and attention.o_proj is original.o_proj
    assert model.model.layers[0].self_attn is anchor
    assert all(p.dtype == torch.float32 and p.requires_grad for p in new)
    with pytest.raises(NotImplementedError, match='num_beams=1'):
        model.generate(inputs=torch.ones(1, 3, dtype=torch.long), max_new_tokens=1, num_beams=2)
    x = torch.randn(2, 35, 32)
    valid = torch.ones(2, 35, dtype=torch.bool)
    valid[1, :3] = False
    positions = (valid.long().cumsum(1) - 1).clamp_min(0)
    rope = model.model.rotary_emb(x, positions)
    q = original.q_proj(x).reshape(2, 35, 4, 8).transpose(1, 2)
    k = original.k_proj(x).reshape(2, 35, 4, 8).transpose(1, 2)
    v = original.v_proj(x).reshape(2, 35, 4, 8)
    q, k = apply_rotary_pos_emb(q, k, *rope)
    expected = original.o_proj(local_attention(q.transpose(1, 2), k.transpose(1, 2), v, valid).reshape(2, 35, 32))
    output, _ = attention(x, rope, prefix_valid_mask=valid)
    torch.testing.assert_close(output, expected)
    with pytest.raises(ValueError, match='TransformersHybridCache'):
        attention(x, rope, prefix_valid_mask=valid, use_cache=True)


def test_hybrid_lora_and_nonreentrant_gradients():
    model = tiny_model()
    new = install_prefix_ttt(model, backend='reference', full_attention_layers=[1])
    with torch.no_grad():
        model.model.layers[0].self_attn.prefix_ttt.gate_weight.normal_(std=.02)
    model = install_lora(model, rank=2, alpha=4, new_parameters=new).train()
    audit_parameters(model, new_parameters=new)
    ids = torch.randint(1, 48, (2, 35))
    valid = torch.ones_like(ids, dtype=torch.bool)
    model(ids, labels=ids, prefix_valid_mask=valid, use_cache=False).loss.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    assert any('a_phi' in n and g.abs().sum() > 0 for n, g in gradients.items())
    assert any('b_phi' in n and g.abs().sum() > 0 for n, g in gradients.items())
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model(ids, labels=ids, prefix_valid_mask=valid, use_cache=False).loss.backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-4, atol=1e-6)


def test_late_readout_gradients_reach_early_kv_and_future_is_causal():
    model = tiny_model()
    attention = PrefixTTTAttention(model.model.layers[0].self_attn, backend='reference')
    with torch.no_grad():
        attention.prefix_ttt.gate_weight.normal_(std=.05)
    x = torch.randn(1, 65, 32, requires_grad=True)
    captured = {}
    def retain(name):
        def hook(module, args, output):
            output.retain_grad()
            captured[name] = output
        return hook
    hooks = [attention.k_proj.register_forward_hook(retain('k')),
             attention.v_proj.register_forward_hook(retain('v'))]
    rope = model.model.rotary_emb(x, torch.arange(65)[None])
    output, _ = attention(x, rope)
    output[:, -1].square().sum().backward()
    assert captured['k'].grad[:, 0].abs().sum() > 0
    assert captured['v'].grad[:, 0].abs().sum() > 0
    for hook in hooks:
        hook.remove()
    changed = x.detach().clone()
    changed[:, 40:] += torch.randn_like(changed[:, 40:])
    other, _ = attention(changed, rope)
    torch.testing.assert_close(output[:, :40], other[:, :40])


@pytest.mark.parametrize('side', ['left', 'right'])
def test_model_padding_matches_independent(side):
    model = tiny_model()
    install_prefix_ttt(model, backend='reference', full_attention_layers=[1])
    with torch.no_grad():
        model.model.layers[0].self_attn.prefix_ttt.gate_weight.normal_(std=.02)
    sequences = [torch.randint(1, 48, (35,)), torch.randint(1, 48, (33,))]
    ids = torch.zeros(2, 35, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        start = 35 - len(sequence) if side == 'left' else 0
        ids[row, start:start+len(sequence)] = sequence
        valid[row, start:start+len(sequence)] = True
    positions = (valid.long().cumsum(1) - 1).clamp_min(0)
    with torch.no_grad():
        batched = model(ids, attention_mask=valid, position_ids=positions,
                        prefix_valid_mask=valid, use_cache=False).logits
        for row, sequence in enumerate(sequences):
            single = model(sequence[None], prefix_valid_mask=torch.ones(1, len(sequence), dtype=torch.bool),
                           use_cache=False).logits[0]
            torch.testing.assert_close(batched[row][valid[row]], single, rtol=1e-4, atol=1e-5)


def test_bf16_cpu_autocast_keeps_new_masters_fp32():
    model = tiny_model().to(dtype=torch.bfloat16)
    new = install_prefix_ttt(model, backend='reference', full_attention_layers=[1])
    ids = torch.randint(1, 48, (1, 35))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = model(ids, labels=ids, prefix_valid_mask=torch.ones_like(ids, dtype=torch.bool), use_cache=False)
    result.loss.backward()
    assert torch.isfinite(result.loss)
    assert all(p.dtype == torch.float32 for p in new)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in new)


def test_multimodal_training_metadata_checkpoint_and_strict_reload(tmp_path):
    from prefix_ttt.model.bridge import bridge_config
    config = bridge_config(dict(text_config=dict(vocab_size=48, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=128), vision_config=dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, image_size=8, patch_size=4)), validate_production=False)
    torch.manual_seed(99)
    model = LlavaLlamaForCausalLM(config)
    new = install_prefix_ttt(model, backend='reference', full_attention_layers=[1])
    with torch.no_grad():
        model.model.layers[0].self_attn.prefix_ttt.gate_weight.normal_(std=.02)
    config_for_reload = copy.deepcopy(model.config)
    model = install_lora(model, rank=2, alpha=4, new_parameters=new).train()
    ids = torch.randint(1, 48, (2, 35))
    ids[0, 1] = -200
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[1, -3:] = False
    labels = ids.clone()
    labels[:, :5] = -100
    labels[~valid] = -100
    images = torch.randn(2, 3, 8, 8)
    batch = dict(input_ids=ids, labels=labels, attention_mask=valid, images=images, use_cache=False)
    # No prefix_valid_mask supplied: it must come from real image expansion.
    loss = model(**batch).loss
    loss.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    assert any('a_phi' in n and g.abs().sum() > 0 for n, g in gradients.items())
    assert all(p.grad is None for n, p in model.named_parameters() if 'vision_tower' in n or 'mm_projector' in n)
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model(**batch).loss.backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-4, atol=1e-6)
    weights = tmp_path / 'tiny-hybrid.pt'
    torch.save(model.state_dict(), weights)
    reloaded = LlavaLlamaForCausalLM(config_for_reload)
    new_reloaded = install_prefix_ttt(reloaded, backend='reference', full_attention_layers=[1])
    reloaded = install_lora(reloaded, rank=2, alpha=4, new_parameters=new_reloaded)
    reloaded.load_state_dict(torch.load(weights, weights_only=True), strict=True)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**batch).logits, reloaded(**batch).logits, rtol=0, atol=0)


def test_real_optimizer_step_and_complete_resume(tmp_path):
    import random
    import numpy as np
    def build():
        model = tiny_model()
        new = install_prefix_ttt(model, backend='reference', full_attention_layers=[1])
        model = install_lora(model, rank=2, alpha=4, new_parameters=new).train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=1e-4, betas=(.9, .95), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return model, optimizer, scheduler

    def step(model, optimizer, scheduler):
        # Random input makes restoring RNG necessary for next-step equality.
        offset = random.randrange(48) + int(np.random.randint(48))
        ids = (torch.randint(0, 48, (2, 35)) + offset).remainder(48)
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        return loss.detach()

    model, optimizer, scheduler = build()
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    initial_new = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    assert torch.isfinite(step(model, optimizer, scheduler))
    assert any(not torch.equal(initial_new[n], p) for n, p in model.named_parameters() if p.requires_grad)
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
    checkpoint = tmp_path / 'resume.pt'
    numpy_state = np.random.get_state()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=torch.get_rng_state(),
                    python_rng=random.getstate(), numpy_rng=(numpy_state[0],
                    numpy_state[1].tolist(), *numpy_state[2:]),
                    global_step=1, samples_seen=2, data_cursor=2), checkpoint)
    loss_next = step(model, optimizer, scheduler)
    expected = {n: p.detach().clone() for n, p in model.named_parameters()}
    restored, optimizer2, scheduler2 = build()
    saved = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(saved['model'], strict=True)
    optimizer2.load_state_dict(saved['optimizer'])
    scheduler2.load_state_dict(saved['scheduler'])
    torch.set_rng_state(saved['rng'])
    random.setstate(saved['python_rng'])
    ns = saved['numpy_rng']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    torch.testing.assert_close(step(restored, optimizer2, scheduler2), loss_next, rtol=0, atol=0)
    assert saved['global_step'] == 1 and saved['samples_seen'] == saved['data_cursor'] == 2
    assert scheduler.state_dict() == scheduler2.state_dict()
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)


@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('anchors', [[0], [1], []])
def test_hybrid_segmented_prefill_and_decode(side, anchors):
    model = tiny_model()
    install_prefix_ttt(model, backend='reference', full_attention_layers=anchors)
    for layer in model.model.layers:
        if isinstance(layer.self_attn, PrefixTTTAttention):
            with torch.no_grad():
                layer.self_attn.prefix_ttt.gate_weight.normal_(std=.02)
    ids = torch.randint(1, 48, (2, 67))
    valid = torch.ones_like(ids, dtype=torch.bool)
    # The first 17-token segment is entirely padding for the shorter row.
    valid[1, :20 if side == 'left' else 0] = False
    if side == 'right':
        valid[1, -3:] = False
    with torch.no_grad():
        whole = model(ids, attention_mask=valid, use_cache=False).logits
        one = model(ids, attention_mask=valid, use_cache=True)
        cache = None
        pieces = []
        for start, end in ((0, 17), (17, 33), (33, 63), (63, 64), (64, 65), (65, 67)):
            output = model(ids[:, start:end], attention_mask=valid[:, start:end],
                           past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            pieces.append(output.logits)
        torch.testing.assert_close(torch.cat(pieces, 1)[valid], whole[valid], rtol=1e-4, atol=1e-5)
        assert cache.get_seq_length() == 67
        assert cache.storage.seen_tokens.tolist() == valid.sum(1).tolist()
        for index, state in cache.storage.layers.items():
            other = one.past_key_values.storage.layers[index]
            if state.state is not None:
                torch.testing.assert_close(state.state, other.state, rtol=1e-4, atol=1e-5)
                assert state.key.shape[1] == 32
            torch.testing.assert_close(state.key, other.key, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(state.value, other.value, rtol=1e-4, atol=1e-5)
        # Physical KV grows for batching, but finished effective state is fixed.
        cache.storage.mark_finished(torch.tensor([True, False]))
        before = cache.storage.layers[0].select(torch.tensor([0]))
        count = cache.storage.seen_tokens[0].clone()
        model(torch.ones(2, 1, dtype=torch.long), past_key_values=cache, use_cache=True)
        assert cache.storage.seen_tokens[0] == count
        after = cache.storage.layers[0]
        if before.state is not None:
            torch.testing.assert_close(after.state[:1], before.state, rtol=0, atol=0)
        torch.testing.assert_close(after.key[:1, :before.key.shape[1]], before.key, rtol=0, atol=0)
        cache.batch_select_indices(torch.tensor([1, 0]))
        assert cache.storage.finished.tolist() == [False, True]


def test_hybrid_greedy_images_once_and_new_token_only():
    from prefix_ttt.model.bridge import bridge_config
    config = bridge_config(dict(text_config=dict(vocab_size=48, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=128), vision_config=dict(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=4, image_size=8, patch_size=4)), validate_production=False)
    model = LlavaLlamaForCausalLM(config).eval()
    install_prefix_ttt(model, backend='reference', full_attention_layers=[1])
    with torch.no_grad():
        model.model.layers[0].self_attn.prefix_ttt.gate_weight.normal_(std=.02)
    images = torch.randn(2, 3, 8, 8)
    ids = torch.randint(1, 48, (2, 35))
    ids[:, 1] = -200
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[1, -3:] = False
    calls, lengths = [], []
    hook = model.get_vision_tower().register_forward_hook(lambda *args: calls.append(1))
    qhook = model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
    batched = model.generate(inputs=ids, images=images, attention_mask=valid,
                             max_new_tokens=3, eos_token_id=[], pad_token_id=0)
    hook.remove()
    qhook.remove()
    assert calls == [1]
    assert lengths == [38, 1, 1]
    for row in range(2):
        single = model.generate(inputs=ids[row:row+1, valid[row]], images=images[row:row+1],
                                max_new_tokens=3, eos_token_id=[], pad_token_id=0)
        torch.testing.assert_close(batched[row], single[0], rtol=0, atol=0)
    new = list(model.model.layers[0].self_attn.prefix_ttt.parameters())
    adapted = install_lora(model, rank=2, alpha=4, new_parameters=new).eval()
    adapted_output = adapted.generate(inputs=ids, images=images, attention_mask=valid,
                                      max_new_tokens=3, eos_token_id=[], pad_token_id=0)
    torch.testing.assert_close(adapted_output, batched, rtol=0, atol=0)
    # Stop on emitted EOS, not on EOS already contained in the prompt.
    with torch.no_grad():
        model.lm_head.weight.zero_()
    emitted = model.generate(inputs=torch.tensor([[0, 1, 2]]), max_new_tokens=5,
                             eos_token_id=0, pad_token_id=0)
    assert emitted.tolist() == [[0]]


def test_bf16_inference_without_external_autocast_and_context_limit():
    model = tiny_model().to(dtype=torch.bfloat16)
    new = install_prefix_ttt(model, backend='reference', full_attention_layers=[1])
    with torch.no_grad():
        model.model.layers[0].self_attn.prefix_ttt.gate_weight.normal_(std=.02)
    ids = torch.randint(1, 48, (1, 35))
    with torch.no_grad():
        forward = model(ids, use_cache=True)
        assert torch.isfinite(forward.logits).all()
        next_step = model(ids[:, :1], past_key_values=forward.past_key_values, use_cache=True)
        assert torch.isfinite(next_step.logits).all()
    model.config.max_position_embeddings = 38
    result = model.generate(inputs=ids, max_new_tokens=3, eos_token_id=[])
    assert result.shape == (1, 3)
    with pytest.raises(ValueError, match='context capacity'):
        model.generate(inputs=ids, max_new_tokens=4, eos_token_id=[])
    assert all(p.dtype == torch.float32 for p in new)
    adapted = install_lora(model, rank=2, alpha=4, new_parameters=new).eval()
    result_adapted = adapted.generate(inputs=ids, max_new_tokens=3, eos_token_id=[])
    torch.testing.assert_close(result_adapted, result, rtol=0, atol=0)
    adapted.train()
    with pytest.raises(ValueError, match='model.eval'):
        adapted.generate(inputs=ids, max_new_tokens=1)
