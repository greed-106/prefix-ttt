from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('lmms_eval')

from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from lmms_eval.models.simple.llava import Llava
from lmms_eval.models import get_model
from prefix_ttt import lmms_model
from prefix_ttt.model.hybrid import install_prefix_ttt
from prefix_ttt.model.trainability import install_lora, merge_lora_weights


def tiny_base():
    model = LlavaLlamaForCausalLM(LlavaConfig(vocab_size=48, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=128))
    model.config.tokenizer_model_max_length = 128
    model.get_vision_tower = lambda: SimpleNamespace(image_processor='processor')
    return model


def test_official_generation_is_inherited():
    assert get_model('prefix_ttt_llava') is lmms_model.PrefixTTTLlava
    assert lmms_model.PrefixTTTLlava.generate_until is Llava.generate_until
    assert lmms_model.PrefixTTTLlava.loglikelihood is Llava.loglikelihood


@pytest.mark.parametrize('layout', ['E0', 'E1', 'E2'])
def test_adapter_loading_and_trainable_restore(tmp_path, monkeypatch, layout):
    torch.manual_seed(42)
    template = tiny_base()
    base_state = template.state_dict()

    def load(path, *, dtype):
        assert dtype == torch.bfloat16
        model = tiny_base()
        model.load_state_dict(base_state)
        for parameter in model.parameters():
            parameter.data = parameter.data.to(dtype)
        return model, {'missing': [], 'unexpected': []}

    def install(model, *, backend):
        assert backend == 'fla'
        return install_prefix_ttt(model, backend='reference', full_attention_layers=[0])

    monkeypatch.setattr(lmms_model, 'load_checkpoint', load)
    monkeypatch.setattr(lmms_model, 'load_tokenizer', lambda path: 'tokenizer')
    monkeypatch.setattr(lmms_model, 'install_prefix_ttt', install)
    monkeypatch.setattr(lmms_model, 'Accelerator', lambda: SimpleNamespace(
        process_index=0, num_processes=1, unwrap_model=lambda model: model))
    checkpoint = None
    expected = {}
    if layout != 'E0':
        model, _ = load(tmp_path, dtype=torch.bfloat16)
        new = install(model, backend='fla') if layout == 'E2' else []
        model = install_lora(model, new_parameters=new)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.fill_(0.125)
        trainable = {name: parameter.detach().clone() for name, parameter
                     in model.named_parameters() if parameter.requires_grad}
        expected = {name.removeprefix('base_model.model.'): value
                    for name, value in trainable.items()}
        checkpoint = tmp_path / 'pilot.pt'
        torch.save(dict(stage='B', layout=layout, diagnostic_only=False,
                        trainable=trainable), checkpoint)
    adapter = lmms_model.PrefixTTTLlava(pretrained=str(tmp_path),
                                       checkpoint=checkpoint, device='cpu')
    assert adapter.model is adapter._model
    assert adapter._model.model.rotary_emb.inv_freq.dtype == torch.float32
    assert adapter.use_cache and not adapter._model.training
    assert all(not p.requires_grad for p in adapter.model.parameters())
    assert not any('lora_' in name for name, _ in adapter.model.named_modules())
    parameters = dict(adapter.model.named_parameters())
    for name, expected_value in expected.items():
        if 'lora_' in name:
            # Folded into the sibling base weight, checked in the merge test below.
            continue
        actual = parameters[name]
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected_value, rtol=0, atol=0)
    if layout != 'E0':
        # The adapter folds the checkpoint's LoRA tensors into the bf16 base weights;
        # the fold is exact up to bf16 storage precision (checked bit-exactly for FP32
        # in test_lora_merge_is_exact_and_removes_lora_branches).
        base = base_state['model.layers.0.self_attn.q_proj.weight'].to(torch.bfloat16)
        folded = base.float() + (expected['model.layers.0.self_attn.q_proj.lora_B.default.weight']
                                 @ expected['model.layers.0.self_attn.q_proj.lora_A.default.weight']
                                 ) * 2.0
        torch.testing.assert_close(parameters['model.layers.0.self_attn.q_proj.weight'].float(),
                                   folded, rtol=1e-2, atol=1e-2)


def test_lora_merge_is_exact_and_removes_lora_branches():
    torch.manual_seed(42)
    model = install_lora(tiny_base())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if 'lora_' in name:
                parameter.normal_(std=.05)
    name = 'base_model.model.model.layers.0.self_attn.q_proj'
    wrapped = model.get_submodule(name)
    expected = (wrapped.base_layer.weight.detach()
                + (wrapped.lora_B['default'].weight.detach()
                   @ wrapped.lora_A['default'].weight.detach()) * wrapped.scaling['default'])
    inputs = torch.randint(0, 48, (1, 20))
    with torch.no_grad():
        unmerged = model.get_base_model().get_model()(inputs, use_cache=False).last_hidden_state
    merged = merge_lora_weights(model)
    assert merged == model.config.num_hidden_layers * 7
    assert not any('lora_' in name for name, _ in model.named_modules())
    torch.testing.assert_close(model.get_submodule(name).weight, expected, rtol=0, atol=0)
    with torch.no_grad():
        after = model.get_base_model().get_model()(inputs, use_cache=False).last_hidden_state
    # Same arithmetic; only the summation order of base and delta differs.
    torch.testing.assert_close(after, unmerged, rtol=1e-5, atol=1e-6)
