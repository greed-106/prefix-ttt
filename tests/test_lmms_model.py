from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('lmms_eval')

from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
from lmms_eval.models.simple.llava import Llava
from lmms_eval.models import get_model
from prefix_ttt import lmms_model
from prefix_ttt.model.hybrid import install_prefix_ttt
from prefix_ttt.model.trainability import install_lora


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
    for name, expected_value in expected.items():
        actual = dict(adapter.model.named_parameters())[name]
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected_value, rtol=0, atol=0)
