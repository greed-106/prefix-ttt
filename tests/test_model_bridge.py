import pytest
import torch
from transformers import LlavaConfig as HFConfig, LlavaForConditionalGeneration

from prefix_ttt.model.bridge import bridge_config, load_checkpoint, map_checkpoint_key, strict_assign_shards
from prefix_ttt.model.labels import LabelPreprocessingError, classify_supervision
from prefix_ttt.model.trainability import install_lora, audit_parameters


def tiny_config():
    return HFConfig(text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128), vision_config=dict(hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        image_size=8, patch_size=4), image_token_index=63, pad_token_id=0)


@pytest.fixture
def models(tmp_path):
    from transformers import CLIPImageProcessor
    torch.manual_seed(7)
    original = LlavaForConditionalGeneration(tiny_config()).eval()
    original.save_pretrained(tmp_path)
    CLIPImageProcessor(size={'shortest_edge': 8}, crop_size={'height': 8, 'width': 8}).save_pretrained(tmp_path)
    bridged, report = load_checkpoint(tmp_path, dtype=torch.float32, validate_production=False)
    return original, bridged.eval(), report


def test_strict_bridge_text_and_image(models):
    original, bridged, report = models
    assert not report['missing'] and not report['unexpected']
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, bridged.state_dict()[map_checkpoint_key(name)], rtol=0, atol=0)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        torch.testing.assert_close(original(input_ids=ids).logits, bridged(ids).logits)
        pixels = torch.randn(1, 3, 8, 8)
        features = original.vision_tower(pixels, output_hidden_states=True).hidden_states[-2][:, 1:]
        expected = original.multi_modal_projector(features)
        torch.testing.assert_close(bridged.encode_images(pixels), expected)
        hf_ids = torch.tensor([[1, 63, 63, 63, 63, 2, 3]])
        llava_ids = torch.tensor([[1, -200, 2, 3]])
        torch.testing.assert_close(original(input_ids=hf_ids, pixel_values=pixels).logits,
                                   bridged(llava_ids, images=pixels).logits)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_original_expansion_metadata(models, side):
    _, model, _ = models
    model.config.tokenizer_padding_side = side
    ids = torch.tensor([[1, -200, 2, 3], [1, 4, 0, 0]])
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    labels = torch.tensor([[-100, -100, 2, 3], [-100, 4, -100, -100]])
    result, meta = model.prepare_inputs_labels_for_multimodal(ids, None, valid, None,
        labels, torch.randn(2, 3, 8, 8), return_metadata=True)
    assert meta['image_token_mask'].sum(1).tolist() == [4, 0]
    assert meta['valid_mask'].sum(1).tolist() == [7, 2]
    assert (result[-1][meta['image_token_mask']] == -100).all()
    assert meta['position_ids'][1][meta['valid_mask'][1]].tolist() == [0, 1]
    model.config.tokenizer_model_max_length = 3
    _, meta = model.prepare_inputs_labels_for_multimodal(ids, None, valid, None,
        labels, torch.randn(2, 3, 8, 8), return_metadata=True)
    assert meta['image_truncated'] == [True, False]


def test_lora_zero_increment_and_checkpoint_gradients(models):
    _, model, _ = models
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        expected = model(ids, use_cache=False).logits
    model = install_lora(model, rank=2, alpha=4)
    torch.testing.assert_close(model(ids, use_cache=False).logits, expected)
    records = audit_parameters(model)
    assert all(r['optimizer_group'] == 'lora' for r in records if r['requires_grad'])
    model.train()
    model(ids, labels=ids, use_cache=False).loss.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    assert gradients and any(x.abs().sum() > 0 for x in gradients.values())
    model.zero_grad(set_to_none=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model(ids, labels=ids, use_cache=False).loss.backward()
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name])


def test_text_generate_and_cached_logits(models):
    _, model, _ = models
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        whole = model(ids, use_cache=False).logits[:, -1]
        prefix = model(ids[:, :-1], use_cache=True)
        last = model(ids[:, -1:], past_key_values=prefix.past_key_values, use_cache=True).logits[:, -1]
        torch.testing.assert_close(last, whole)
        output = model.generate(inputs=ids, max_new_tokens=2, do_sample=False,
                                pad_token_id=0, eos_token_id=None)
        assert output.shape == (1, 2)


def test_schema_and_supervision_fail_closed():
    with pytest.raises(ValueError, match='architecture'):
        bridge_config(tiny_config().to_dict())
    with pytest.raises(ValueError, match='Unmapped'):
        map_checkpoint_key('unknown.weight')
    with pytest.raises(LabelPreprocessingError):
        classify_supervision(has_assistant_content=True, targets_before_truncation=0, targets_after_truncation=0)
    assert classify_supervision(has_assistant_content=True, targets_before_truncation=5,
        targets_after_truncation=0) == 'protocol_answer_truncated'


def test_strict_bridge_rejects_missing_and_bad_shapes(models, tmp_path):
    from safetensors.torch import save_file
    _, model, _ = models
    shard = tmp_path / 'bad.safetensors'
    save_file({'language_model.lm_head.weight': torch.zeros(1, 1)}, shard)
    with pytest.raises(ValueError, match='Shape mismatch'):
        strict_assign_shards(model, [shard])
    save_file({'language_model.lm_head.weight': model.lm_head.weight.detach().clone()}, shard)
    with pytest.raises(ValueError, match='Missing checkpoint'):
        strict_assign_shards(model, [shard])


def test_local_tokenizer_original_v1_supervision():
    from pathlib import Path
    from transformers import LlamaTokenizer
    from llava import conversation as conversation_lib
    from llava.train.train import preprocess_v1
    path = Path('data/llava-v1.5-assets-v1/models/llava-1.5-7b-hf-b234b804b114d9e37bb655e11cbbb5f5e971b7a9')
    if not path.exists():
        pytest.skip('Local checkpoint tokenizer not available')
    tokenizer = LlamaTokenizer.from_pretrained(path, local_files_only=True, model_max_length=2048)
    previous = conversation_lib.default_conversation
    conversation_lib.default_conversation = conversation_lib.conv_templates['v1']
    try:
        for image in (False, True):
            source = [[{'from': 'human', 'value': ('<image>\n' if image else '') + 'What is two plus two?'},
                       {'from': 'gpt', 'value': 'Four.'},
                       {'from': 'human', 'value': 'Why?'},
                       {'from': 'gpt', 'value': 'Adding two and two gives four.'}]]
            result = preprocess_v1(source, tokenizer, has_image=image)
            target = result['labels'][0]
            assert (target[1:] != -100).any()
            supervised = tokenizer.decode(target[target != -100])
            assert 'Four' in supervised and 'Adding' in supervised
            assert 'What' not in supervised and 'Why' not in supervised
    finally:
        conversation_lib.default_conversation = previous


@pytest.mark.parametrize('answer', ['中文回答。', '  Four.', ': yes', 'USER: literal </s> text', '🙂 fine'])
def test_v1_boundary_content(answer):
    from pathlib import Path
    from prefix_ttt.model.bridge import load_tokenizer
    from prefix_ttt.model.supervision import v1_targets
    from llava.conversation import conv_templates
    path = Path('data/llava-v1.5-assets-v1/models/llava-1.5-7b-hf-b234b804b114d9e37bb655e11cbbb5f5e971b7a9')
    if not path.exists():
        pytest.skip('Local tokenizer unavailable')
    tokenizer = load_tokenizer(path)
    conv = conv_templates['v1'].copy()
    conv.append_message(conv.roles[0], 'Question?')
    conv.append_message(conv.roles[1], answer)
    ids = torch.tensor(tokenizer(conv.get_prompt()).input_ids)
    labels = v1_targets(conv, tokenizer, ids, has_image=False)
    decoded = tokenizer.decode(labels[labels != -100])
    assert 'Question' not in decoded
    assert (labels[1:] != -100).any()


def test_bf16_bridge_preserves_rope_buffers_and_long_positions(models, tmp_path):
    original, _, _ = models
    original.save_pretrained(tmp_path)
    hf = LlavaForConditionalGeneration.from_pretrained(tmp_path, torch_dtype=torch.bfloat16).eval()
    bridge, _ = load_checkpoint(tmp_path, dtype=torch.bfloat16, validate_production=False)
    hf_rope = hf.language_model.model.rotary_emb
    bridge_rope = bridge.model.rotary_emb
    assert hf_rope.inv_freq.dtype == torch.float32
    torch.testing.assert_close(bridge_rope.inv_freq, hf_rope.inv_freq, rtol=0, atol=0)
    torch.testing.assert_close(bridge_rope.original_inv_freq, hf_rope.original_inv_freq, rtol=0, atol=0)
    ids = torch.tensor([[1, 2, 3, 4]])
    positions = torch.tensor([[0, 512, 1024, 2047]])
    with torch.no_grad():
        torch.testing.assert_close(hf(input_ids=ids, position_ids=positions).logits,
                                   bridge(ids, position_ids=positions).logits, rtol=0, atol=0)
        pixels = torch.randn(1, 3, 8, 8, dtype=torch.bfloat16)
        image_ids = torch.tensor([[1, -200, 2, 3]])
        prepared = bridge.prepare_inputs_labels_for_multimodal(image_ids, None, None,
                                                               None, None, pixels)
        hf_ids = torch.tensor([[1, 63, 63, 63, 63, 2, 3]])
        image_positions = torch.tensor([[0, 128, 256, 512, 1024, 1536, 2047]])
        torch.testing.assert_close(hf(input_ids=hf_ids, pixel_values=pixels,
                                     position_ids=image_positions).logits,
            bridge(inputs_embeds=prepared[4], position_ids=image_positions).logits, rtol=0, atol=0)
