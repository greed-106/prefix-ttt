"""Bounded offline checkpoint/greedy-cache regression, not quality evaluation."""
import argparse
import gc
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor, LlavaForConditionalGeneration

from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token
from prefix_ttt.config import load_config
from prefix_ttt.model.bridge import load_checkpoint, load_tokenizer
from prefix_ttt.model.hybrid import install_prefix_ttt
from prefix_ttt.model.trainability import install_lora


def compare(actual, expected, *, max_abs, relative):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    if actual.shape != expected.shape:
        raise AssertionError(f'Shape mismatch: {actual.shape} != {expected.shape}')
    delta = actual - expected
    absolute = delta.abs().max().item()
    rel = (delta.norm() / expected.norm().clamp_min(1e-12)).item()
    passed = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()
                  and absolute <= max_abs and rel <= relative)
    return {'passed': passed, 'max_absolute': absolute, 'relative_frobenius': rel,
            'limits': {'max_absolute': max_abs, 'relative_frobenius': relative}}


def expanded_ids(ids, image_id, patches):
    pieces = [torch.full((patches,), image_id, dtype=ids.dtype, device=ids.device)
              if value.item() == -200 else value.reshape(1) for value in ids[0]]
    return torch.cat(pieces)[None]


@torch.no_grad()
def cached_check(model, ids, pixels, record, prefix):
    prompt = model(ids, images=pixels, use_cache=True)
    cache = prompt.past_key_values
    token = prompt.logits[:, -1].argmax(-1, keepdim=True)
    sequence = ids
    for step in range(2):
        sequence = torch.cat((sequence, token), 1)
        cached = model(token, past_key_values=cache, use_cache=True)
        cache = cached.past_key_values
        whole = model(sequence, images=pixels, use_cache=False)
        record(f'{prefix}_decode_{step}', cached.logits[:, -1], whole.logits[:, -1],
               max_abs=0.25, relative=0.02)
        token = cached.logits[:, -1].argmax(-1, keepdim=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.json')
    parser.add_argument('--image', required=True, help='Existing local RGB-readable image')
    parser.add_argument('--output', required=True)
    parser.add_argument('--skip-hybrid', action='store_true')
    args = parser.parse_args()
    report = {'status': 'running', 'diagnostic_only': True, 'checks': {}, 'args': vars(args)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2) + '\n')

    def record(name, actual, expected, max_abs=0.125, relative=0.005):
        result = compare(actual, expected, max_abs=max_abs, relative=relative)
        report['checks'][name] = result
        save()
        if not result['passed']:
            raise AssertionError(f'{name}: {result}')

    save()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('GPU regression requires a CUDA device')
        torch.cuda.set_device(0)
        torch.manual_seed(42)
        device = torch.device('cuda', 0)
        report['device'] = torch.cuda.get_device_name(device)
        config = load_config(args.config)
        path = Path(config['data_root']) / config['model_relative_path']
        tokenizer = load_tokenizer(path)
        processor = CLIPImageProcessor.from_pretrained(path, local_files_only=True)
        with Image.open(args.image) as source:
            pixel_cpu = processor(source.convert('RGB'), return_tensors='pt')['pixel_values']
        pixels = pixel_cpu.to(device=device, dtype=torch.bfloat16)
        cases = {}
        for image in (False, True):
            conv = conv_templates['v1'].copy()
            question = '<image>\nDescribe what you see.' if image else 'What is two plus two?'
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            ids = (tokenizer_image_token(prompt, tokenizer, return_tensors='pt') if image
                   else torch.tensor(tokenizer(prompt).input_ids))[None].to(device)
            cases['image' if image else 'text'] = (ids, pixels if image else None)
        report['prompts'] = {name: ids.cpu().tolist() for name, (ids, _) in cases.items()}
        with torch.no_grad():
            hf = LlavaForConditionalGeneration.from_pretrained(path, local_files_only=True,
                torch_dtype=torch.bfloat16).eval().to(device)
            report['hf_backends'] = {'vision': hf.vision_tower.config._attn_implementation,
                                     'language': hf.language_model.config._attn_implementation}
            features = hf.vision_tower(pixels, output_hidden_states=True).hidden_states[-2][:, 1:]
            projected = hf.multi_modal_projector(features)
            patch_count = projected.shape[1]
            report['image_patch_count'] = patch_count
            expected_projected = projected.cpu()
            expected = {}
            for name, (ids, image_pixels) in cases.items():
                hf_ids = expanded_ids(ids, hf.config.image_token_index, patch_count)
                embeds = hf.get_input_embeddings()(hf_ids)
                if image_pixels is not None:
                    embeds = embeds.masked_scatter(
                        (hf_ids == hf.config.image_token_index)[..., None].expand_as(embeds), projected)
                expected[name] = {'embeds': embeds.cpu(),
                    'logits': hf(input_ids=hf_ids, pixel_values=image_pixels, use_cache=False).logits.cpu()}
            del hf, features, projected, embeds
            gc.collect()
            torch.cuda.empty_cache()
            model, report['bridge'] = load_checkpoint(path, dtype=torch.bfloat16)
            model = model.eval().to(device)
            report['bridge_backends'] = {'vision': model.get_vision_tower().config._attn_implementation,
                                         'language': model.config._attn_implementation}
            if report['hf_backends'] != report['bridge_backends']:
                raise AssertionError('HF/bridge attention backend mismatch')
            record('projector', model.encode_images(pixels), expected_projected)
            for name, (ids, image_pixels) in cases.items():
                if image_pixels is None:
                    embeds = model.get_input_embeddings()(ids)
                else:
                    embeds = model.prepare_inputs_labels_for_multimodal(
                        ids, None, None, None, None, image_pixels)[4]
                record(f'{name}_expanded_embeddings', embeds, expected[name]['embeds'])
                record(f'{name}_E0_logits', model(ids, images=image_pixels, use_cache=False).logits,
                       expected[name]['logits'])
                cached_check(model, ids, image_pixels, record, f'{name}_E0')
            model = install_lora(model).eval()
            for name, (ids, image_pixels) in cases.items():
                record(f'{name}_zero_lora_logits', model(ids, images=image_pixels, use_cache=False).logits,
                       expected[name]['logits'])
            if not args.skip_hybrid:
                base = model.get_base_model()
                install_prefix_ttt(base, backend='fla')
                for layer in base.get_model().layers:
                    if hasattr(layer.self_attn, 'prefix_ttt'):
                        layer.self_attn.prefix_ttt.gate_weight.normal_(std=0.01 / 64)
                report['hybrid_fixture'] = {'backend': 'fla', 'gate_std': 0.01 / 64,
                                            'trained_checkpoint': False}
                for name, (ids, image_pixels) in cases.items():
                    cached_check(base, ids, image_pixels, record, f'{name}_E2_nonzero_gate')
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
