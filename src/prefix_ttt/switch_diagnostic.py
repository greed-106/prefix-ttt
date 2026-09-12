"""Bounded fixed-dev diagnosis of the completed A -> A9-T23 model switch."""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from prefix_ttt.config import load_config
from prefix_ttt.data_pipeline import load_manifest, build_dataset, prepare_sample
from prefix_ttt.gpu_regression import compare
from prefix_ttt.digests import digest_file, digest_json
from prefix_ttt.model.bridge import load_checkpoint, load_tokenizer
from prefix_ttt.model.hybrid import install_prefix_ttt


def finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f'Nonfinite {name}')


@torch.no_grad()
def generation_check(model, values, tokenizer, device):
    """Two greedy steps from the first assistant target; image expanded once."""
    first_target = int(values['labels'][0, 1:].ne(-100).nonzero()[0]) + 1
    if 'inputs_embeds' in values:
        embeds = values['inputs_embeds'][:, :first_target].to(device)
    else:
        embeds = model.get_input_embeddings()(values['input_ids'][:, :first_target].to(device))
    output = model(inputs_embeds=embeds, use_cache=True)
    finite(output.logits, 'prefill logits')
    cache = output.past_key_values
    token = output.logits[:, -1].argmax(-1, keepdim=True)
    tokens, checks = [], []
    for step in range(min(2, 2048 - first_target)):
        tokens.append(token.item())
        embeds = torch.cat((embeds, model.get_input_embeddings()(token)), dim=1)
        cached = model(input_ids=token, past_key_values=cache, use_cache=True)
        cache = cached.past_key_values
        whole = model(inputs_embeds=embeds, use_cache=False)
        check = compare(cached.logits[:, -1], whole.logits[:, -1], max_abs=0.25, relative=0.02)
        checks.append(check)
        if not check['passed']:
            raise AssertionError(f'Cache/full decode step {step}: {check}')
        token = cached.logits[:, -1].argmax(-1, keepdim=True)
    return {'token_ids': tokens, 'text': tokenizer.decode(tokens), 'cache_full_checks': checks,
            'prompt_expanded_tokens': first_target}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.json')
    parser.add_argument('--manifest', default='artifacts/cpu/fixed_manifest.json')
    parser.add_argument('--stage-a-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Diagnostic output exists; do not overwrite previous evidence')
    config = load_config(args.config)
    manifest, manifest_sha = load_manifest(args.manifest, config['data_root'])
    config_sha = digest_json(config)
    stage_a = torch.load(args.stage_a_checkpoint, map_location='cpu', weights_only=False)
    expected = {'stage': 'A', 'complete': True, 'diagnostic_only': False,
                'manifest_sha256': manifest_sha, 'config_sha256': config_sha,
                'samples_seen': len(manifest['A']), 'global_step': math.ceil(len(manifest['A']) / 128)}
    if any(stage_a.get(key) != value for key, value in expected.items()):
        raise ValueError('A checkpoint identity/completion differs from fixed experiment')
    if not torch.cuda.is_available():
        raise RuntimeError('Switch diagnostic requires CUDA/FLA; no CPU fallback')
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    device = torch.device('cuda', 0)
    path = Path(config['data_root']) / config['model_relative_path']
    model, _ = load_checkpoint(path, dtype=torch.bfloat16)
    model.requires_grad_(False).eval().to(device)
    tokenizer = load_tokenizer(path)
    dataset, collate = build_dataset(config, model, tokenizer, manifest)
    # First four examples per modality in the immutable dev order, at most eight.
    selected, counts = [], {'visual': 0, 'text': 0}
    for index in manifest['dev']:
        modality = 'visual' if dataset.list_data_dict[index].get('image') else 'text'
        if counts[modality] < 4:
            selected.append((index, modality))
            counts[modality] += 1
        if all(count == 4 for count in counts.values()):
            break
    if not all(counts.values()):
        raise ValueError('Fixed dev must contain both image and text examples')
    report = {'status': 'running', 'diagnostic_only': True, 'stage_a_sha256': digest_file(args.stage_a_checkpoint),
              'manifest_sha256': manifest_sha, 'config_sha256': config_sha,
              'backend': 'fla', 'lora': False, 'samples': [], 'generation': {}}
    prepared, teacher = {}, {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for index, modality in selected:
            values, _ = prepare_sample(model, collate([dataset[index]]), device)
            prepared[index] = values
            inputs = {key: value.to(device) for key, value in values.items() if key != 'labels'}
            logits = model(**inputs, use_cache=False).logits[:, :-1]
            mask = values['labels'][:, 1:].ne(-100).to(device)
            target = values['labels'][:, 1:].to(device)[mask]
            logits = logits[mask].float()
            finite(logits, 'E0 assistant logits')
            teacher[index] = logits.cpu()
            report['samples'].append({'index': index, 'modality': modality, 'assistant_targets': target.numel(),
                                      'e0_ce': F.cross_entropy(logits, target).item()})
            if modality not in report['generation']:
                report['generation'][modality] = {'index': index, 'E0': generation_check(model, values, tokenizer, device)}
        install_prefix_ttt(model, backend='fla')
        for index, layer in enumerate(model.get_model().layers):
            if hasattr(layer.self_attn, 'prefix_ttt'):
                layer.self_attn.prefix_ttt.load_state_dict(stage_a['features'][str(index)], strict=True)
        del stage_a
        model.requires_grad_(False).eval()
        for sample in report['samples']:
            index, modality = sample['index'], sample['modality']
            values = prepared[index]
            inputs = {key: value.to(device) for key, value in values.items() if key != 'labels'}
            mask = values['labels'][:, 1:].ne(-100).to(device)
            target = values['labels'][:, 1:].to(device)[mask]
            logits = model(**inputs, use_cache=False).logits[:, :-1][mask].float()
            finite(logits, 'A9-T23 assistant logits')
            teacher_logp = teacher.pop(index).to(device).log_softmax(-1)
            logp = logits.log_softmax(-1)
            kl = (teacher_logp.exp() * (teacher_logp - logp)).sum(-1).mean()
            finite(kl, 'teacher KL')
            sample.update(switched_ce=F.cross_entropy(logits, target).item(), teacher_kl=kl.item())
            if report['generation'][modality]['index'] == index:
                report['generation'][modality]['A9_T23'] = generation_check(model, values, tokenizer, device)
    targets = sum(sample['assistant_targets'] for sample in report['samples'])
    report['token_weighted'] = {key: sum(s[key] * s['assistant_targets'] for s in report['samples']) / targets
                              for key in ('e0_ce', 'switched_ce', 'teacher_kl')}
    report['status'] = 'passed'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(output)
    print(json.dumps(report['token_weighted']), flush=True)


if __name__ == '__main__':
    main()
