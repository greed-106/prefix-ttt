"""Fixed eight-example Pilot diagnosis; quality is recorded, never a pass threshold."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from prefix_ttt.data_pipeline import load_manifest, build_dataset, prepare_sample
from prefix_ttt.manifests import digest_file, digest_json
from prefix_ttt.model.bridge import load_checkpoint, load_tokenizer
from prefix_ttt.model.hybrid import install_prefix_ttt
from prefix_ttt.model.trainability import install_lora
from prefix_ttt.sft import load_trainable
from prefix_ttt.switch_diagnostic import finite, generation_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.json')
    parser.add_argument('--manifest', default='artifacts/cpu/fixed_manifest.json')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('Diagnostic output exists; do not overwrite previous evidence')
    config = json.loads(Path(args.config).read_text())
    manifest, manifest_sha = load_manifest(args.manifest, config['data_root'])
    config_sha = digest_json(config)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    expected = dict(stage='B', diagnostic_only=False, global_step=391, samples_seen=50048,
                    total_steps=5182, manifest_sha256=manifest_sha, config_sha256=config_sha)
    if (checkpoint.get('layout') not in ('E1', 'E2')
            or any(checkpoint.get(key) != value for key, value in expected.items())):
        raise ValueError('Checkpoint is not the fixed formal B Pilot trajectory')
    if not torch.cuda.is_available():
        raise RuntimeError('Pilot diagnostic requires CUDA/FLA; no CPU fallback')
    report = {**expected, 'status': 'running', 'diagnostic_only': True,
              'checkpoint_sha256': digest_file(args.checkpoint), 'layout': checkpoint['layout'],
              'samples': [], 'generation': {}}
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    device = torch.device('cuda', 0)
    path = Path(config['data_root']) / config['model_relative_path']
    model, _ = load_checkpoint(path, dtype=torch.bfloat16)
    tokenizer = load_tokenizer(path)
    dataset, collate = build_dataset(config, model, tokenizer, manifest)
    new_parameters = install_prefix_ttt(model, backend='fla') if checkpoint['layout'] == 'E2' else []
    model = install_lora(model, new_parameters=new_parameters)
    load_trainable(model, checkpoint['trainable'])
    del checkpoint
    model = model.get_base_model().to(device).eval().requires_grad_(False)
    model.config.use_cache = True
    selected, counts = [], {'visual': 0, 'text': 0}
    for index in manifest['dev']:
        modality = 'visual' if dataset.list_data_dict[index].get('image') else 'text'
        if counts[modality] < 4:
            selected.append((index, modality))
            counts[modality] += 1
        if all(count == 4 for count in counts.values()):
            break
    if any(count != 4 for count in counts.values()):
        raise ValueError('Fixed dev requires four image and four text examples')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for index, modality in selected:
            values, _ = prepare_sample(model, dataset, collate, index, device)
            inputs = {key: value.to(device) for key, value in values.items() if key != 'labels'}
            mask = values['labels'][:, 1:].ne(-100).to(device)
            target = values['labels'][:, 1:].to(device)[mask]
            if target.numel() == 0:
                raise ValueError(f'Fixed dev example {index} has no assistant targets')
            logits = model(**inputs, use_cache=False).logits[:, :-1][mask].float()
            finite(logits, 'Pilot assistant logits')
            ce = F.cross_entropy(logits, target)
            finite(ce, 'Pilot assistant CE')
            report['samples'].append(dict(index=index, modality=modality,
                                          assistant_targets=target.numel(), ce=ce.item()))
            del logits
            if modality not in report['generation']:
                report['generation'][modality] = {
                    'index': index, **generation_check(model, values, tokenizer, device)}
    targets = sum(sample['assistant_targets'] for sample in report['samples'])
    report['token_weighted_ce'] = sum(s['ce'] * s['assistant_targets'] for s in report['samples']) / targets
    report['status'] = 'passed'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(output)
    print(json.dumps({'layout': report['layout'], 'token_weighted_ce': report['token_weighted_ce']}), flush=True)


if __name__ == '__main__':
    main()
