"""Teacher-forced CE and teacher KL on fixed dev samples.

Phase B's logged loss mixes the answer cross-entropy with the distillation term. This
splits them on real dev samples, and reports the frozen teacher's own CE as the
reference, so "the student is worse at the task" can be separated from "the student
predicts different but equally well".

Usage: CUDA_VISIBLE_DEVICES=7 uv run --locked python \
  scripts/experiments/prefix_ttt_p32/dev_loss.py --checkpoint <phase-B latest.pt> \
  --samples 16 --output artifacts/experiments/prefix_ttt_p32/dev-loss.json
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from prefix_ttt.config import load_config
from prefix_ttt.data_pipeline import build_dataset, load_manifest, prepare_sample
from prefix_ttt.model.bridge import load_checkpoint, load_tokenizer
from prefix_ttt.model.hybrid import LAYOUT_ANCHORS, install_prefix_ttt
from prefix_ttt.model.labels import IGNORE_INDEX
from prefix_ttt.model.trainability import install_lora, merge_lora_weights
from prefix_ttt.runtime import load_trainable


@torch.no_grad()
def token_losses(student, teacher, inputs, labels, device):
    valid = inputs['prefix_valid_mask']
    with torch.autocast('cuda', dtype=torch.bfloat16):
        student_logits = student(**inputs, use_cache=False).logits
        teacher_logits = teacher(**{k: v for k, v in inputs.items()
                                    if k != 'prefix_valid_mask'}, use_cache=False).logits
    mask = labels[..., 1:].ne(IGNORE_INDEX)
    targets = int(mask.sum())
    flat = torch.log_softmax(student_logits[..., :-1, :].float(), dim=-1)
    ce = F.nll_loss(flat.reshape(-1, flat.shape[-1]), labels[..., 1:].reshape(-1),
                    ignore_index=IGNORE_INDEX, reduction='sum')
    teacher_log = torch.log_softmax(teacher_logits[..., :-1, :].float(), dim=-1)
    kl = F.kl_div(flat, teacher_log, reduction='none', log_target=True).sum(-1)
    return float(ce), float((kl * mask).sum()), targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/p32.json')
    parser.add_argument('--manifest', default='artifacts/cpu/fixed_manifest.json')
    parser.add_argument('--checkpoint')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--split', choices=['dev', 'train'], default='dev')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--batch', type=int, default=1, help='collate this many samples per forward, like training')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output')
    args = parser.parse_args()
    device = torch.device(args.device)
    config = load_config(args.config)
    path = Path(config['data_root']) / config['model_relative_path']
    manifest, _ = load_manifest(args.manifest, config['data_root'])

    student, _ = load_checkpoint(path, dtype=torch.bfloat16)
    new_parameters = install_prefix_ttt(student, backend='fla',
                                        full_attention_layers=LAYOUT_ANCHORS['P32'])
    if args.checkpoint:
        student = install_lora(student, new_parameters=new_parameters)
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        load_trainable(student, state['trainable'])
        merge_lora_weights(student)
        student = student.get_base_model()
        del state
    student = student.to(device).eval().requires_grad_(False)
    teacher, _ = load_checkpoint(path, dtype=torch.bfloat16)
    teacher = teacher.to(device).eval().requires_grad_(False)
    tokenizer = load_tokenizer(path)
    dataset, collate = build_dataset(config, student, tokenizer, manifest)

    totals = {'student_ce': 0.0, 'student_kl': 0.0, 'teacher_ce': 0.0, 'targets': 0}
    for offset in range(0, args.samples, args.batch):
        indices = [manifest[args.split][args.offset + offset + step]
                   for step in range(min(args.batch, args.samples - offset))]
        staged, metadata = prepare_sample(student, collate([dataset[index] for index in indices]), device)
        inputs = {key: value.to(device) for key, value in staged.items()}
        labels = inputs.pop('labels')
        ce, kl, targets = token_losses(student, teacher, inputs, labels, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            teacher_logits = teacher(**{k: v for k, v in inputs.items()
                                        if k != 'prefix_valid_mask'}, use_cache=False).logits
        mask = labels[..., 1:].ne(IGNORE_INDEX)
        teacher_ce = F.cross_entropy(teacher_logits[..., :-1, :].float().reshape(-1, teacher_logits.shape[-1]),
                                     labels[..., 1:].reshape(-1), ignore_index=IGNORE_INDEX,
                                     reduction='sum')
        totals['student_ce'] += ce
        totals['student_kl'] += kl
        totals['teacher_ce'] += float(teacher_ce)
        totals['targets'] += targets
        print(json.dumps({'sample': offset, 'tokens': targets,
                          'ce': round(ce / max(targets, 1), 3),
                          'kl': round(kl / max(targets, 1), 3)}), flush=True)
        del teacher_logits
    report = {'split': args.split, 'samples': args.samples, 'target_tokens': totals['targets'],
              'student_ce': totals['student_ce'] / totals['targets'],
              'student_kl': totals['student_kl'] / totals['targets'],
              'student_loss': (totals['student_ce'] + totals['student_kl']) / totals['targets'],
              'teacher_ce': totals['teacher_ce'] / totals['targets']}
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
