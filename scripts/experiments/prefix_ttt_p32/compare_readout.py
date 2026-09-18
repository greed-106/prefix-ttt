"""Compare the trained student's per-layer readout magnitude with the teacher's.

The phase-A objective only reached ~27% of the teacher's attention-output energy,
and phase B then optimised the end-to-end loss. This check answers one question with
real dev samples: after phase B, how large is the pure Prefix-TTT readout (the input
of each o_proj) compared with the full-attention output it replaces?

  ratio << 1  -> the magnitude deficit survived end-to-end training (a capacity or
                 warm-start problem, not a scale the optimiser simply had to learn)
  ratio ~= 1  -> the magnitude is fine and any residual gap is directional

Usage: CUDA_VISIBLE_DEVICES=7 uv run --locked python \
  scripts/experiments/prefix_ttt_p32/compare_readout.py --checkpoint <phase-B latest.pt>
"""
import argparse
import json
from pathlib import Path

import torch

from prefix_ttt.config import load_config
from prefix_ttt.data_pipeline import build_dataset, load_manifest, prepare_sample
from prefix_ttt.model.bridge import load_checkpoint, load_tokenizer
from prefix_ttt.model.hybrid import LAYOUT_ANCHORS, install_prefix_ttt
from prefix_ttt.model.trainability import install_lora, merge_lora_weights
from prefix_ttt.runtime import load_trainable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/p32.json')
    parser.add_argument('--manifest', default='artifacts/cpu/fixed_manifest.json')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--stage-a-checkpoint')
    parser.add_argument('--samples', type=int, default=4)
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
    if args.stage_a_checkpoint and not args.checkpoint:
        state = torch.load(args.stage_a_checkpoint, map_location='cpu', weights_only=False)
        for index, layer in enumerate(student.model.layers):
            layer.self_attn.prefix_ttt.load_state_dict(state['features'][str(index)], strict=True)
        del state
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

    teacher_rms, student_rms = {}, {}
    for index, layer in enumerate(teacher.model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(
            lambda module, arguments, index=index: teacher_rms.__setitem__(
                index, arguments[0].detach().float()))
    for index, layer in enumerate(student.model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(
            lambda module, arguments, index=index: student_rms.__setitem__(
                index, arguments[0].detach().float()))

    totals = {}
    for offset in range(args.samples):
        staged, metadata = prepare_sample(
            student, collate([dataset[manifest['dev'][offset]]]), device)
        inputs = {key: value.to(device) for key, value in staged.items()
                  if key not in ('labels', 'prefix_valid_mask')}
        valid = metadata['valid_mask'].to(device)
        teacher_rms.clear(), student_rms.clear()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            student(**inputs, prefix_valid_mask=valid, use_cache=False)
            teacher(**inputs, use_cache=False)
        for index in teacher_rms:
            weight = valid.sum().clamp_min(1).item()
            for name, store in (('teacher', teacher_rms), ('student', student_rms)):
                values = store[index].square().mean(-1).sqrt()
                totals.setdefault(index, {})[name] = totals.setdefault(index, {}).get(name, 0.0) \
                    + float((values * valid).sum() / weight)
    report = {index: {'teacher_rms': values['teacher'] / args.samples,
                      'student_rms': values['student'] / args.samples,
                      'ratio': values['student'] / max(values['teacher'], 1e-9)}
              for index, values in totals.items()}
    ratios = [values['ratio'] for values in report.values()]
    summary = {'samples': args.samples, 'layers': report,
               'mean_ratio': sum(ratios) / len(ratios),
               'min_ratio': min(ratios), 'max_ratio': max(ratios)}
    print(json.dumps({'mean_ratio': summary['mean_ratio'], 'min_ratio': summary['min_ratio'],
                      'max_ratio': summary['max_ratio'],
                      'per_layer_ratio': {k: round(v['ratio'], 4) for k, v in report.items()}},
                     indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
