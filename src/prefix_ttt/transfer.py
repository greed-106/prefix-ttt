"""Independent pre-o_proj residual transfer; teacher continuation stays unchanged."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from prefix_ttt.data_pipeline import load_manifest, build_dataset, prepare_sample
from prefix_ttt.manifests import digest_json
from prefix_ttt.model.bridge import load_checkpoint, load_tokenizer
from prefix_ttt.ops.features import FeatureReadout
from prefix_ttt.ops.fla import fla_prefix
from prefix_ttt.ops.local import local_attention
from prefix_ttt.ops.reference import chunk_prefix
from prefix_ttt.sft import rng_state, restore_rng, sample_group
from prefix_ttt.training import accumulation_steps, cosine_factor, residual_transfer_loss


def check_baselines(paths):
    """Require complete native lmms outputs, without recomputing any metrics."""
    required = {
        'mme': (2374, ('mme_perception_score', 'mme_cognition_score')),
        'pope': (9000, ('pope_accuracy', 'pope_precision', 'pope_recall',
                        'pope_f1_score', 'pope_yes_ratio')),
        'gqa': (12578, ('exact_match',)),
    }
    seen = set()
    for path in paths:
        files = list(Path(path).rglob('*_results.json'))
        if len(files) != 1:
            raise ValueError(f'Expected exactly one native lmms result in {path}')
        result_path = files[0]
        summary = json.loads(result_path.read_text())
        tasks = set(summary.get('results', {}))
        if len(tasks) != 1 or not tasks <= required.keys() or tasks & seen:
            raise ValueError('E0 baseline requires distinct MME/POPE/GQA results')
        task = next(iter(tasks))
        count, metrics = required[task]
        if ('limit' not in summary.get('config', {}) or summary['config']['limit'] is not None
                or summary.get('n-samples', {}).get(task) != {'original': count, 'effective': count}):
            raise ValueError(f'Incomplete or diagnostic E0 baseline: {task}')
        for metric in metrics:
            value = summary['results'][task].get(f'{metric},none')
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f'Missing or nonfinite native metric: {task}/{metric}')
        sample_path = result_path.with_name(result_path.name.removesuffix('_results.json')
                                            + f'_samples_{task}.jsonl')
        if not sample_path.is_file():
            raise ValueError(f'Missing native sample log: {sample_path}')
        with sample_path.open() as stream:
            ids = [json.loads(line).get('doc_id') for line in stream]
        if len(ids) != count or any(type(i) is not int for i in ids) or set(ids) != set(range(count)):
            raise ValueError(f'Incomplete or duplicate native sample log: {task}')
        seen.add(task)
    if seen != required.keys():
        raise ValueError('E0 baseline requires MME/POPE/GQA results')


class TransferHooks:
    """Observe original projections; backward one independent branch per layer."""
    def __init__(self, teacher, branches, backend='fla'):
        self.branches, self.backend = branches, backend
        self.handles, self.captured, self.diagnostics = [], {}, {}
        self.valid = self.visual = None
        self.denominator = None
        for key in branches:
            attention = teacher.get_model().layers[int(key)].self_attn
            self.handles.append(attention.register_forward_pre_hook(self._capture_input(key), with_kwargs=True))
            for name in ('q_proj', 'k_proj', 'v_proj'):
                self.handles.append(getattr(attention, name).register_forward_hook(self._capture_projection(key, name)))
            self.handles.append(attention.o_proj.register_forward_pre_hook(self._transfer(key, attention)))

    def _capture_input(self, key):
        def capture(module, args, kwargs):
            self.captured[key] = {'x': kwargs.get('hidden_states', args[0] if args else None),
                                  'rope': kwargs['position_embeddings']}
        return capture

    def _capture_projection(self, key, name):
        def capture(module, args, output):
            self.captured[key][name] = output
        return capture

    def _transfer(self, key, attention):
        def transfer(module, args):
            item = self.captured.pop(key)
            x = item['x']
            b, t, _ = x.shape
            heads, dim = attention.config.num_attention_heads, attention.head_dim
            shape = (b, t, heads, dim)
            q = item['q_proj'].view(shape).transpose(1, 2)
            k = item['k_proj'].view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *item['rope'])
            q, k = q.transpose(1, 2), k.transpose(1, 2)
            v = item['v_proj'].view(shape)
            full = args[0].view(shape).detach()
            local = local_attention(q, k, v, self.valid)
            branch = self.branches[key]
            with torch.enable_grad():
                qf, kf = branch.features(q.detach()), branch.features(k.detach())
                if self.backend == 'reference':
                    memory, state = chunk_prefix(qf, kf, v.detach(), valid=self.valid)
                else:
                    memory, state = fla_prefix(qf, kf, v.detach(), valid=self.valid, need_final_state=True)
                readout = branch.readout(x.detach(), memory)
                loss = residual_transfer_loss(readout, full, local, self.visual, self.valid).sum() / self.denominator
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite phase-A loss')
                loss.backward()
            # Device-side sums; no per-token/layer CPU logging synchronization.
            with torch.no_grad():
                errors = (readout.float() - (full - local).float()).square().mean((-1, -2))
                energy = full.float().square().mean((-1, -2))
                values = [loss.detach() * self.denominator]
                for mask in (self.visual & self.valid, ~self.visual & self.valid):
                    count = mask.sum().clamp_min(1)
                    raw = (errors * mask).sum() / count
                    normalized = raw / ((energy * mask).sum() / count + 1e-6)
                    values.extend([raw, normalized, mask.any().float()])
                gate = torch.nn.functional.linear(x, branch.gate_weight).float()
                values.extend([gate.square().mean().sqrt(), state.detach().float().square().mean().sqrt()])
                value = torch.stack(values)
                self.diagnostics[key] = self.diagnostics.get(key, torch.zeros_like(value)) + value
            # None is essential: original full attention goes through original o_proj.
        return transfer

    def close(self):
        for handle in self.handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.json')
    parser.add_argument('--manifest', default='artifacts/cpu/fixed_manifest.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-baseline', nargs=3, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--save-every', type=int, default=25)
    args = parser.parse_args()
    if args.save_every <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        parser.error('Step limits must be positive')
    check_baselines(args.require_baseline)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    accumulation_steps(world, 1)
    if not torch.cuda.is_available():
        raise RuntimeError('Phase A requires validated CUDA/FLA')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    hooks = None
    try:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        config = json.loads(Path(args.config).read_text())
        manifest, sha = load_manifest(args.manifest)
        total_steps = math.ceil(len(manifest['A']) / 128)
        identity = dict(stage='A', manifest_sha256=sha, config_sha256=digest_json(config),
                        world_size=world, total_steps=total_steps, diagnostic_only=args.max_steps is not None)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if not args.resume and (output / 'latest.pt').exists():
            raise ValueError('Existing phase A requires explicit resume')
        path = Path(config['data_root']) / config['model_relative_path']
        teacher, _ = load_checkpoint(path, dtype=torch.bfloat16)
        teacher.requires_grad_(False).eval().to(device)
        tokenizer = load_tokenizer(path)
        dataset, collate = build_dataset(config, teacher, tokenizer, manifest)
        branches = nn.ModuleDict({str(index): FeatureReadout(seed=42 + index)
                                  for index in config['candidate_ttt_layers']}).to(device)
        optimizer = torch.optim.AdamW(branches.parameters(), lr=1e-4,
                                     betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_factor(step, total_steps))
        cursor = step = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            if any(checkpoint.get(key) != value for key, value in identity.items()):
                raise ValueError('Phase-A resume identity differs')
            for key, branch in branches.items():
                branch.load_state_dict(checkpoint['features'][key], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            cursor, step = checkpoint['samples_seen'], checkpoint['global_step']
            if cursor != min(step * 128, len(manifest['A'])):
                raise ValueError('Invalid phase-A cursor')
            restore_rng(checkpoint['rng_by_rank'][rank])
            del checkpoint
        hooks = TransferHooks(teacher, branches)
        if rank == 0:
            (output / 'run.json').write_text(json.dumps({**identity, 'argv': vars(args), 'config': config}, indent=2))

        def save():
            states = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(rng_state(), states, dst=0)
            else:
                states = [rng_state()]
            if rank == 0:
                checkpoint = {**identity, 'complete': cursor == len(manifest['A']),
                    'global_step': step, 'samples_seen': cursor, 'rng_by_rank': states,
                    'features': {key: {name: p.detach().cpu() for name, p in branch.state_dict().items()}
                                 for key, branch in branches.items()},
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
                temporary = output / 'latest.pt.tmp'
                torch.save(checkpoint, temporary)
                temporary.replace(output / 'latest.pt')
            if world > 1:
                dist.barrier()

        stop = min(total_steps, args.max_steps) if args.max_steps is not None else total_steps
        while step < stop:
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            hooks.denominator = min(128, len(manifest['A']) - cursor)
            hooks.diagnostics.clear()
            for index in sample_group(manifest['A'], cursor, rank, world):
                staged, metadata = prepare_sample(teacher, dataset, collate, index, device)
                inputs = {key: value.to(device) for key, value in staged.items()
                          if key not in ('labels', 'prefix_valid_mask')}
                hooks.valid = metadata['valid_mask'].to(device)
                hooks.visual = metadata['image_token_mask'].to(device)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    teacher.get_model()(**inputs, use_cache=False)
            diagnostics = {}
            for key, branch in branches.items():
                for parameter in branch.parameters():
                    if parameter.grad is None:
                        raise RuntimeError('Missing phase-A gradient')
                    if world > 1:
                        dist.all_reduce(parameter.grad)
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError('Nonfinite phase-A gradient')
                norm = torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
                values = hooks.diagnostics[key]
                if world > 1:
                    dist.all_reduce(values)
                values = values.tolist()
                diagnostics[key] = {'normalized_sample_mean': values[0] / hooks.denominator,
                    'visual_raw': values[1] / max(values[3], 1), 'visual_normalized': values[2] / max(values[3], 1),
                    'visual_samples': values[3], 'text_raw': values[4] / max(values[6], 1),
                    'text_normalized': values[5] / max(values[6], 1), 'text_samples': values[6],
                    'gate_rms': values[7] / hooks.denominator, 'state_rms': values[8] / hooks.denominator,
                    'grad_norm': float(norm)}
            optimizer.step()
            scheduler.step()
            cursor += hooks.denominator
            step += 1
            if rank == 0:
                record = {'step': step, 'samples_seen': cursor, 'seconds': time.perf_counter() - started,
                          'learning_rate': scheduler.get_last_lr()[0], 'layers': diagnostics,
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                with (output / 'steps.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
            if step == 1 or step % args.save_every == 0 or step == stop:
                save()
        if rank == 0:
            (output / 'result.json').write_text(json.dumps({**identity, 'complete': cursor == len(manifest['A']),
                'samples_seen': cursor, 'global_step': step}, indent=2))
    finally:
        if hooks:
            hooks.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
