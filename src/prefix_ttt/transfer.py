"""Independent pre-o_proj residual transfer; teacher continuation stays unchanged."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from prefix_ttt.config import load_config
from prefix_ttt.data_pipeline import (load_manifest, build_dataset, prepare_sample,
                                      micro_batches, batch_loader)
from prefix_ttt.digests import digest_json
from prefix_ttt.runtime import (LATEST, SEED, distributed_context, load_rng, restore_rng,
                                save_atomic, save_rng, seed_everything)
from prefix_ttt.model.bridge import load_checkpoint, load_tokenizer
from prefix_ttt.ops.features import FeatureReadout
from prefix_ttt.ops.fla import fla_prefix
from prefix_ttt.ops.local import local_attention
from prefix_ttt.ops.reference import chunk_prefix
from prefix_ttt.training import (EFFECTIVE_BATCH_SIZE, GRAD_CLIP, accumulation_steps,
                                 cosine_factor, residual_transfer_loss)


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
                # Average per sample first, then over samples, so the logged sums
                # do not depend on how samples are grouped into micro-batches.
                samples = torch.tensor(errors.shape[0], dtype=torch.float32, device=errors.device)
                for mask in (self.visual & self.valid, ~self.visual & self.valid):
                    tokens = mask.sum(1).clamp_min(1)
                    present = mask.any(1)
                    raw = (errors * mask).sum(1) / tokens
                    normalized = raw / ((energy * mask).sum(1) / tokens + 1e-6)
                    values.extend([(raw * present).sum(), (normalized * present).sum(),
                                   present.sum().float()])
                gate = torch.nn.functional.linear(x, branch.gate_weight).float()
                values.extend([gate.square().mean().sqrt() * samples,
                               state.detach().float().square().mean().sqrt() * samples])
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
    parser.add_argument('--resume')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--save-every', type=int, default=25)
    args = parser.parse_args()
    if args.save_every <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        parser.error('Step limits must be positive')
    config = load_config(args.config)
    rank, world, device = distributed_context('Phase A')
    micro = int(config['training']['micro_batch_size'])
    batches_per_step = accumulation_steps(world, micro)
    hooks = None
    try:
        seed_everything(SEED)
        manifest, sha = load_manifest(args.manifest, config['data_root'])
        total_steps = math.ceil(len(manifest['A']) / EFFECTIVE_BATCH_SIZE)
        identity = dict(stage='A', manifest_sha256=sha, config_sha256=digest_json(config),
                        total_steps=total_steps, diagnostic_only=args.max_steps is not None)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if not args.resume and (output / LATEST).exists():
            raise ValueError('Existing phase A requires explicit resume')
        path = Path(config['data_root']) / config['model_relative_path']
        teacher, _ = load_checkpoint(path, dtype=torch.bfloat16)
        teacher.requires_grad_(False).eval().to(device)
        tokenizer = load_tokenizer(path)
        dataset, collate = build_dataset(config, teacher, tokenizer, manifest)
        anchors = set(config['full_attention_layers'])
        branches = nn.ModuleDict({str(index): FeatureReadout(seed=42 + index)
                                  for index in config['candidate_ttt_layers'] if index not in anchors}).to(device)
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
            if cursor != min(step * EFFECTIVE_BATCH_SIZE, len(manifest['A'])):
                raise ValueError('Invalid phase-A cursor')
            # Per-rank sidecar first; checkpoints written before it carry the
            # gathered states in the payload, so an old file still resumes exactly.
            rng = load_rng(args.resume, rank)
            if rng is None:
                states = checkpoint.get('rng_by_rank', [])
                rng = states[rank] if rank < len(states) else None
            if rng is None:
                print(f'No RNG state for rank {rank}; keeping the initial RNG', flush=True)
            else:
                restore_rng(rng)
            del checkpoint
        hooks = TransferHooks(teacher, branches)
        if rank == 0:
            (output / 'run.json').write_text(json.dumps({**identity, 'world_size': world, 'argv': vars(args), 'config': config}, indent=2))

        def save():
            save_rng(output / LATEST, rank)
            if rank == 0:
                save_atomic(output / LATEST, {
                    **identity, 'world_size': world, 'complete': cursor == len(manifest['A']),
                    'global_step': step, 'samples_seen': cursor,
                    'features': {key: {name: p.detach().cpu() for name, p in branch.state_dict().items()}
                                 for key, branch in branches.items()},
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()})
            if world > 1:
                dist.barrier()

        stop = min(total_steps, args.max_steps) if args.max_steps is not None else total_steps
        index_batches = [batch for start in range(cursor, len(manifest['A']), EFFECTIVE_BATCH_SIZE)
                         for batch in micro_batches(manifest['A'], start, rank, world, micro)]
        stream = iter(batch_loader(dataset, collate, index_batches,
                                   int(config['training'].get('dataloader_workers', 0))))
        while step < stop:
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad(set_to_none=True)
            hooks.denominator = min(EFFECTIVE_BATCH_SIZE, len(manifest['A']) - cursor)
            hooks.diagnostics.clear()
            for batch in [next(stream) for _ in range(batches_per_step)]:
                staged, metadata = prepare_sample(teacher, batch, device)
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
                norm = torch.nn.utils.clip_grad_norm_(branch.parameters(), GRAD_CLIP)
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
            (output / 'result.json').write_text(json.dumps({**identity, 'world_size': world, 'complete': cursor == len(manifest['A']),
                'samples_seen': cursor, 'global_step': step}, indent=2))
    finally:
        if hooks:
            hooks.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
