"""One immutable B trajectory: matched E1/E2 Pilot, then exact continuation."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer

from prefix_ttt.config import load_config
from prefix_ttt.data_pipeline import (load_manifest, build_dataset, prepare_sample,
                                      micro_batches, batch_loader)
from prefix_ttt.digests import digest_file, digest_json
from prefix_ttt.runtime import (LATEST, PILOT, SEED, distributed_context, load_rng,
                                load_trainable, restore_rng, save_atomic, save_rng,
                                seed_everything, to_cpu)
from prefix_ttt.model.bridge import load_checkpoint, load_tokenizer
from prefix_ttt.model.hybrid import install_prefix_ttt
from prefix_ttt.model.labels import IGNORE_INDEX
from prefix_ttt.model.trainability import install_lora, enable_full_finetuning, audit_parameters
from prefix_ttt.optim import MasterWeightAdamW
from prefix_ttt.training import (ADAM_BETAS, ADAM_EPS, EFFECTIVE_BATCH_SIZE, GRAD_CLIP,
                                 accumulation_steps, all_reduce_gradients, cosine_factor,
                                 optimizer_groups, token_normalized_ce, trajectory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.json')
    parser.add_argument('--manifest', default='artifacts/cpu/fixed_manifest.json')
    parser.add_argument('--layout', choices=['E1', 'E2'], required=True)
    parser.add_argument('--trainable', choices=['lora', 'full'], default='lora',
                        help='LoRA adapters (default) or full weight fine-tuning')
    parser.add_argument('--stage-a-checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--stop-at-pilot', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Run this many steps from the start point, then stop')
    parser.add_argument('--max-steps', type=int, help='Disposable debug trajectory only')
    parser.add_argument('--save-every', type=int, default=100)
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('--max-steps must be positive')
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error('--stop-after must be positive')
    if args.save_every <= 0:
        parser.error('--save-every must be positive')
    if args.layout == 'E2' and not args.stage_a_checkpoint:
        parser.error('E2 requires completed phase A; zero-gate debug is not formal B')
    config = load_config(args.config, require_base_lr=args.trainable == 'full')
    full = args.trainable == 'full'
    rank, world, device = distributed_context('Formal SFT')
    micro = int(config['training']['micro_batch_size'])
    batches_per_step = accumulation_steps(world, micro)
    try:
        seed_everything(SEED)
        manifest, manifest_sha = load_manifest(args.manifest, config['data_root'])
        schedule = trajectory(len(manifest['train']))
        identity = dict(stage='B', layout=args.layout, trainable_mode=args.trainable,
                        manifest_sha256=manifest_sha,
                        total_steps=schedule['total_steps'],
                        config_sha256=digest_json(config),
                        stage_a_sha256=digest_file(args.stage_a_checkpoint) if args.layout == 'E2' else None,
                        diagnostic_only=args.max_steps is not None)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if not args.resume and (output / LATEST).exists():
            raise ValueError('Existing trajectory requires explicit --resume')
        path = Path(config['data_root']) / config['model_relative_path']
        # Both arms keep the pretrained weights in BF16, the dtype the forward
        # computes in. Full fine-tuning adds a sharded FP32 master in optimizer
        # state (see prefix_ttt.optim) instead of holding the model in FP32, which
        # would make every projection cast its weight on every forward.
        model, _ = load_checkpoint(path, dtype=torch.bfloat16)
        tokenizer = load_tokenizer(path)
        dataset, collate = build_dataset(config, model, tokenizer, manifest)
        new_parameters = install_prefix_ttt(model, backend='fla') if args.layout == 'E2' else []
        if args.layout == 'E2':
            stage_a = torch.load(args.stage_a_checkpoint, map_location='cpu', weights_only=False)
            if (stage_a.get('stage') != 'A' or not stage_a.get('complete')
                    or stage_a.get('diagnostic_only', False) or stage_a.get('manifest_sha256') != manifest_sha
                    or stage_a.get('config_sha256') != identity['config_sha256']
                    or stage_a.get('samples_seen') != len(manifest['A'])
                    or stage_a.get('global_step') != math.ceil(len(manifest['A']) / EFFECTIVE_BATCH_SIZE)):
                raise ValueError('Phase A checkpoint is incomplete or uses a different manifest')
            installed = {str(index) for index, layer in enumerate(model.model.layers)
                         if hasattr(layer.self_attn, 'prefix_ttt')}
            unused = set(stage_a['features']) - installed
            if unused:
                print(f'Phase A checkpoint carries unused feature layers: {sorted(unused, key=int)}', flush=True)
            for index, layer in enumerate(model.model.layers):
                if hasattr(layer.self_attn, 'prefix_ttt'):
                    layer.self_attn.prefix_ttt.load_state_dict(stage_a['features'][str(index)], strict=True)
            del stage_a
        if full:
            enabled = enable_full_finetuning(model)
            base = model
        else:
            model = install_lora(model, new_parameters=new_parameters)
            base = model.get_base_model()
        model.to(device)  # Preserve FP32 master parameters AND original RoPE buffers.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        base.get_vision_tower().eval()
        parameters = [p for p in model.parameters() if p.requires_grad]
        groups = optimizer_groups(model, new_parameters, allow_base=full)
        if full:
            # ZeRO shards the FP32 master and Adam moments of the pretrained weights,
            # which stay BF16 in the module; the Prefix-TTT parameters are FP32 and
            # small enough that sharding them would only add collectives.
            base_groups = [group for group in groups if group['name'] == 'base']
            new_groups = [group for group in groups if group['name'] == 'new_module']
            if any(p.dtype != torch.bfloat16 for group in base_groups for p in group['params']):
                raise ValueError('Base weights must stay BF16 for the sharded master')
            if any(p.dtype != torch.float32 for group in new_groups for p in group['params']):
                raise ValueError('Prefix-TTT parameters must stay FP32')
            optimizers = [
                ZeroRedundancyOptimizer(base_groups, optimizer_class=MasterWeightAdamW,
                                        betas=ADAM_BETAS, eps=ADAM_EPS),
                torch.optim.AdamW(new_groups, betas=ADAM_BETAS, eps=ADAM_EPS)]
        else:
            optimizers = [torch.optim.AdamW(groups, betas=ADAM_BETAS, eps=ADAM_EPS)]
        schedule_lr = lambda step: cosine_factor(step, schedule['total_steps'])
        schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lr)
                      for optimizer in optimizers]
        scheduler = schedulers[0]
        cursor = step = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            if any(checkpoint.get(key) != value for key, value in identity.items()):
                raise ValueError('Resume trajectory identity differs')
            load_trainable(model, checkpoint['trainable'])
            # ZeRO's public state_dict() needs a full gather that does not fit in
            # memory, so the full arm stores one optimizer shard per rank instead.
            shard = Path(f'{args.resume}.optim-rank{rank}')
            if not full:
                optimizer.load_state_dict(checkpoint['optimizer'])
            elif shard.exists() and checkpoint.get('world_size') == world:
                stored = torch.load(shard, map_location='cpu', weights_only=False)
                optimizers[0].optim.load_state_dict(stored['base'])
                optimizers[1].load_state_dict(stored['ttt'])
            elif rank == 0:
                print('No usable optimizer shard next to the resume file; '
                      'Adam restarts cold', flush=True)
            for target, state in zip(schedulers, checkpoint['scheduler'] if full
                                     else [checkpoint['scheduler']]):
                target.load_state_dict(state)
            cursor, step = checkpoint['samples_seen'], checkpoint['global_step']
            if cursor != min(step * EFFECTIVE_BATCH_SIZE, len(manifest['train'])):
                raise ValueError('Invalid resume cursor')
            if checkpoint.get('world_size') != world:
                print(f'Resuming across world_size {checkpoint.get("world_size")} -> {world}; '
                      f'ranks without a stored state keep their initial RNG', flush=True)
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
        if rank == 0:
            (output / 'trainable_params.json').write_text(json.dumps(
                audit_parameters(model, new_parameters=new_parameters, allow_base=full), indent=2))
            (output / 'run.json').write_text(json.dumps({**identity, 'world_size': world, **schedule, 'config': config, 'argv': vars(args)}, indent=2))

        def trainable_state():
            """Every tensor the run owns, in the dtype the evaluation model uses.

            The LoRA arm stores its adapters and reads the base from the released
            checkpoint; the full arm trained the base itself, so its weights are
            stored too. Both keep BF16 base weights and FP32 Prefix-TTT parameters,
            which is already how the model holds them.
            """
            if not full:
                return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            return {n: p.detach().cpu() for n, p in model.named_parameters()}

        def save(name, complete=False, optimizer_shards=False):
            save_rng(output / name, rank)
            if rank == 0:
                state = {**identity, 'world_size': world, 'complete': complete, 'global_step': step,
                         'samples_seen': cursor,
                         'trainable': trainable_state(),
                         'scheduler': ([s.state_dict() for s in schedulers] if full
                                       else scheduler.state_dict())}
                if not full:
                    state['optimizer'] = optimizer.state_dict()
                save_atomic(output / name, state)
            if full and optimizer_shards:
                save_atomic(output / f'{name}.optim-rank{rank}',
                            {'base': to_cpu(optimizers[0].optim.state_dict()),
                             'ttt': to_cpu(optimizers[1].state_dict())})
            if world > 1:
                dist.barrier()

        stop = schedule['pilot_step'] if args.stop_at_pilot else schedule['total_steps']
        if args.max_steps is not None:
            stop = min(stop, args.max_steps)
        if args.stop_after is not None:
            stop = min(stop, step + args.stop_after)
        index_batches = [batch for start in range(cursor, len(manifest['train']), EFFECTIVE_BATCH_SIZE)
                         for batch in micro_batches(manifest['train'], start, rank, world, micro)]
        stream = iter(batch_loader(dataset, collate, index_batches,
                                   int(config['training'].get('dataloader_workers', 0))))
        while step < stop:
            started = time.perf_counter()
            group = [prepare_sample(base, next(stream), device,
                                    trainable_embedding=full)[0] for _ in range(batches_per_step)]
            targets = torch.tensor(sum(int(batch['labels'][:, 1:].ne(IGNORE_INDEX).sum()) for batch in group), device=device)
            if world > 1:
                dist.all_reduce(targets)
            for target in optimizers:
                target.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            for staged in group:
                batch = {key: value.to(device) for key, value in staged.items()}
                labels = batch.pop('labels')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    result = model(**batch, use_cache=False)
                    # Manual SUM below, hence no DDP world-size multiplication.
                    loss = token_normalized_ce(result.logits, labels, int(targets))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite SFT loss')
                loss.backward()
                loss_sum += loss.detach()
                del result, loss, batch
            for parameter in parameters:
                if parameter.grad is None:
                    if group:
                        raise RuntimeError('Missing trainable gradient')
                    parameter.grad = torch.zeros_like(parameter)
            # One collective per dtype instead of one per tensor: on three hosts the
            # per-call latency of hundreds of small all-reduces dominated the step.
            all_reduce_gradients(parameters, world)
            norm = torch.nn.utils.clip_grad_norm_(parameters, GRAD_CLIP)
            for target in optimizers:
                target.step()
            for target in schedulers:
                target.step()
            step += 1
            cursor = min(cursor + EFFECTIVE_BATCH_SIZE, len(manifest['train']))
            if world > 1:
                dist.all_reduce(loss_sum)
            if rank == 0:
                record = dict(step=step, samples_seen=cursor, targets=int(targets), loss=float(loss_sum),
                              grad_norm=float(norm), seconds=time.perf_counter() - started,
                              learning_rates=scheduler.get_last_lr())
                with (output / 'steps.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
            if step == schedule['pilot_step']:
                save(PILOT, optimizer_shards=True)
            if step == 1 or step % args.save_every == 0 or step == stop:
                save(LATEST, complete=cursor == len(manifest['train']),
                     optimizer_shards=step == stop)
        if rank == 0:
            (output / 'result.json').write_text(json.dumps({**identity, 'world_size': world, 'global_step': step,
                'samples_seen': cursor, 'complete': cursor == len(manifest['train']),
                'pilot_reached': step >= schedule['pilot_step']}, indent=2))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
