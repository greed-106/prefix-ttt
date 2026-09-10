"""One immutable B trajectory: matched E1/E2 Pilot, then exact continuation."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist

from prefix_ttt.data_pipeline import load_manifest, build_dataset, prepare_sample
from prefix_ttt.manifests import digest_file, digest_json
from prefix_ttt.model.bridge import load_checkpoint, load_tokenizer
from prefix_ttt.model.hybrid import install_prefix_ttt
from prefix_ttt.model.trainability import install_lora, audit_parameters
from prefix_ttt.training import accumulation_steps, cosine_factor, optimizer_groups, token_normalized_ce, trajectory


def sample_group(order, cursor, rank, world_size):
    """A partial last group is never padded with repeated samples."""
    return order[cursor:cursor + 128][rank::world_size]


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def load_trainable(model, state):
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError('Checkpoint trainable parameter names differ')
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f'Checkpoint shape mismatch: {name}')
            parameter.copy_(state[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.json')
    parser.add_argument('--manifest', default='artifacts/cpu/fixed_manifest.json')
    parser.add_argument('--layout', choices=['E1', 'E2'], required=True)
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
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    accumulation_steps(world, 1)
    if not torch.cuda.is_available():
        raise RuntimeError('Formal SFT requires validated CUDA/FLA; no CPU fallback')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    try:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        config = json.loads(Path(args.config).read_text())
        manifest, manifest_sha = load_manifest(args.manifest, config['data_root'])
        schedule = trajectory(len(manifest['train']))
        identity = dict(stage='B', layout=args.layout, manifest_sha256=manifest_sha,
                        total_steps=schedule['total_steps'],
                        config_sha256=digest_json(config),
                        stage_a_sha256=digest_file(args.stage_a_checkpoint) if args.layout == 'E2' else None,
                        diagnostic_only=args.max_steps is not None)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        if not args.resume and (output / 'latest.pt').exists():
            raise ValueError('Existing trajectory requires explicit --resume')
        path = Path(config['data_root']) / config['model_relative_path']
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
                    or stage_a.get('global_step') != (len(manifest['A']) + 127) // 128):
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
        model = install_lora(model, new_parameters=new_parameters)
        model.to(device)  # Preserve FP32 master parameters AND original RoPE buffers.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        base = model.get_base_model()
        base.get_vision_tower().eval()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(optimizer_groups(model, new_parameters), betas=(0.9, 0.95), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_factor(step, schedule['total_steps']))
        cursor = step = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            if any(checkpoint.get(key) != value for key, value in identity.items()):
                raise ValueError('Resume trajectory identity differs')
            load_trainable(model, checkpoint['trainable'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            cursor, step = checkpoint['samples_seen'], checkpoint['global_step']
            if cursor != min(step * 128, len(manifest['train'])):
                raise ValueError('Invalid resume cursor')
            if checkpoint.get('world_size') != world:
                print(f'Resuming across world_size {checkpoint.get("world_size")} -> {world}; '
                      f'ranks without a stored state keep their initial RNG', flush=True)
            states = checkpoint['rng_by_rank']
            if rank < len(states):
                restore_rng(states[rank])
            del checkpoint
        if rank == 0:
            (output / 'trainable_params.json').write_text(json.dumps(audit_parameters(model, new_parameters=new_parameters), indent=2))
            (output / 'run.json').write_text(json.dumps({**identity, 'world_size': world, **schedule, 'config': config, 'argv': vars(args)}, indent=2))

        def save(name, complete=False):
            states = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(rng_state(), states, dst=0)
            else:
                states = [rng_state()]
            if rank == 0:
                state = {**identity, 'world_size': world, 'complete': complete, 'global_step': step,
                    'samples_seen': cursor, 'rng_by_rank': states,
                    'trainable': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
                temporary = output / (name + '.tmp')
                torch.save(state, temporary)
                temporary.replace(output / name)
            if world > 1:
                dist.barrier()

        stop = schedule['pilot_step'] if args.stop_at_pilot else schedule['total_steps']
        if args.max_steps is not None:
            stop = min(stop, args.max_steps)
        if args.stop_after is not None:
            stop = min(stop, step + args.stop_after)
        while step < stop:
            started = time.perf_counter()
            group = [prepare_sample(base, dataset, collate, index, device)[0]
                     for index in sample_group(manifest['train'], cursor, rank, world)]
            targets = torch.tensor(sum(int(batch['labels'][:, 1:].ne(-100).sum()) for batch in group), device=device)
            if world > 1:
                dist.all_reduce(targets)
            optimizer.zero_grad(set_to_none=True)
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
                if world > 1:
                    dist.all_reduce(parameter.grad)
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite SFT gradient')
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            cursor = min(cursor + 128, len(manifest['train']))
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
                save('pilot.pt')
            if step == 1 or step % args.save_every == 0 or step == stop:
                save('latest.pt', complete=cursor == len(manifest['train']))
        if rank == 0:
            (output / 'result.json').write_text(json.dumps({**identity, 'world_size': world, 'global_step': step,
                'samples_seen': cursor, 'complete': cursor == len(manifest['train']),
                'pilot_reached': step >= schedule['pilot_step']}, indent=2))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
