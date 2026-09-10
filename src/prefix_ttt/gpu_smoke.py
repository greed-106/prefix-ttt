"""Disposable real-checkpoint SFT capacity diagnostic; never a formal B run.

Run directly on one GPU or via torch.distributed.run on authorized GPUs.
Uses the frozen manifest and the copied original LLaVA dataset/preprocessor.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import ijson
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from prefix_ttt.model.bridge import load_checkpoint, load_tokenizer
from prefix_ttt.model.hybrid import install_prefix_ttt
from prefix_ttt.model.trainability import install_lora, audit_parameters
from prefix_ttt.training import optimizer_groups, token_normalized_ce


def selected_records(manifest, audit_path, count, minimum_length):
    """Select fixed train examples for stress testing, without modifying order."""
    eligible = set()
    with Path(audit_path).open() as handle:
        for line in handle:
            row = json.loads(line)
            if row['status'] == 'valid' and row['expanded_length'] >= minimum_length:
                eligible.add(row['index'])
    indices = [i for i in manifest['train'] if i in eligible][:count]
    if len(indices) < count:
        raise ValueError(f'Only {len(indices)} eligible samples for requested {count}')
    wanted = set(indices)
    with Path(manifest['annotation']).open('rb') as handle:
        records = {i: row for i, row in enumerate(ijson.items(handle, 'item')) if i in wanted}
    return indices, [records[i] for i in indices]


class BranchTimer:
    """CUDA events around the actual called branches, including recomputation."""
    def __init__(self):
        import prefix_ttt.model.hybrid as hybrid
        self.module = hybrid
        self.original = {}
        self.events = []
        for name in ('local_attention', 'fla_prefix'):
            original = getattr(hybrid, name)
            self.original[name] = original

            def wrapped(*args, _name=name, _original=original, **kwargs):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                output = _original(*args, **kwargs)
                end.record()
                self.events.append((_name, start, end))
                return output

            setattr(hybrid, name, wrapped)

    def collect(self):
        result = {}
        for name, start, end in self.events:
            row = result.setdefault(name, {'calls': 0, 'cuda_ms': 0.0})
            row['calls'] += 1
            row['cuda_ms'] += start.elapsed_time(end)
        self.events.clear()
        return result

    def close(self):
        for name, function in self.original.items():
            setattr(self.module, name, function)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/base.json')
    parser.add_argument('--manifest', default='artifacts/cpu/fixed_manifest.json')
    parser.add_argument('--label-audit', default='artifacts/cpu/labels-full/label-audit.jsonl')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--minimum-length', type=int, default=2000)
    parser.add_argument('--layout', choices=['E1', 'E2'], default='E2')
    args = parser.parse_args()
    if args.steps < 1 or args.accumulation < 1:
        parser.error('steps and accumulation must be positive')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f'rank-{rank}.json'
    report = {'status': 'running', 'diagnostic_only': True, 'rank': rank, 'world_size': world,
              'device': torch.cuda.get_device_name(device), 'args': vars(args), 'steps': []}
    def write_report():
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    write_report()
    timer = None
    try:
        torch.manual_seed(42)
        config = json.loads(Path(args.config).read_text())
        manifest_bytes = Path(args.manifest).read_bytes()
        manifest = json.loads(manifest_bytes)
        report['manifest_sha256'] = hashlib.sha256(manifest_bytes).hexdigest()
        count = args.steps * args.accumulation * world
        indices, records = selected_records(manifest, args.label_audit, count, args.minimum_length)
        report['sample_indices'] = indices
        checkpoint_path = Path(config['data_root']) / config['model_relative_path']
        model, report['checkpoint'] = load_checkpoint(checkpoint_path, dtype=torch.bfloat16)
        tokenizer = load_tokenizer(checkpoint_path)
        from llava import conversation
        from llava.train.train import LazySupervisedDataset, DataCollatorForSupervisedDataset
        conversation.default_conversation = conversation.conv_templates['v1']
        # Reuse the exact original dataset implementation; avoid loading 665k
        # conversations per rank when this diagnostic needs only a few rows.
        dataset = LazySupervisedDataset.__new__(LazySupervisedDataset)
        dataset.tokenizer = tokenizer
        dataset.list_data_dict = records
        dataset.data_args = SimpleNamespace(is_multimodal=True, mm_use_im_start_end=False,
            image_aspect_ratio=model.config.image_aspect_ratio,
            image_folder=str(Path(config['data_root']) / 'datasets/llava-665k/images'),
            image_processor=model.get_vision_tower().image_processor)
        collate = DataCollatorForSupervisedDataset(tokenizer)
        new_parameters = install_prefix_ttt(model, backend='fla') if args.layout == 'E2' else []
        model = install_lora(model, new_parameters=new_parameters)
        model.to(device)  # Never cast FP32 RoPE buffers or new master parameters.
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
        base = model.get_base_model()
        base.get_vision_tower().eval()
        audit = audit_parameters(model, new_parameters=new_parameters)
        report['trainable_parameters'] = sum(p['numel'] for p in audit if p['requires_grad'])
        report['anchor_attention_implementation'] = model.config._attn_implementation
        report['ttt_backend'] = 'fla' if args.layout == 'E2' else None
        report['local_backend'] = 'batched nonoverlapping causal SDPA blocks32' if args.layout == 'E2' else None
        optimizer = torch.optim.AdamW(optimizer_groups(model, new_parameters),
                                     betas=(0.9, 0.95), eps=1e-8)
        train_model = (DistributedDataParallel(model, device_ids=[local_rank],
                         broadcast_buffers=False, find_unused_parameters=False,
                         gradient_as_bucket_view=True) if world > 1 else model)
        timer = BranchTimer()
        write_report()
        for step in range(args.steps):
            prepared_group = []
            target_count = 0
            lengths = []
            # Frozen original multimodal expansion creates labels before the
            # accumulation-group denominator is chosen. Do not infer labels
            # from the unexpanded text sequence.
            for micro in range(args.accumulation):
                index = (step * args.accumulation + micro) * world + rank
                batch = {key: tensor.to(device) for key, tensor in collate([dataset[index]]).items()}
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    prepared, metadata = base.prepare_inputs_labels_for_multimodal(
                        batch['input_ids'], None, batch['attention_mask'], None,
                        batch['labels'], batch['images'], return_metadata=True)
                ids, positions, mask, _, embeds, labels = prepared
                length = int(metadata['valid_mask'].sum())
                if not args.minimum_length <= length <= 2048:
                    raise ValueError(f'Unexpected expanded stress length: {length}')
                targets = int(labels[:, 1:].ne(-100).sum())
                if targets == 0:
                    raise ValueError('Preprocessing produced no supervision; do not discard sample')
                target_count += targets
                lengths.append(length)
                # These frozen input embeddings are about 16MiB per sample.
                # CPU staging avoids accumulation-dependent GPU activation retention.
                prepared_group.append({key: value.cpu() for key, value in dict(
                    input_ids=ids, position_ids=positions, attention_mask=mask,
                    inputs_embeds=embeds, labels=labels,
                    prefix_valid_mask=metadata['valid_mask']).items() if value is not None})
            total_targets = torch.tensor(target_count, dtype=torch.int64, device=device)
            if world > 1:
                dist.all_reduce(total_targets)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss_value = 0.0
            for micro, cpu_batch in enumerate(prepared_group):
                batch = {key: value.to(device) for key, value in cpu_batch.items()}
                labels = batch.pop('labels')
                sync = train_model.no_sync() if world > 1 and micro + 1 < args.accumulation else nullcontext()
                with sync:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = train_model(**batch, use_cache=False)
                        loss = token_normalized_ce(result.logits, labels, int(total_targets), world)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite real SFT loss')
                    loss.backward()
                loss_value += float(loss.detach())
                del result, loss, batch
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError(f'Missing/nonfinite gradient: {name}')
                    kind = 'feature' if 'a_phi' in name or 'b_phi' in name else ('gate' if 'gate_weight' in name else 'lora')
                    gradients[kind] = gradients.get(kind, 0.0) + float(parameter.grad.float().square().sum())
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            report['steps'].append({'step': step + 1, 'loss_rank_contribution': loss_value,
                'global_targets': int(total_targets), 'expanded_lengths': lengths,
                'seconds': time.perf_counter() - start, 'grad_norm': float(grad_norm),
                'gradient_squared_norm_by_kind': gradients, 'branches': timer.collect(),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
            print(json.dumps({'rank': rank, **report['steps'][-1]}), flush=True)
            write_report()
        # Disposable trainable-only debug checkpoint, plus real AdamW state;
        # never claim this artifact is a phase-A or official phase-B checkpoint.
        if rank == 0:
            checkpoint = {'trainable': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                          'optimizer': optimizer.state_dict(), 'steps': args.steps,
                          'diagnostic_only': True, 'manifest_sha256': report['manifest_sha256']}
            save_path = output / 'debug-checkpoint.pt'
            torch.save(checkpoint, save_path)
            del checkpoint
            restored = torch.load(save_path, map_location='cpu', weights_only=True)
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    if not torch.equal(parameter.detach().cpu(), restored['trainable'][name]):
                        raise AssertionError(f'Checkpoint roundtrip mismatch: {name}')
            optimizer.load_state_dict(restored['optimizer'])
            report['checkpoint_roundtrip'] = True
            report['debug_checkpoint'] = str(save_path)
        if world > 1:
            dist.barrier()
        report['status'] = 'passed'
        report['limitations'] = ['No phase-A weights: zero-gate initialization and disposable optimizer only',
            'No benchmark or generation acceptance; no full training trajectory',
            'Branch events include forward and checkpoint recomputation, not independent backward kernel attribution',
            'First iteration includes compilation; not a steady-state speed benchmark']
        write_report()
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        write_report()
        raise
    finally:
        if timer is not None:
            timer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
