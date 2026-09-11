"""Shared run-time plumbing: process group, RNG capture, checkpoints, trainable IO.

Every formal GPU entry point (phase A, phase B, the diagnostics) needs the same
four things, so they live here instead of in whichever training script happened to
define them first.
"""
import json
import os
from pathlib import Path
import random
import socket

import numpy as np
import torch
import torch.distributed as dist


SEED = 42                    # data order, initialisation and sampling anchor
LATEST = 'latest.pt'         # rolling checkpoint, rewritten every save interval
PILOT = 'pilot.pt'           # snapshot at the pilot step of the fixed schedule


def distributed_context(purpose):
    """Set the CUDA device and join the NCCL group; returns rank, world, device.

    Ranks come from the launcher environment (torchrun), never from a config file.
    """
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if not torch.cuda.is_available():
        raise RuntimeError(f'{purpose} requires validated CUDA/FLA; no CPU fallback')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
        print(json.dumps({'host': socket.gethostname(), 'rank': rank, 'world_size': world,
                          'local_rank': local_rank,
                          'device': torch.cuda.get_device_name(local_rank)}), flush=True)
    return rank, world, device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def gather_rng(rank, world):
    """One snapshot per rank, collected on rank 0 so a resume stays bit-exact."""
    if world == 1:
        return [rng_state()]
    states = [None] * world if rank == 0 else None
    dist.gather_object(rng_state(), states, dst=0)
    return states


def save_atomic(path, state):
    """Write beside the final name and rename, so a killed job leaves no half file."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def load_trainable(model, state):
    """Restore every trainable tensor by name, rejecting any name or shape drift."""
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError('Checkpoint trainable parameter names differ')
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape:
                raise ValueError(f'Checkpoint shape mismatch: {name}')
            parameter.copy_(state[name])
