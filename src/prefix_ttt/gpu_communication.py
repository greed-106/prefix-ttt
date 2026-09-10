"""Short four-rank NCCL correctness and timing check, not a model benchmark."""
import json
import os
import time

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    value = torch.tensor([float(dist.get_rank() + 1)], device='cuda')
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    assert value.item() == expected
    payload = torch.ones(16 * 1024 * 1024, device='cuda')
    for _ in range(3):
        dist.all_reduce(payload)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        dist.all_reduce(payload)
    torch.cuda.synchronize()
    print(json.dumps({'rank': rank, 'device': torch.cuda.get_device_name(),
                      'world_size': dist.get_world_size(), 'sum': value.item(),
                      'payload_bytes': payload.numel() * payload.element_size(),
                      'mean_allreduce_ms': (time.perf_counter() - start) * 100}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
