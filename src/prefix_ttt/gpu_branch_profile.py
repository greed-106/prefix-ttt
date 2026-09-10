"""Profile production-size Local/FLA forward operators after warmup."""
import argparse
import json

import torch

from prefix_ttt.ops.fla import fla_prefix
from prefix_ttt.ops.local import local_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=2048)
    args = parser.parse_args()
    torch.manual_seed(42)
    values = [torch.randn(1, args.length, 32, 128, device='cuda', dtype=torch.bfloat16)
              for _ in range(3)]
    with torch.no_grad():
        for name, function in [('local32', local_attention), ('prefix_fla', fla_prefix)]:
            for _ in range(2):
                function(*values)
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True) as profile:
                function(*values)
                torch.cuda.synchronize()
            events = profile.events()
            kernels = sorted({event.name for event in events
                              if event.device_type == torch.autograd.DeviceType.CUDA})
            sdpa_shapes = [event.input_shapes for event in events
                           if event.name == 'aten::scaled_dot_product_attention']
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                function(*values)
            end.record()
            torch.cuda.synchronize()
            print(json.dumps({'branch': name, 'length': args.length, 'batch': 1,
                              'mean_forward_ms': start.elapsed_time(end) / 10,
                              'sdpa_input_shapes': sdpa_shapes, 'cuda_kernels': kernels}), flush=True)


if __name__ == '__main__':
    main()
