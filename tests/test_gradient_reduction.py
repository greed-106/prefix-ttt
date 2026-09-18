"""Two-rank gloo test: the bucketed reduction must equal per-tensor all_reduce."""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from prefix_ttt.training import all_reduce_gradients


def _worker(rank, world, init_file, queue):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(29531 + rank)
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank, world_size=world)
    torch.manual_seed(rank)
    # Mixed shapes and dtypes, plus a non-contiguous transpose, to exercise bucketing.
    parameters = []
    expected = []
    for shape in ((4, 3), (7,), (2, 5)):
        parameter = torch.nn.Parameter(torch.zeros(shape))
        grad = torch.randn(shape)
        parameter.grad = grad.clone()
        expected.append(grad.clone())
        parameters.append(parameter)
    # A non-contiguous gradient of matching shape, so the flatten path is exercised.
    parameter = torch.nn.Parameter(torch.zeros(5, 3))
    grad = torch.randn(3, 5).t()
    parameter.grad = grad.clone()
    expected.append(grad.clone())
    parameters.append(parameter)
    bf16 = torch.nn.Parameter(torch.zeros(6, dtype=torch.bfloat16))
    bf16.grad = torch.randn(6, dtype=torch.bfloat16)
    expected.append(bf16.grad.clone())
    parameters.append(bf16)

    # Reference: one collective per tensor.
    for parameter in parameters:
        dist.all_reduce(parameter.grad.clone())  # warm the communicator, value discarded
    for index, parameter in enumerate(parameters):
        reference = parameter.grad.clone()
        dist.all_reduce(reference)
        parameter.grad.copy_(expected[index])
        dist.all_reduce(parameter.grad)
        assert torch.equal(parameter.grad, reference), f'tensor {index} reference disagrees'

    # Bucketed implementation.
    for index, parameter in enumerate(parameters):
        parameter.grad.copy_(expected[index])
    all_reduce_gradients(parameters, world)
    for index, parameter in enumerate(parameters):
        reference = expected[index].clone()
        dist.all_reduce(reference)
        queue.put((rank, index, torch.equal(parameter.grad, reference)))
    dist.destroy_process_group()


def test_bucketed_reduction_matches_per_tensor_reduction():
    world = 2
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, 'init')
        context = mp.get_context('spawn')
        queue = context.Queue()
        processes = [context.Process(target=_worker, args=(rank, world, init_file, queue))
                     for rank in range(world)]
        for process in processes:
            process.start()
        results = [queue.get(timeout=120) for _ in range(world * 5)]
        for process in processes:
            process.join(timeout=120)
            assert process.exitcode == 0
    assert len(results) == world * 5
    assert all(match for _, _, match in results), results
