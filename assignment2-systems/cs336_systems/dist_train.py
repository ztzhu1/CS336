import os
from timeit import default_timer

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size, device="cpu"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if str(device) == "cuda":
        backend = "nccl"
    else:
        backend = "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    # if str(device) == "cuda":
    # torch.cuda.set_device(rank)


def distributed_demo(rank, world_size, device="cpu"):
    setup(rank, world_size, device=device)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")
    dist.destroy_process_group()


def run_simple_sum():
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size,), nprocs=world_size, join=True)


def distributed_sum(rank, world_size, data_size, device, t0):
    setup(rank, world_size, device=device)
    for _ in range(3):  # warm up
        data = torch.randn(int(data_size * 1e6 // 4 // world_size)).to(device)
        dist.all_reduce(data, async_op=False)
        if str(device) != "cpu":
            torch.cuda.synchronize()
    data = torch.randn(int(data_size * 1e6 // 4 // world_size)).to(device)
    t1 = default_timer() - t0
    print(
        f"rank {rank} data (before all-reduce,{t1:.3f}): {data.nbytes / 1e6} MB, {data[:5]} ..."
    )
    dist.all_reduce(data, async_op=False)
    if str(device) != "cpu":
        torch.cuda.synchronize()
    t2 = default_timer() - t0
    dt = t2 - t1
    print(
        f"rank {rank} data (after all-reduce,{t2:.3f},{dt:.3f}): {data.nbytes / 1e6} MB, {data[:5]} ..."
    )
    dist.destroy_process_group()


def run_sum(world_size=2, data_size=1, device="cpu"):
    """
    data_size: MB
    """
    t0 = default_timer()
    mp.spawn(
        fn=distributed_sum,
        args=(world_size, data_size, device, t0),
        nprocs=world_size,
        join=True,
    )
