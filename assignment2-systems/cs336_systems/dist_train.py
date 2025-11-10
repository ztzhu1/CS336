from copy import deepcopy
from functools import partial
import os
import random
from timeit import default_timer
from typing import Any, Type

import numpy as np
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import SGD, Optimizer


def set_random_seed(seed):
    """
    from vllm.model_executor
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


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
    return default_timer() - t0


class NaiveModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 10)
        self.linear2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        return x


def _naive_ddp(
    rank: int, world_size: int, model_class: torch.nn.Module, data, device, flatten
):
    setup(rank, world_size, device=device)
    set_random_seed(0)
    model = model_class().to(device)
    opt = SGD(model.parameters(), lr=1)
    local_model = deepcopy(model)
    local_opt = SGD(local_model.parameters(), lr=1)
    local_data = data.clone().view(-1, data.shape[-1]).to(device)
    for step in range(3):
        n = step
        data = data + n

        local_data = local_data + n
        local_loss = local_model(local_data).mean()
        local_loss.backward()
        local_opt.step()
        local_opt.zero_grad()

        loss = model(data[rank].to(device)).mean()
        loss.backward()
        # dist.barrier()
        if flatten:
            flattened_tensor = _flatten_dense_tensors(
                [p.grad for p in model.parameters()]
            )
            dist.all_reduce(flattened_tensor, op=dist.ReduceOp.AVG, async_op=False)
            grads = _unflatten_dense_tensors(
                flattened_tensor, [p.grad for p in model.parameters()]
            )
            for p, g in zip(model.parameters(), grads):
                p.grad = g
        else:
            handles = []
            for name, p in model.named_parameters():
                handle = dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
                handles.append(handle)
            for handle in handles:
                handle.wait()
        opt.step()
        opt.zero_grad()
        for param, local_param in zip(model.parameters(), local_model.parameters()):
            if not torch.allclose(param.data, local_param.data, rtol=0, atol=1e-7):
                print(f"rank {rank} param-local_param:", param.data - local_param.data)
                # print(f"rank {rank} local_param:", local_param.data)
                raise ValueError(f"rank {rank} step {step} parameters do not match")
        print(f"rank {rank} step {step} pass check")
    dist.destroy_process_group()


def naive_ddp(world_size=4, device="cpu", flatten=False):
    t0 = default_timer()
    set_random_seed(1)
    data = torch.randn((world_size, 2, 5))
    mp.spawn(
        fn=_naive_ddp,
        args=(world_size, NaiveModel, data, device, flatten),
        nprocs=world_size,
        join=True,
    )
    return default_timer() - t0


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.hook_handles = []
        for param in self.module.parameters():
            if param.requires_grad:
                hook_handle = param.register_post_accumulate_grad_hook(
                    partial(communication_hook, handles=self.handles)
                )
                self.hook_handles.append(hook_handle)
        obj = [module.state_dict()]
        dist.broadcast_object_list(obj, src=0)
        module.load_state_dict(obj[0])

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


class Bucket:
    def __init__(self, bucket_size_mb):
        self.bucket_size_mb = bucket_size_mb
        self.curr_size_mb = 0
        self.tensors = []
        self.tensor_flattened_tensor_pair = []
        self.handles = []

    def add_tensor(self, tensor: torch.Tensor):
        tensor_size = tensor.nbytes / 1e6
        self.tensors.append(tensor)
        self.curr_size_mb += tensor_size
        if self.curr_size_mb >= self.bucket_size_mb:
            self.all_reduce()

    def all_reduce(self):
        if len(self.tensors) == 0:
            assert self.curr_size_mb == 0
            return
        flattened_tensor = _flatten_dense_tensors(self.tensors)
        self.tensor_flattened_tensor_pair.append((self.tensors, flattened_tensor))
        handle = dist.all_reduce(flattened_tensor, op=dist.ReduceOp.AVG, async_op=True)
        self.handles.append(handle)
        self.tensors = []
        self.curr_size_mb = 0

    def finish_gradient_synchronization(self):
        self.all_reduce()
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        for tensors, flattened_tensor in self.tensor_flattened_tensor_pair:
            torch.nn.utils.vector_to_parameters(flattened_tensor, tensors)
        self.tensor_flattened_tensor_pair = []


class DDPBucketedParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket = Bucket(bucket_size_mb)
        self.hook_handles = []
        for param in self.module.parameters():
            if param.requires_grad:
                hook_handle = param.register_post_accumulate_grad_hook(
                    partial(bucket_communication_hook, bucket=self.bucket)
                )
                self.hook_handles.append(hook_handle)
        obj = [module.state_dict()]
        dist.broadcast_object_list(obj, src=0)
        module.load_state_dict(obj[0])

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        self.bucket.finish_gradient_synchronization()


def communication_hook(tensor, handles):
    handle = dist.all_reduce(tensor.grad, op=dist.ReduceOp.AVG, async_op=True)
    handles.append(handle)


def bucket_communication_hook(tensor, bucket: Bucket):
    bucket.add_tensor(tensor.grad)


#TODO
class ShardedStateOptimizer(Optimizer):
    """
    An optimizer that shards its state across multiple processes.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters to be optimized.
        optimizer_cls (:class:`torch.nn.Optimizer`): the class of the underlying
            optimizer to shard.
        **kwargs: additional keyword arguments passed to the underlying
            optimizer.
    """

    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs):
        params = list(params)  # for lower version torch?
        super().__init__(params, kwargs)
        self.opt = optimizer_cls(params, **kwargs)

    def step(self, closure=None):
        self.opt.step(closure)

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
