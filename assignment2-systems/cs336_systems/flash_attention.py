import argparse
import math
from pathlib import Path
import sys

project_dir = Path(__file__).parent.parent
if project_dir.joinpath("cs336-basics").as_posix() not in sys.path:
    sys.path.insert(0, project_dir.joinpath("cs336-basics").as_posix())

import einops
import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx
from tqdm import tqdm
import triton
import triton.language as tl

from cs336_basics.model import BasicsTransformerLM, annoted_scaled_dot_product_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_attention(d_model=16, seq_len=256):
    d_k = d_model
    warmup_steps = 5
    exe_steps = 10
    Q = torch.randn((8, seq_len, d_k), device=device, requires_grad=True)
    K = torch.randn((8, seq_len, d_k), device=device, requires_grad=True)
    V = torch.randn((8, seq_len, d_k), device=device, requires_grad=True)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()

    for i in range(warmup_steps):
        with nvtx.range(f"warmup {i}"):
            out = annoted_scaled_dot_product_attention(Q, K, V, mask=mask)
            torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    for i in range(exe_steps):
        with nvtx.range(f"forward {i}"):
            out = annoted_scaled_dot_product_attention(Q, K, V, mask=mask)
            torch.cuda.synchronize()
        loss = out.sum()
        with nvtx.range(f"backward {i}"):
            loss.backward()
            torch.cuda.synchronize()

    torch.cuda.memory._dump_snapshot(
        project_dir
        / "data"
        / "profile_memory"
        / f"torch_attn_d_model_{d_model}_seq_len_{seq_len}.pickle"
    )
    torch.cuda.memory._record_memory_history(enabled=None)


def jit_attention(d_model=16, seq_len=256):
    d_k = d_model
    warmup_steps = 5
    exe_steps = 10
    Q = torch.randn((8, seq_len, d_k), device=device, requires_grad=True)
    K = torch.randn((8, seq_len, d_k), device=device, requires_grad=True)
    V = torch.randn((8, seq_len, d_k), device=device, requires_grad=True)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
    compiled_scaled_dot_product_attention = torch.compile(
        annoted_scaled_dot_product_attention
    )

    for i in range(warmup_steps):
        with nvtx.range(f"warmup {i}"):
            out = compiled_scaled_dot_product_attention(Q, K, V, mask=mask)
            torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    for i in range(exe_steps):
        with nvtx.range(f"forward {i}"):
            out = compiled_scaled_dot_product_attention(Q, K, V, mask=mask)
            torch.cuda.synchronize()
        loss = out.sum()
        with nvtx.range(f"backward {i}"):
            loss.backward()
            torch.cuda.synchronize()

    torch.cuda.memory._dump_snapshot(
        project_dir
        / "data"
        / "profile_memory"
        / f"jit_attn_d_model_{d_model}_seq_len_{seq_len}.pickle"
    )
    torch.cuda.memory._record_memory_history(enabled=None)


def cdiv(a, b):
    return (a + b - 1) // b


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,  # Input pointers
    output_ptr,  # Output pointer
    x_stride_row,
    x_stride_dim,  # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim,  # Likely 1
    output_stride_row,  # Likely 1
    ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,  # Tile shapes must be known at compile time
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)

    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to use load/store at a time
    # - The order of the dimensions in memory from major to minor
    # axes (= np.argsort(strides)) for optimizations, especially useful on H100

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # Initialize a buffer to write to
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        # Load the current block pointer
        # Since ROWS_TILE_SIZE might not divide ROWS, and D_TILE_SIZE might not divide D,
        # we need boundary checks for both dimensions
        row = tl.load(
            x_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(
            weight_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (D_TILE_SIZE,)

        # Compute the weighted sum of the row.
        output += tl.sum(row * weight[None, :], axis=1)

        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas
        x_block_ptr = x_block_ptr.advance(
            (0, D_TILE_SIZE)
        )  # Move by D_TILE_SIZE in the last dimension
        weight_block_ptr = weight_block_ptr.advance(
            (D_TILE_SIZE,)
        )  # Move by D_TILE_SIZE

    # Write output to the output block pointer (a single scalar per row).
    # Since ROWS_TILE_SIZE might not divide ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward(
    x_ptr,
    weight_ptr,  # Input
    grad_output_ptr,  # Grad input
    grad_x_ptr,
    partial_grad_weight_ptr,  # Grad outputs
    stride_xr,
    stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr,
    stride_gxd,
    stride_gwb,
    stride_gwd,
    NUM_ROWS,
    D,
    ROWS_TILE_SIZE: tl.constexpr,
    D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    # Inputs
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(
            grad_output_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (ROWS_TILE_SIZE,)

        # Outer product for grad_x
        weight = tl.load(
            weight_block_ptr, boundary_check=(0,), padding_option="zero"
        )  # (D_TILE_SIZE,)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # Reduce as many rows as possible for the grad_weight result
        row = tl.load(
            x_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(
            partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)
        )  # Never out of bounds for dim 0

        # Move the pointers to the next tile along D
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance(
            (0, D_TILE_SIZE)
        )
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass, when we
        # only receive the gradient wrt. the output tensor, and

        # need to compute the gradients wrt. x and weight.
        D, output_dims = x.shape[-1], x.shape[:-1]

        # Reshape input tensor to 2D
        input_shape = x.shape
        x = einops.rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = (
            triton.next_power_of_2(D) // 16
        )  # Roughly 16 loops through the embedding dimension
        ctx.ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time
        ctx.input_shape = input_shape

        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        y = torch.empty(output_dims, device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x,
            weight,
            y,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = (
            ctx.ROWS_TILE_SIZE,
            ctx.D_TILE_SIZE,
        )  # These don't have to be the same
        n_rows, D = x.shape

        # Our strategy is for each thread block to first write to a partial buffer,
        # then we reduce over this buffer to get the final gradient.
        partial_grad_weight = torch.empty(
            (cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype
        )
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(cdiv(n_rows, ROWS_TILE_SIZE),)](
            x,
            weight,
            grad_out,
            grad_x,
            partial_grad_weight,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE,
            D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight


class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq = 16
        Bk = 16
        d = Q.shape[-1]
        Nq = Q.shape[-2]
        Nk = K.shape[-2]
        assert K.shape[-1] == d
        assert V.shape[-1] == d
        Tq = cdiv(Nq, Bq)
        Tk = cdiv(Nk, Bk)
        ctx.save_for_backward(Q, K, V)
        ctx.Bq = Bq
        ctx.Bk = Bk
        ctx.d = d
        ctx.Nq = Nq
        ctx.Nk = Nk
        ctx.Tq = Tq
        ctx.Tk = Tk
        ctx.is_causal = is_causal
        O = torch.empty_like(Q)  # (..., Nq, d)
        L = torch.empty_like(Q[..., 0])  # (..., Nq)
        for i in range(Tq):
            Qi = Q[..., i * Bq : (i + 1) * Bq, :]  # (..., Bq, d)
            Oi = torch.zeros_like(Qi)  # (..., Bq, d)
            li = torch.zeros_like(Qi[..., :1])  # (..., Bq, 1)
            mi = torch.full_like(Qi[..., :1], -float("inf"))  # (..., Bq, 1)
            for j in range(Tk):
                Kj = K[..., j * Bk : (j + 1) * Bk, :]  # (..., Bk, d)
                Vj = V[..., j * Bk : (j + 1) * Bk, :]  # (..., Bk, d)
                Sij = einops.einsum(
                    Qi, Kj, "... Bq d, ... Bk d -> ... Bq Bk"
                ) / math.sqrt(d)
                new_mi = torch.max(mi, Sij.max(-1, keepdims=True)[0])  # (..., Bq, 1)
                Pij = torch.exp(Sij - new_mi)  # (..., Bq, Bk)
                exp_diff = torch.exp(mi - new_mi)
                li = li * exp_diff + Pij.sum(-1, keepdims=True)  # (..., Bq, 1)
                Oi = Oi * exp_diff + einops.einsum(
                    Pij, Vj, "... Bq Bk, ... Bk d -> ... Bq d"
                )
                mi = new_mi  # (..., Bq, 1)
            Oi = Oi / li  # (..., Bq, d)
            Li = mi + torch.log(li)  # (..., Bq, 1)
            O[..., i * Bq : (i + 1) * Bq, :] = Oi
            L[..., i * Bq : (i + 1) * Bq] = Li[..., 0]
        ctx.save_for_backward(O, L)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError()


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (Bq, d)

    Oi = tl.zeros_like(Qi)  # (Bq, d)

    li = tl.zeros((Q_TILE_SIZE,), dtype=Qi.dtype)  # (Bq, )
    mi = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=Qi.dtype)  # (Bq, )
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(
            K_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (Bk, d)
        Vj = tl.load(
            V_block_ptr, boundary_check=(0, 1), padding_option="zero"
        )  # (Bk, d)

        Sij = tl.dot(Qi, Kj.T) * scale  # (Bq, Bk)
        Sij_max = tl.max(Sij, -1)
        new_mi = tl.where(mi >= Sij_max, mi, Sij_max)  # (Bq, )
        Pij = tl.exp(Sij.T - new_mi).T  # (Bq, Bk)
        exp_diff = tl.exp(mi - new_mi)  # (Bq,)
        li = li * exp_diff + tl.sum(Pij, -1)  # (Bq, )
        Oi = (Oi.T * exp_diff).T + tl.dot(Pij, Vj)
        mi = new_mi  # (Bq, )
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    Oi = (Oi.T / li).T  # (Bq, d)
    Li = mi + tl.log(li)  # (Bq, 1)
    tl.store(O_block_ptr, Oi, boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Q = Q.to(device)
        K = K.to(device)
        V = V.to(device)
        Bq = 16
        Bk = 16
        assert Q.ndim == 3
        batch_size = Q.shape[0]
        d = Q.shape[-1]
        Nq = Q.shape[-2]
        Nk = K.shape[-2]
        assert K.shape[-1] == d
        assert V.shape[-1] == d
        Tq = cdiv(Nq, Bq)
        Tk = cdiv(Nk, Bk)
        ctx.save_for_backward(Q, K, V)
        ctx.Bq = Bq
        ctx.Bk = Bk
        ctx.d = d
        ctx.Nq = Nq
        ctx.Nk = Nk
        ctx.Tq = Tq
        ctx.Tk = Tk
        ctx.is_causal = is_causal
        O = torch.empty_like(Q)  # (..., Nq, d)
        L = torch.empty_like(Q[..., 0])  # (..., Nq)
        flash_fwd_kernel[(Tq, batch_size)](
            Q,
            K,
            V,
            O,
            L,
            Q.stride(0),
            Q.stride(1),
            Q.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            V.stride(0),
            V.stride(1),
            V.stride(2),
            O.stride(0),
            O.stride(1),
            O.stride(2),
            L.stride(0),
            L.stride(1),
            Nq,
            Nk,
            1 / math.sqrt(d),
            d,
            Bq,
            Bk,
        )
        ctx.save_for_backward(O, L)
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--type")
    args = parser.parse_args()
    d_model = args.d_model
    seq_len = args.seq_len
    type = args.type
    print(
        f"\033[0;34mrunning d_model={d_model}, seq_len={seq_len}, {type}_attention\033[0m"
    )
    if type == "torch":
        torch_attention(d_model=d_model, seq_len=seq_len)
    elif type == "jit":
        jit_attention(d_model=d_model, seq_len=seq_len)
    else:
        raise
