import argparse
from contextlib import nullcontext
import os
from pathlib import Path

project_dir = Path(__file__).parent.parent

import einops
import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx
from tqdm import tqdm

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--seq_len", type=int)
    args = parser.parse_args()
    d_model = args.d_model
    seq_len = args.seq_len
    print(f"\033[0;34mrunning d_model={d_model}, seq_len={seq_len}\033[0m")
    torch_attention(d_model=d_model, seq_len=seq_len)
