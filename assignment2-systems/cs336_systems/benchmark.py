from contextlib import nullcontext
import os
from pathlib import Path
import sys
import timeit

project_dir = Path(os.path.abspath("")).parent
basics_path = (project_dir / "cs336-basics").as_posix()
if sys.path[0] != basics_path:
    sys.path.insert(0, basics_path)

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cs336_basics.model import BasicsTransformerLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_configs = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12}, # fmt:skip
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16}, # fmt:skip
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20}, # fmt:skip
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25}, # fmt:skip
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}, # fmt:skip
}

def benchmark_model(
    model_name,
    context_length: int,
    warmup_steps: int,
    exe_steps,
    only_forward: bool,
    dtype=None,
):
    batch_size = 4
    vocab_size = 10000
    rope_theta = 10000
    x = torch.randint(0, vocab_size, (batch_size, context_length)).to(device)
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        rope_theta=rope_theta,
        **model_configs[model_name],
    )
    param_count = 0
    for p in model.parameters():
        param_count += p.numel()
    model.to(device)

    if dtype is None:
        context = nullcontext()
    elif dtype=='float16':
        context = torch.autocast(device_type="cuda", dtype=torch.float16)
    elif dtype=='bfloat16':
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        raise
    with context:
        for _ in range(warmup_steps):
            logits = model(x)
            if not only_forward:
                loss = logits.sum()
                loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = timeit.default_timer()
        ts = []
        for _ in range(exe_steps):
            logits = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if not only_forward:
                loss = logits.sum()
                loss.backward()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        ts.append(timeit.default_timer() - start)
    ts = np.diff(ts, prepend=0)
    del x, model
    return ts, param_count

def run():
    benchmark_model('small', context_length=64, warmup_steps=5, exe_steps=10, only_forward=True, dtype=None)


if __name__ == "__main__":
    run()
