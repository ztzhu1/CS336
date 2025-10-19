import math
import os
from pathlib import Path
from typing import IO, BinaryIO, Union

import einops
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm, trange

from . import nn_336


def cross_entropy(inputs, targets):
    """
    inputs (Float[Tensor, "batch_size vocab_size"])
    targets (Int[Tensor, "batch_size"])
    """
    max_x = inputs.max(-1, keepdim=True)[0]
    denominator = torch.exp(inputs - max_x).sum(-1, keepdim=True)
    targets = targets.unsqueeze(-1)
    inputs = torch.gather(inputs, -1, targets)
    loss = -((inputs - max_x[..., 0]) - torch.log(denominator))
    return loss.mean()


class AdamW(Optimizer):
    def __init__(
        self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                t = state.get("t", 1)
                if t == 1:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m = state["m"]
                v = state["v"]
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                state["m"] = m
                state["v"] = v
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t + 1

        return loss


def get_lr_cosine_schedule(t, alphas, T_w, T_c):
    alpha_min, alpha_max = alphas
    if t < T_w:
        return alpha_max * t / T_w
    elif T_w <= t <= T_c:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (
            1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))
        )
    else:
        return alpha_min


def gradient_clipping(parameters, max_l2_norm):
    eps = 1e-6
    norm_square = 0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        norm_square += param_norm**2
    clip_coef = max_l2_norm / (math.sqrt(norm_square) + eps)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data *= clip_coef


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    pass


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]],
):
    obj = {
        "model_state_dict": model.state_dict(),
        "opt_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    optimizer: Optimizer,
):
    obj = torch.load(src)
    model.load_state_dict(obj["model_state_dict"])
    optimizer.load_state_dict(obj["opt_state_dict"])
    iteration = obj["iteration"]
    return iteration


def load_data(x, batch_size, context_length, device):
    N = len(x)
    max_index = N - context_length - 1
    assert max_index >= 0
    indexes = np.random.randint(0, max_index + 1, (batch_size,))[:, None]
    offset = np.arange(context_length)
    indexes = indexes + offset
    seq1 = torch.LongTensor(x[indexes], device=device)
    seq2 = torch.LongTensor(x[indexes + 1], device=device)
    return seq1, seq2


def train(
    epochs, batch_size, model_kwargs, opt_kwargs, checkpoint_dir=None, save_every=10
):
    model = nn_336.TransformerLM(**model_kwargs)
    optimizer = AdamW(model.parameters(), **opt_kwargs)
    losses = []
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in trange(epochs):
        epoch += 1
        running_loss = 0.0
        batch_progress = tqdm(dataloader, desc="Batches", leave=False)

        for iter, (images, labels) in enumerate(batch_progress):
            images = images.to(device)
            tgt = images.clone()
            pred = model(images)
            loss = bce(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            losses.append(loss.item())
        avg_loss = running_loss * batch_size / len(train_dataset)
        if epoch % save_every == 0 and checkpoint_dir is not None:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                (checkpoint_dir / f"checkpoint_epoch_{epoch}.pt").as_posix(),
            )

        tqdm.write(f"----\nEpoch [{epoch}/{epochs}], Average Loss: {avg_loss:.4f}\n")
