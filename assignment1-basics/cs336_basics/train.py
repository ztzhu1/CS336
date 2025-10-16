import math

import einops
import torch
from torch import nn
from torch.optim import Optimizer


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
