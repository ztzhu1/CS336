import einops
import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        assert not bias  # following most LLMs
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 2.0 / sum(self.weight.shape)
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(
            x, self.weight, "... in_feat, out_feat in_feat -> ... out_feat"
        )

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

    def extra_repr(self) -> str:
        return "num_embeddings={}, embedding_dim={}".format(
            self.num_embeddings, self.embedding_dim
        )


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, sequence_length, d_model)
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS = torch.sqrt((x**2).mean(-1, keepdim=True) + self.eps)
        x = x / RMS * self.weight
        return x.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.w1(x)
        y3 = self.w3(x)
        return self.w2(self.silu(y1) * y3)


class RoPE(nn.Module):
    def __init__(self, Theta: float, d_k: int, max_seq_len: int, device=None):
        assert d_k % 2 == 0
        super().__init__()
        self.Theta = Theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        k = torch.arange(1.0, d_k // 2 + 0.5, device=device)[None]  # (1, d_k/2)
        i = torch.arange(0.0, max_seq_len, device=device)[:, None]  # (max_seq_len, 1)
        theta = i / (Theta ** ((2.0 * k - 2.0) / d_k))  # (max_seq_len, d_k/2)
        self.register_buffer(
            "cos_theta", torch.cos(theta).repeat_interleave(2, dim=-1), persistent=False
        )
        self.register_buffer(
            "sin_theta", torch.sin(theta).repeat_interleave(2, dim=-1), persistent=False
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        """
        cos_theta = self.cos_theta[token_positions]  # (seq_len, d_k/2)
        sin_theta = self.sin_theta[token_positions]  # (seq_len, d_k/2)
        x1 = torch.concat((-x[..., 1::2, None], x[..., ::2, None]), dim=-1)
        x1 = x1.reshape(*x.shape)
        return x * cos_theta + x1 * sin_theta
