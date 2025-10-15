import math

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
        cos_theta = self.cos_theta[token_positions]  # (seq_len, d_k)
        sin_theta = self.sin_theta[token_positions]  # (seq_len, d_k)
        x1 = torch.concat((-x[..., 1::2, None], x[..., ::2, None]), dim=-1)
        x1 = x1.reshape(*x.shape)
        return x * cos_theta + x1 * sin_theta


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = None,
        Theta=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads

        self.weight = nn.Parameter(
            torch.empty((3, num_heads, self.d_k, d_model), **factory_kwargs)
        )  # store W_Q, W_K, W_V together
        self.wo = Linear(num_heads * self.d_v, d_model, device=device, dtype=dtype)
        if max_seq_len is not None:
            self.rope = RoPE(Theta, self.d_k, max_seq_len)
        else:
            self.register_parameter("rope", None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 2.0 / sum(self.weight.shape[-2:])
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor = None
    ) -> torch.Tensor:
        seq_len = x.shape[-2]
        QKV = einops.einsum(
            x,
            self.weight,
            "... seq_len d_model, QKV num_heads d_k d_model -> ... QKV num_heads seq_len d_k",
        )
        QK = QKV[..., 0:2, :, :, :]  # (..., 2, num_heads, seq_len, d_k)
        if self.rope is not None:
            QK = self.rope(QK, token_positions)
        Q = QK[..., 0, :, :, :]  # (..., num_heads, seq_len, d_k)
        K = QK[..., 1, :, :, :]  # (..., num_heads, seq_len, d_k)
        V = QKV[..., 2, :, :, :]  # (..., num_heads, seq_len, d_v)
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=bool, device=x.device))
        multi_head = scaled_dot_product_attention(
            Q, K, V, mask
        )  # (..., num_heads, seq_len, d_v)
        multi_head = einops.rearrange(
            # multi_head, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)"
            multi_head,
            "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)",
        )
        a = self.wo(multi_head)
        return a


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len=None,
        Theta=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.norm1 = RMSNorm(d_model, **factory_kwargs)
        self.multi_head = MultiheadSelfAttention(
            d_model, num_heads, max_seq_len, Theta, **factory_kwargs
        )
        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)
        self.norm2 = RMSNorm(d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_head.rope is not None:
            token_positions = torch.arange(
                x.shape[-2], device=x.device, dtype=torch.long
            )
        else:
            token_positions = None
        a = self.multi_head(self.norm1(x), token_positions)
        x = x + a
        a = self.ffn(self.norm2(x))
        x = x + a
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        max_seq_len=None,
        Theta=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)
        layers = []
        for _ in range(num_layers):
            layers.append(
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len,
                    Theta,
                    **factory_kwargs,
                )
            )
        self.layers = nn.Sequential(*layers)
        self.ln_final = RMSNorm(d_model, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        x = self.layers(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


def softmax(x: torch.Tensor, dim: int, mask=None):
    x = torch.exp(x - x.max(dim, keepdim=True)[0])
    if mask is not None:
        mask = ~mask
        x = x.masked_fill_(mask, 0.0)
    x = x / (x.sum(dim, keepdim=True))
    if mask is not None:
        x = x.masked_fill_(mask, 0.0)
    return x


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None
):
    """
    Parameters
    ------
    Q, K: (..., seq_len, d_k)
    V: (..., seq_len, d_v)
    mask: (seq_len, seq_len)

    Return
    ------
    attention: (..., seq_len, d_v)
    """
    d_k = K.shape[-1]
    score_scaled = einops.einsum(
        Q, K, "... seq_len_q d_k, ... seq_len_k d_k -> ... seq_len_q seq_len_k"
    ) / math.sqrt(d_k)
    attention_weight = softmax(score_scaled, -1, mask=mask)
    attention = einops.einsum(
        attention_weight,
        V,
        "... seq_len_q seq_len_k, ... seq_len_k d_v -> ... seq_len_q d_v",
    )
    return attention
