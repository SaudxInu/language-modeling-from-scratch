import torch
import torch.nn as nn
from einops import einsum

from .softmax import softmax


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    x = einsum(
        q, k, "... seq_len_1 d_k, ... seq_len_2 d_k -> ... seq_len_1 seq_len_2"
    ) / (q.shape[-1] ** 0.5)
    y = torch.zeros(mask.shape)
    y[~mask] = -float("inf")
    x = x + y
    x = softmax(x, -1)
    x = einsum(x, v, "... seq_len_1 seq_len_2, ... seq_len_2 d_v -> ... seq_len_1 d_v")
    return x
