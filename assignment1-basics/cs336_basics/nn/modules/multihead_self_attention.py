import torch
import torch.nn as nn
from einops import rearrange

from cs336_basics.nn.modules.linear import Linear
from cs336_basics.nn.modules.rope import RoPE
from cs336_basics.nn.modules.scaled_dot_product_attention import (
    scaled_dot_product_attention,
)


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(MultiheadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        if theta and max_seq_len:
            self.rope = RoPE(theta, d_model // num_heads, max_seq_len, device)
        self.w = Linear(d_model, 3 * d_model, device, dtype)
        self.o = Linear(d_model, d_model, device, dtype)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.w(x)
        q = x[..., : self.d_model]
        k = x[..., self.d_model : 2 * self.d_model]
        v = x[..., 2 * self.d_model : 3 * self.d_model]
        q = rearrange(
            q,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        k = rearrange(
            k,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        v = rearrange(
            v,
            "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k",
            num_heads=self.num_heads,
        )
        mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2]).bool()).T
        mask = rearrange(mask, "seq_len_1 seq_len_2 -> 1 1 seq_len_1 seq_len_2")
        if self.theta and self.max_seq_len:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        x = scaled_dot_product_attention(q, k, v, mask)
        x = rearrange(x, "... num_heads seq_len d_k -> ... seq_len (num_heads d_k)")
        x = self.o(x)
        return x
