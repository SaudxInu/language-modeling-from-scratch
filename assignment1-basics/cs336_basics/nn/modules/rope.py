import torch
import torch.nn as nn
from einops import einsum, rearrange


class RoPE(nn.Module):
    def __init__(
        self,
        base: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super(RoPE, self).__init__()
        self.base = base
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.d_k, 2, device=self.device)[
                    : (self.d_k // 2)
                ].float()
                / self.d_k
            )
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(
            self.max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        idx_theta = einsum(seq_idx, self.theta, "max_seq_len, d -> max_seq_len d")
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)
        token_positions = rearrange(token_positions, "... seq_len -> (... seq_len)")
        rope_cache = self.cache[token_positions]
        rope_cache = rearrange(
            rope_cache,
            "(batch_size seq_len) ... -> batch_size 1 seq_len ...",
            seq_len=seq_len,
        )
        x = rearrange(x, "... seq_len (d d2) -> ... seq_len d d2", d2=2)
        x = torch.stack(
            [
                x[..., 0] * rope_cache[..., 0] - x[..., 1] * rope_cache[..., 1],
                x[..., 0] * rope_cache[..., 1] + x[..., 1] * rope_cache[..., 0],
            ],
            -1,
        )
        x = rearrange(x, "... seq_len d d2 -> ... seq_len (d d2)")
        return x
