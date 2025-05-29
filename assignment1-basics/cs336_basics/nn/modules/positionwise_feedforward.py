import math

import torch
import torch.nn as nn
from einops import einsum

from cs336_basics.nn.modules.linear import Linear


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(SwiGLUFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff else int(math.ceil((8 * d_model) / (3 * 64)) * 64)
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(self.d_model, self.d_ff, device=device)
        self.w2 = Linear(self.d_ff, self.d_model, device=device)
        self.w3 = Linear(self.d_model, self.d_ff, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        a = a * torch.sigmoid(a)
        b = self.w3(x)
        c = einsum(a, b, "... d_ff, ... d_ff -> ... d_ff")
        d = self.w2(c)
        return d
