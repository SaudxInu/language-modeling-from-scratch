import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        initial_value = torch.empty(
            (out_features, in_features), dtype=dtype, device=device
        )
        mean = 0
        std = (2 / (in_features + out_features)) ** 0.5
        initial_value = nn.init.trunc_normal_(
            initial_value,
            0,
            std,
            -3 * std,
            3 * std,
        )
        self.weight = nn.Parameter(initial_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return x
