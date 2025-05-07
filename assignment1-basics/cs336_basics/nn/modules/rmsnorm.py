import torch
import torch.nn as nn
from einops import einsum


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        initial_value = torch.ones((d_model,), dtype=dtype, device=device)
        self.g = nn.Parameter(initial_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        numerator = einsum(x, self.g, "... d_model, d_model -> ... d_model")
        denominator = (torch.sum(x**2, axis=-1) / self.d_model + self.eps) ** 0.5
        x = einsum(numerator, 1 / denominator, "... d_model, ... -> ... d_model")
        return x.to(in_dtype)
