import torch
import torch.nn as nn

from cs336_basics.nn.modules.rmsnorm import RMSNorm
from cs336_basics.nn.modules.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.nn.modules.positionwise_feedforward import SwiGLUFFN


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.rms_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.multihead_self_attention = MultiheadSelfAttention(
            d_model, num_heads, theta, max_seq_len, device, dtype
        )
        self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLUFFN(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        y = self.rms_norm_1(x)
        token_positions = torch.arange(n, device=self.device).unsqueeze(0).expand(b, n)
        x = x + self.multihead_self_attention(y, token_positions)
        z = self.rms_norm_2(x)
        x = x + self.ffn(z)
        return x
