# uv run cs336_basics/nn/modules/transformer_lm.py
# >> Trainable parameters: 2.1 B
# >> Model size: 8 GiB
# FFN requires the most FLOPs.
# FFN take up proportionally more of total FLOPs as model size increases.
# As context length increases the total FLOPs for one forward pass should increase linearly
# but relative contribution of FLOPs of the model components should remain same.
import torch
import torch.nn as nn

from cs336_basics.nn.modules.embedding import Embedding
from cs336_basics.nn.modules.transformer_block import TransformerBlock
from cs336_basics.nn.modules.rmsnorm import RMSNorm
from cs336_basics.nn.modules.linear import Linear
from cs336_basics.nn.modules.softmax import softmax


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.embedding = Embedding(
            vocab_size,
            d_model,
            device,
            dtype,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, num_heads, d_ff, theta, context_length, device, dtype
                )
                for _ in range(num_layers)
            ]
        )
        self.rms_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.o = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[..., : self.context_length]
        x = self.embedding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.rms_norm(x)
        x = self.o(x)
        return x


if __name__ == "__main__":
    model = TransformerLM(50257, 1024, 48, 1600, 25, 6400, 10000.0)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable}")
    print(f"Model size: {(trainable * 32) / (8 * 1024 ** 2)} MiB")
