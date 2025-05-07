import torch
import torch.nn as nn
from einops import rearrange


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        initial_value = torch.empty(
            (num_embeddings, embedding_dim), dtype=dtype, device=device
        )
        initial_value = nn.init.trunc_normal_(
            initial_value,
            0,
            1,
            -3,
            3,
        )
        self.embeddings = nn.Parameter(initial_value)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            token_ids,
            f"batch_size sequence_length -> (batch_size sequence_length)",
        )
        x = self.embeddings[x]
        x = rearrange(
            x,
            f"(batch_size sequence_length) d_embed -> batch_size sequence_length d_embed",
            batch_size=token_ids.shape[0],
        )
        return x
