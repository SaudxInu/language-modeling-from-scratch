import torch


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)
