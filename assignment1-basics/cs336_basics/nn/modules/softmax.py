import torch
import torch.nn as nn


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    x = torch.exp(x)
    x = x / torch.sum(x, dim=dim, keepdim=True)
    return x
