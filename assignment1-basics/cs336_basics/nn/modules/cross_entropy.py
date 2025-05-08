import torch
import torch.nn as nn


def cross_entropy(
    predicted_logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    x = predicted_logits - torch.max(predicted_logits, dim=-1, keepdim=True)[0]
    x = torch.log(torch.sum(torch.exp(x), dim=-1, keepdim=True)) - torch.gather(
        x, -1, targets.unsqueeze(1)
    )
    loss = torch.mean(x)
    return loss
