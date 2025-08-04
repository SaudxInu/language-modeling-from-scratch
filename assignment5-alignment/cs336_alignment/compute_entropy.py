import torch


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -torch.sum(log_probs * torch.exp(log_probs), dim=-1)
