import torch


def gradient_clipping(parameters, maximum_norm):
    norm = torch.norm(torch.stack([p.grad for p in parameters if p.grad is not None]))

    if norm > maximum_norm:
        scaling_factor = maximum_norm / (norm + 1e-6)

        for p in parameters:
            if p.grad is None:
                continue

            p.grad.data *= scaling_factor
