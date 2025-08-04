import torch

from cs336_alignment.masked_normalize import masked_normalize


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = (
        -1
        * masked_normalize(
            policy_log_probs, response_mask, normalize_constant, dim=-1
        ).mean()
        / gradient_accumulation_steps
    )
    loss.backward()
    return loss.detach(), {}
