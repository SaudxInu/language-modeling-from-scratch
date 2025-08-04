import torch


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    probs_ratio = torch.exp(policy_log_probs) / torch.exp(old_log_probs)
    probs_ratio_clipped = torch.clamp(probs_ratio, 1 - cliprange, 1 + cliprange)
    return (
        -torch.minimum(probs_ratio * advantages, probs_ratio_clipped * advantages),
        {},
    )
