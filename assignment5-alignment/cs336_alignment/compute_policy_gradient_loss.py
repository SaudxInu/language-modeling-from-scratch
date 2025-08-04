from typing import Literal

import torch

from cs336_alignment.compute_group_normalized_rewards import (
    compute_group_normalized_rewards,
)
from cs336_alignment.compute_grpo_clip_loss import compute_grpo_clip_loss
from cs336_alignment.compute_naive_policy_gradient_loss import (
    compute_naive_policy_gradient_loss,
)


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal[
        "no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"
    ],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        advantages = raw_rewards
    if loss_type == "grpo_clip":
        loss, stats = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    elif loss_type == "grpo_no_clip":
        loss = -(torch.exp(policy_log_probs) / torch.exp(old_log_probs)) * advantages
    else:
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        stats = {}
    return loss, stats
