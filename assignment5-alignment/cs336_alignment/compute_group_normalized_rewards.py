import torch


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    rewards = [
        reward_fn(rollout_response, repeated_ground_truth)
        for rollout_response, repeated_ground_truth in zip(
            rollout_responses, repeated_ground_truths
        )
    ]
    rewards = torch.tensor([reward["reward"] for reward in rewards]).reshape(
        -1, group_size
    )
    mean_reward = torch.mean(rewards, dim=1, keepdim=True)
    normalized_rewards = rewards - mean_reward
    if normalize_by_std:
        std_reward = torch.std(rewards, dim=1, keepdim=True)
        normalized_rewards /= std_reward + advantage_eps
    return normalized_rewards.reshape(-1), rewards.reshape(-1), {}
