import torch
from vllm import LLM, SamplingParams

from cs336_alignment.compute_entropy import compute_entropy
from cs336_alignment.math_baseline import evaluate_vllm
from drgrpo_grader import r1_zero_reward_fn


def log_generations(
    llm: LLM,
    prompt_strs: list[str],
    output_strs: list[str],
):
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        min_tokens=4,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=20,
    )
    responses, rewards, logprobs = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompt_strs,
        output_strs,
        sampling_params,
    )
    gen_strs = responses
    rewards = [
        r1_zero_reward_fn(response, ground_truth)
        for response, ground_truth in zip(gen_strs, output_strs)
    ]
    avg_token_entropy = compute_entropy(logprobs, True).mean().item()
    len_correct_responses = [
        len(gen_str)
        for reward, gen_str in zip(rewards, gen_strs)
        if reward.get("reward", 0) == 1
    ]
    len_incorrect_responses = [
        len(gen_str)
        for reward, gen_str in zip(rewards, gen_strs)
        if reward.get("reward", 0) == 0
    ]
    avg_len_correct_responses = (
        sum(len_correct_responses) / len(len_correct_responses)
        if len_correct_responses
        else 0
    )
    avg_len_incorrect_responses = (
        sum(len_incorrect_responses) / len(len_incorrect_responses)
        if len_incorrect_responses
        else 0
    )
    return {
        "ground_texts": output_strs,
        "generated_texts": gen_strs,
        "rewards": rewards,
        "avg_token_entropy": avg_token_entropy,
        "avg_len_correct_responses": avg_len_correct_responses,
        "avg_len_incorrect_responses": avg_len_incorrect_responses,
    }
