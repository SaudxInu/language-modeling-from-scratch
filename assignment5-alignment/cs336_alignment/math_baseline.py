from typing import Callable, List

import torch
from vllm import LLM, SamplingParams
import pandas as pd

from drgrpo_grader import r1_zero_reward_fn

from torch.nn.utils.rnn import pad_sequence


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    rewards = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(responses, ground_truths)
    ]
    logprobs_raw = [output.outputs[0].logprobs for output in outputs]
    logprobs = []
    for sequence in logprobs_raw:
        temp_sequence = []
        for token_logprobs in sequence:
            token_probs = [entry.logprob for entry in token_logprobs.values()][:20]
            temp_sequence.append(token_probs)
        logprobs.append(torch.tensor(temp_sequence))
    logprobs = pad_sequence(logprobs, batch_first=True, padding_value=0)
    return responses, rewards, logprobs


def main():
    df = pd.read_json("data/math/validation.jsonl", lines=True)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()
    df["prompt"] = df["problem"].apply(
        lambda x: str(prompt_template).format(question=x)
    )
    responses, rewards, _ = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        df["prompt"].tolist(),
        df["solution"].tolist(),
        sampling_params,
    )
    df["response"] = responses
    df["format_reward"] = [reward.get("format_reward", 0) for reward in rewards]
    df["answer_reward"] = [reward.get("answer_reward", 0) for reward in rewards]
    df["reward"] = [reward.get("reward", 0) for reward in rewards]
    print("=" * 20)
    print("MATH Baseline Results")
    print("=" * 20)
    print(df[["format_reward", "answer_reward", "reward"]].sum(axis=0) / len(df))
    print("=" * 20)
    df.to_csv("data/math_baseline.csv", index=False)


if __name__ == "__main__":
    main()

# ====================
# MATH Baseline Results
# ====================
# format_reward        0.1724
# answer_reward        0.0278
# reward               0.0278

# F == 1 and A == 1    0.0278
# F == 1 and A == 0    0.1446
# F == 0 and A == 0    0.8276

# Parser Error Cases

# </think> \n <answer> or directly starting with <answer> or not having </think>.

# {-4, -2, 0, 2, 4}
# vs
# {+-4, 0, +-2}

# There are $\\binom{20}{2} = 190$ ways to choose two members of the group. There are $12$ ways to choose a boy and $8$ ways to choose a girl for a total of $12 \\cdot 8 = 96$ ways to choose a boy and a girl.  This means that there is a $\\dfrac{96}{190} = \\boxed{\\dfrac{48}{95}}$ chance that the two random members of the group are a boy and a girl.
# vs
#  The club has 20 members, and two are chosen at random. We need to find the probability that one of the chosen members is a boy and the other is a girl. We can approach this by first calculating the total number of ways to choose 2 members out of 20, and then calculating the number of ways to choose 1 boy out of 12 and 1 girl out of 8. The probability is then the ratio of these two quantities. </think> <answer> \nThe total number of ways to choose 2 members out of 20 is given by the combination formula C(20,2), which is equal to 20! / (2! * (20-2)!). This simplifies to 20 * 19 / 2 = 190. The number of ways to choose 1 boy out of 12 and 1 girl out of 8 is given by the product of the two combinations C(12,1) * C(8,1), which is equal to 12 * 8 = 96. Therefore, the probability of choosing one boy and one girl is 96 / 190 = 48 / 95.

# ====================
