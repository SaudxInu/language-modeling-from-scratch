from typing import Callable, List
import glob
import os

from vllm import LLM, SamplingParams
import pandas as pd
import torch


def gsm8k_baseline(x: str) -> str | None:
    try:
        res = [seg for seg in x.split(" ") if seg.isdigit()][-1]
    except Exception as _:
        res = None
    return res


def evaluate_vllm(
    vllm_model: LLM,
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    rewards = [
        int(gsm8k_baseline(ground_truth) == gsm8k_baseline(response))
        for response, ground_truth in zip(responses, ground_truths)
    ]
    return responses, rewards


def main(model_name):
    df = pd.read_json("data/gsm8k/test.jsonl", lines=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        include_stop_str_in_output=True,
    )
    llm = LLM(model=model_name)
    prompt_template = "{question}\nAnswer:"
    df["prompt"] = df.apply(
        lambda x: str(prompt_template).format(question=x["question"]),
        axis=1,
    )
    responses, rewards = evaluate_vllm(
        llm,
        df["prompt"].tolist(),
        df["answer"].tolist(),
        sampling_params,
    )
    df["response"] = responses
    df["reward"] = rewards
    print("=" * 20)
    print("GSM8K Baseline Results")
    print("=" * 20)
    print(df[["reward"]].sum(axis=0) / len(df))
    print("=" * 20)
    df.to_csv("data/gsm8k_baseline.csv", index=False)


if __name__ == "__main__":
    main(model_name="meta-llama/Llama-3.1-8B")
    # main(model_name="Llama-3.1-8B-Instruct")
