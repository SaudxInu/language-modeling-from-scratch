from typing import Callable, List
import glob
import os

from vllm import LLM, SamplingParams
import pandas as pd
import torch


def evaluate_vllm(
    vllm_model: LLM,
    prompts: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    return responses


def main(model_name):
    df = pd.read_json("data/alpaca_eval/alpaca_eval.jsonl", lines=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        include_stop_str_in_output=True,
    )
    llm = LLM(model=model_name)
    prompt_template = "{instruction}"
    df["prompt"] = df.apply(
        lambda x: str(prompt_template).format(instruction=x["instruction"]),
        axis=1,
    )
    responses = evaluate_vllm(
        llm,
        df["prompt"].tolist(),
        sampling_params,
    )
    df["output"] = responses
    df["generator"] = model_name
    df["dataset"] = "alpaca_eval"
    df.to_json("data/alpaca_eval_baseline.json", orient="records")


if __name__ == "__main__":
    main(model_name="meta-llama/Llama-3.1-8B")
    # main(model_name="Llama-3.1-8B-Instruct")
    # main(model_name="Llama-3.1-8B-DPO")
