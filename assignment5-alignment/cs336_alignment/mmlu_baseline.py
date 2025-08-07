from typing import Callable, List
import glob
import os

from vllm import LLM, SamplingParams
import pandas as pd
import torch


def mmlu_baseline(y: dict, x: str) -> str | None:
    idx = x.find("The correct answer is ")
    if idx == -1:
        return None
    idx += len("The correct answer is ")
    x = x[idx : idx + 1]
    if x is None or x == "":
        return None
    if x not in "ABCD":
        return None
    return x


def evaluate_vllm(
    vllm_model: LLM,
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    rewards = [
        int(ground_truth == mmlu_baseline({}, response))
        for response, ground_truth in zip(responses, ground_truths)
    ]
    return responses, rewards


def main(model_name):
    csv_files = glob.glob(os.path.join("data/mmlu/val", "*.csv"))
    dfs = []
    for file in csv_files:
        df = pd.read_csv(
            file,
            header=None,
            columns=[
                "problem",
                "option_1",
                "option_2",
                "option_3",
                "option_4",
                "answer",
            ],
        )
        df["subject"] = (
            file.split("/")[-1].split(".")[0].removesuffix("_val").replace("_", " ")
        )
        dfs.append(df)
    df = pd.concat(
        dfs,
        ignore_index=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        include_stop_str_in_output=True,
    )
    llm = LLM(model=model_name)
    prompt_template = (
        "Answer the following multiple choice question about {subject}. "
        'Respond with a single sentence of the form "The correct answer is _", '
        "filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).\n"
        "\n"
        "Question: {question}\n"
        "A. {option_1}\n"
        "B. {option_2}\n"
        "C. {option_3}\n"
        "D. {option_4}\n"
        "Answer:"
    )
    df["prompt"] = df.apply(
        lambda x: str(prompt_template).format(
            question=x["question"],
            option_1=x["option_1"],
            option_2=x["option_2"],
            option_3=x["option_3"],
            option_4=x["option_4"],
            subject=x["subject"],
        ),
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
    print("MMLU Baseline Results")
    print("=" * 20)
    print(df[["reward"]].sum(axis=0) / len(df))
    print("=" * 20)
    df.to_csv("data/mmlu_baseline.csv", index=False)


if __name__ == "__main__":
    main(model_name="meta-llama/Llama-3.1-8B")
    # main(model_name="Llama-3.1-8B-Instruct")
