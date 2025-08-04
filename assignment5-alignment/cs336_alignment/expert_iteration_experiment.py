import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.log_generations import log_generations
from cs336_alignment.sft_experiment import (
    run,
    init_vllm,
    load_policy_into_vllm_instance,
)


def run_expert_iteration_experiment():
    training_data = pd.read_json("data/math/training.jsonl", lines=True)
    validation_data = pd.read_json("data/math/validation.jsonl", lines=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B").to("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    vllm = init_vllm("cuda:0", seed=42)
    load_policy_into_vllm_instance(model, vllm)
    for _ in range(5):
        db = training_data.sample(n=512)
        db_results = log_generations(
            vllm,
            db["prompt"].tolist(),
            db["response"].tolist(),
        )
        db["response_pred"] = db_results["generated_texts"]
        db["format_reward"] = [
            reward.get("format_reward", 0) for reward in db_results["rewards"]
        ]
        db["answer_reward"] = [
            reward.get("answer_reward", 0) for reward in db_results["rewards"]
        ]
        db["reward"] = [reward.get("reward", 0) for reward in db_results["rewards"]]
        dsft = db[db["reward"] == 1]
        run(model, tokenizer, dsft, validation_data, vllm)
