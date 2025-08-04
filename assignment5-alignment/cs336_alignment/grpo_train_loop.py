import os
from typing import Literal

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import pandas as pd
import torch
import wandb

from cs336_alignment.compute_group_normalized_rewards import (
    compute_group_normalized_rewards,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.grpo_microbatch_train_step import grpo_microbatch_train_step
from cs336_alignment.log_generations import log_generations
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output


os.environ["WANDB_PROJECT"] = "Qwen2.5-Math-1.5B-GRPO"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def init_vllm(device: str, seed: int, gpu_memory_utilization: float = 0.85):
    vllm_set_random_seed(seed)
    return LLM(
        model="Qwen/Qwen2.5-Math-1.5B",
        device=device,
        dtype=torch.bfloat16,
        enable_prefix_caching=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


class TrainingDataset(Dataset):
    def __init__(self, df):
        self.prompts = df["prompt"].to_list()
        self.responses = df["response"].to_list()

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        return prompt, response


def run(policy, tokenizer, training_data, validation_data, vllm):
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    rollout_batch_size: int = 256
    group_size: int = 8
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4
    sampling_max_tokens: int = 1024
    epochs_per_rollout_batch: int = 1
    train_batch_size: int = 256
    gradient_accumulation_steps: int = 128
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    max_tokens = 2048
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size
    train_dataset = TrainingDataset(training_data)
    train_dataloader = DataLoader(
        train_dataset, batch_size=rollout_batch_size, shuffle=True
    )
    print("Starting training...")
    wandb.init()
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    batch_loss = 0.0
    batch_token_entropy = 0.0
    k = 0
    for _ in range(n_grpo_steps):
        for idx, batch in enumerate(train_dataloader):
            load_policy_into_vllm_instance(policy, vllm)
            prompts, responses = batch
            temp_responses = [[] for _ in range(len(input_ids))]
            for _ in range(group_size):
                sampling_params = SamplingParams(
                    temperature=sampling_temperature,
                    top_p=1.0,
                    max_tokens=sampling_max_tokens,
                    min_tokens=sampling_min_tokens,
                    stop=["</answer>"],
                    include_stop_str_in_output=True,
                )
                outputs = vllm.generate(prompts, sampling_params)
                for i, output in enumerate(outputs):
                    temp_responses[i].append(output.outputs[0].text)
            rollout_responses = []
            repeated_ground_truths = []
            repeated_prompts = []
            for i, t in enumerate(temp_responses):
                rollout_responses += t
                repeated_ground_truths += [responses[i]] * group_size
                repeated_prompts += [prompts[i]] * group_size
            tokenized_data = tokenize_prompt_and_output(
                repeated_prompts, rollout_responses, tokenizer
            )
            input_ids = tokenized_data["input_ids"].to("cuda:1")[:, :max_tokens]
            labels = tokenized_data["labels"].to("cuda:1")[:, :max_tokens]
            response_mask = tokenized_data["response_mask"].to("cuda:1")[:, :max_tokens]
            policy.eval()
            with torch.no_grad():
                old_log_probs = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=True,
                )["log_probs"]
            normalized_rewards, raw_rewards = compute_group_normalized_rewards(
                r1_zero_reward_fn,
                rollout_responses,
                repeated_ground_truths,
                group_size,
                advantage_eps,
                use_std_normalization,
            )
            policy.train()
            for _ in range(epochs_per_rollout_batch):
                res = get_response_log_probs(
                    model=policy,
                    input_ids=input_ids,
                    labels=labels,
                    return_token_entropy=True,
                )
                log_probs = res["log_probs"]
                token_entropy = res["token_entropy"].mean()
                loss, _ = grpo_microbatch_train_step(
                    log_probs,
                    response_mask,
                    gradient_accumulation_steps,
                    loss_type,
                    raw_rewards=raw_rewards,
                    advantages=normalized_rewards,
                    old_log_probs=old_log_probs,
                    cliprange=None,
                )
                current_loss = loss.cpu().item()
                current_token_entropy = (
                    token_entropy.detach().cpu().item() / gradient_accumulation_steps
                )
                batch_loss += current_loss
                batch_token_entropy += current_token_entropy
                print(
                    f"  Micro Batch {idx + 1}, Loss: {current_loss}, Token Entropy: {current_token_entropy}"
                )
                del (
                    res,
                    log_probs,
                    token_entropy,
                    loss,
                )
                torch.cuda.empty_cache()
                if (
                    (idx * n_microbatches_per_rollout_batch) + 1
                ) % gradient_accumulation_steps == 0:
                    clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    print(
                        f"  Batch {(idx + 1) // gradient_accumulation_steps}, Loss: {batch_loss}, Token Entropy: {batch_token_entropy}"
                    )
                    wandb.log(
                        {
                            "train_step": k,
                            "train/loss": batch_loss,
                            "train/avg_token_entropy": batch_token_entropy,
                        }
                    )
                    batch_loss = 0.0
                    batch_token_entropy = 0.0
                    validation_sample = validation_data.sample(n=100)
                    with open("cs336_alignment/prompts/r1_zero.prompt", "r") as f:
                        prompt_template = f.read()
                    validation_sample["prompt"] = validation_sample["problem"].apply(
                        lambda x: str(prompt_template).format(question=x)
                    )
                    load_policy_into_vllm_instance(policy, vllm)
                    eval_results = log_generations(
                        vllm,
                        validation_sample["prompt"].tolist(),
                        validation_sample["solution"].tolist(),
                    )
                    validation_sample["response"] = eval_results["generated_texts"]
                    validation_sample["format_reward"] = [
                        reward.get("format_reward", 0)
                        for reward in eval_results["rewards"]
                    ]
                    validation_sample["answer_reward"] = [
                        reward.get("answer_reward", 0)
                        for reward in eval_results["rewards"]
                    ]
                    validation_sample["reward"] = [
                        reward.get("reward", 0) for reward in eval_results["rewards"]
                    ]
                    validation_stats = validation_sample[
                        ["format_reward", "answer_reward", "reward"]
                    ].sum(axis=0) / len(validation_sample)
                    wandb.log(
                        {
                            "eval_step": k,
                            "eval/format_reward": validation_stats["format_reward"],
                            "eval/answer_reward": validation_stats["answer_reward"],
                            "eval/reward": validation_stats["reward"],
                            "eval/avg_token_entropy": eval_results["avg_token_entropy"],
                            "eval/avg_len_correct_responses": eval_results[
                                "avg_len_correct_responses"
                            ],
                            "eval/avg_len_incorrect_responses": eval_results[
                                "avg_len_incorrect_responses"
                            ],
                        }
                    )
                    print("=" * 20)
                    for _, row in validation_sample.sample(n=3).iterrows():
                        print(f"Prompt: {row['prompt']}")
                        print(f"Response: {row['response']}")
                        print(f"Ground truth: {row['solution']}")
                        print("-" * 40)
                    k += 1
    policy.save_pretrained("Qwen2.5-Math-1.5B-GRPO")
    tokenizer.save_pretrained("Qwen2.5-Math-1.5B-GRPO")


def main():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    training_data = pd.read_json("data/math/training.jsonl", lines=True)
    validation_data = pd.read_json("data/math/validation.jsonl", lines=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen2.5-Math-1.5B-SFT").to("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-Math-1.5B-SFT")
    gpu_memory_utilization: float = 0.85
    vllm = init_vllm("cuda:0", seed=42, gpu_memory_utilization=gpu_memory_utilization)
    run(model, tokenizer, training_data, validation_data, vllm)


if __name__ == "__main__":
    main()
