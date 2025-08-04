import os

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed
import pandas as pd
import torch
import wandb

from cs336_alignment.get_response_log_probs import get_response_log_probs
from cs336_alignment.log_generations import log_generations
from cs336_alignment.sft_microbatch_train_step import sft_microbatch_train_step
from cs336_alignment.tokenize_prompt_and_output import tokenize_prompt_and_output


os.environ["WANDB_PROJECT"] = "Qwen2.5-Math-1.5B-SFT"
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


def run(model, tokenizer, training_data, validation_data, vllm):
    epochs = 3
    max_tokens = 2048
    batch_size = 2
    gradient_accumulation_steps = 64
    lr = 1e-6
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_dataset = TrainingDataset(training_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Starting training...")
    wandb.init()
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    model.train()
    batch_loss = 0.0
    batch_token_entropy = 0.0
    k = 0
    for _ in range(epochs):
        for idx, batch in enumerate(train_dataloader):
            prompts, responses = batch
            tokenized_data = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = tokenized_data["input_ids"].to("cuda:1")[:, :max_tokens]
            labels = tokenized_data["labels"].to("cuda:1")[:, :max_tokens]
            response_mask = tokenized_data["response_mask"].to("cuda:1")[:, :max_tokens]
            res = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True,
            )
            log_probs = res["log_probs"]
            token_entropy = res["token_entropy"].mean()
            loss, _ = sft_microbatch_train_step(
                log_probs,
                response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=1.0,
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
                input_ids,
                labels,
                response_mask,
                tokenized_data,
                res,
                log_probs,
                token_entropy,
                loss,
            )
            torch.cuda.empty_cache()
            if (idx + 1) % gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                load_policy_into_vllm_instance(model, vllm)
                eval_results = log_generations(
                    vllm,
                    validation_sample["prompt"].tolist(),
                    validation_sample["solution"].tolist(),
                )
                validation_sample["response"] = eval_results["generated_texts"]
                validation_sample["format_reward"] = [
                    reward.get("format_reward", 0) for reward in eval_results["rewards"]
                ]
                validation_sample["answer_reward"] = [
                    reward.get("answer_reward", 0) for reward in eval_results["rewards"]
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
    model.save_pretrained("Qwen2.5-Math-1.5B-SFT")
    tokenizer.save_pretrained("Qwen2.5-Math-1.5B-SFT")


def main():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    training_data = pd.read_json("data/math/training.jsonl", lines=True)
    validation_data = pd.read_json("data/math/validation.jsonl", lines=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B").to("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    vllm = init_vllm("cuda:0", seed=42)
    run(model, tokenizer, training_data, validation_data, vllm)


if __name__ == "__main__":
    main()
