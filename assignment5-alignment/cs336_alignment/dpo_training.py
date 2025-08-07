import os

from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import wandb

from cs336_alignment.dpo_loss import dpo_loss


os.environ["WANDB_PROJECT"] = "Llama-3.1-8B-DPO"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def run(model, model_ref, tokenizer, train_dataset, validation_dataset):
    epochs = 1
    batch_size = 1
    gradient_accumulation_steps = 64
    lr = 1e-6
    beta = 0.1
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    print("Starting training...")
    wandb.init()
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    batch_loss = 0.0
    k = 0
    for _ in range(epochs):
        for idx, row in train_dataset.iterrows():
            instruction, chosen, rejected = (
                row["instruction"],
                row["chosen"],
                row["rejected"],
            )
            model.train()
            loss = (
                dpo_loss(
                    model, model_ref, tokenizer, beta, instruction, chosen, rejected
                )
                / gradient_accumulation_steps
            )
            loss.backward()
            current_loss = loss.detach().cpu().item()
            batch_loss += current_loss
            print(f"  Micro Batch {idx + 1}, Loss: {current_loss}")
            del loss
            torch.cuda.empty_cache()
            if (idx + 1) % gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                print(
                    f"  Batch {(idx + 1) // gradient_accumulation_steps}, Loss: {batch_loss}"
                )
                wandb.log(
                    {
                        "train_step": k,
                        "train/loss": batch_loss,
                    }
                )
                model.eval()
                with torch.no_grad():
                    batch_loss = 0.0
                    for _, row in validation_dataset.iterrows():
                        instruction, chosen, rejected = (
                            row["instruction"],
                            row["chosen"],
                            row["rejected"],
                        )
                        loss = (
                            dpo_loss(
                                model,
                                model_ref,
                                tokenizer,
                                beta,
                                instruction,
                                chosen,
                                rejected,
                            )
                            / gradient_accumulation_steps
                        )
                        batch_loss += loss.detach().cpu().item()
                    print(f"  Validation Loss: {batch_loss / len(validation_dataset)}")
                wandb.log(
                    {
                        "eval_step": k,
                        "eval/loss": batch_loss,
                    }
                )
                k += 1
    model.save_pretrained("Llama-3.1-8B-DPO")
    tokenizer.save_pretrained("Llama-3.1-8B-DPO")


def main():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    data = pd.read_json(
        "data/hh-rlhf/processed_hh.jsonl.gz",
        lines=True,
        compression="gzip",
    )
    data = data.sample(frac=1).reset_index(drop=True)
    training_data = data.iloc[: int(0.9 * len(data))]
    validation_data = data.iloc[int(0.9 * len(data)) :]
    model = AutoModelForCausalLM.from_pretrained(
        "Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda:0")
    model_ref = AutoModelForCausalLM.from_pretrained(
        "Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained("Llama-3.1-8B-Instruct")
    run(model, model_ref, tokenizer, training_data, validation_data)


if __name__ == "__main__":
    main()
