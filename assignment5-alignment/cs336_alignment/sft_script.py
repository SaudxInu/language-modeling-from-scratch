import os

from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import wandb
import torch.nn.functional as F
from transformers.optimization import get_cosine_schedule_with_warmup

from cs336_alignment.data_loading import InstructionDataset, get_batch


os.environ["WANDB_PROJECT"] = "Llama-3.1-8B-Instruct"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def run(model, tokenizer, train_dataset, validation_dataset):
    epochs = 1
    batch_size = 2
    gradient_accumulation_steps = 16
    lr = 2e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.03 * epochs * len(train_dataset) // batch_size,
        num_training_steps=epochs * len(train_dataset) // batch_size,
    )
    train_dataloader = get_batch(train_dataset, batch_size, shuffle=True)
    validation_dataloader = get_batch(validation_dataset, batch_size, shuffle=False)
    print("Starting training...")
    wandb.init()
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")
    batch_loss = 0.0
    k = 0
    for _ in range(epochs):
        for idx, batch in enumerate(train_dataloader):
            model.train()
            input_ids = batch["input_ids"].to("cuda:1")
            labels = batch["labels"].to("cuda:1")
            logits = model(input_ids).logits
            loss = (
                F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                / gradient_accumulation_steps
            )
            loss.backward()
            current_loss = loss.detach().cpu().item()
            batch_loss += current_loss
            print(f"  Micro Batch {idx + 1}, Loss: {current_loss}")
            del (
                input_ids,
                labels,
                loss,
            )
            torch.cuda.empty_cache()
            if (idx + 1) % gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
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
                    for batch in validation_dataloader:
                        input_ids = batch["input_ids"].to("cuda:1")
                        labels = batch["labels"].to("cuda:1")
                        logits = model(input_ids).logits
                        loss = (
                            F.cross_entropy(
                                logits.view(-1, logits.size(-1)), labels.view(-1)
                            )
                            .detach()
                            .cpu()
                            .item()
                        )
                        batch_loss += loss.detach().cpu().item()
                    print(
                        f"  Validation Loss: {batch_loss / len(validation_dataloader)}"
                    )
                wandb.log(
                    {
                        "eval_step": k,
                        "eval/loss": batch_loss,
                    }
                )
                k += 1
    model.save_pretrained("Llama-3.1-8B-Instruct")
    tokenizer.save_pretrained("Llama-3.1-8B-Instruct")


def main():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    training_data = pd.read_json(
        "data/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz",
        lines=True,
        compression="gzip",
    )
    validation_data = pd.read_json(
        "data/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz",
        lines=True,
        compression="gzip",
    )
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    max_tokens = 512
    train_dataset = InstructionDataset(
        tokenizer,
        dataset_path="data/safety_augmented_ultrachat_200k_single_turn/train.jsonl.gz",
        seq_length=max_tokens,
        shuffle=True,
        raw_data=training_data,
    )
    validation_dataset = InstructionDataset(
        tokenizer,
        dataset_path="data/safety_augmented_ultrachat_200k_single_turn/test.jsonl.gz",
        seq_length=max_tokens,
        shuffle=False,
        raw_data=validation_data,
    )
    run(model, tokenizer, train_dataset, validation_dataset)


if __name__ == "__main__":
    main()
