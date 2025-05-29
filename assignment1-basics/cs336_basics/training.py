import numpy as np
import torch
import wandb
from tqdm import tqdm

from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.data_loading import data_loading
from cs336_basics.decoding import generate
from cs336_basics.nn.modules.learning_rate_schedule import learning_rate_schedule
from cs336_basics.nn.modules.adamw import AdamW
from cs336_basics.nn.modules.gradient_clipping import gradient_clipping
from cs336_basics.nn.modules.perplexity import perplexity
from cs336_basics.nn.modules.transformer_lm import TransformerLM
from cs336_basics.experiment_log import experiment_log, log_metric
from cs336_basics.tokenizer import Tokenizer


def train(
    training_data_file_path: str,
    validation_data_file_path: str,
    vocab_size: int = 10000,
    context_length: int = 100,
    num_layers: int = 2,
    d_model: int = 16,
    num_heads: int = 2,
    d_ff: int = 32,
    theta: float = 10000.0,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    batch_size: int = 2,
    num_iters: int = 1000,
    checkpoint_dir: str = "checkpoints",
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    **kwargs: dict,
):
    vocab_filepath = "results/vocab_tinystories.json"
    merges_filepath = "results/merges_tinystories.json"
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    model = TransformerLM(
        vocab_size,
        context_length,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        theta,
        device,
        dtype,
    )

    model = torch.compile(model, backend="aot_eager")

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate_schedule(1, lr, lr / 10, num_iters // 2, num_iters),
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    training_data = np.load(training_data_file_path, mmap_mode="r")
    validation_data = np.load(validation_data_file_path, mmap_mode="r")

    for iter in tqdm(range(1, num_iters + 1)):
        model.train()

        x, y = data_loading(training_data, batch_size, context_length, device)

        x = x.to(device=device, dtype=torch.int32)
        y = y.to(device=device, dtype=torch.long)

        y_pred = model(x)

        loss = torch.nn.functional.cross_entropy(
            y_pred.view(-1, y_pred.shape[-1]), y.view(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        norm = gradient_clipping(model.parameters(), maximum_norm=1.0)

        optimizer.step()

        if iter % 250 == 0:
            training_loss = loss.item()
            training_norm = norm.item()

            with torch.no_grad():
                training_perplexity = perplexity(
                    y_pred.view(-1, y_pred.shape[-1]), y.view(-1)
                ).item()

            print(
                f"Iteration {iter}/{num_iters}, Training Loss: {training_loss:.4f}, "
                f"Training Norm: {training_norm:.4f}, "
                f"Training Perplexity: {training_perplexity:.4f}, "
                f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
            )

            if iter % 1000 == 0:
                model.eval()

                with torch.no_grad():
                    x, y = data_loading(
                        validation_data, batch_size, context_length, device
                    )

                    x = x.to(device=device, dtype=torch.int32)
                    y = y.to(device=device, dtype=torch.long)

                    y_pred = model(x)

                    loss = torch.nn.functional.cross_entropy(
                        y_pred.view(-1, y_pred.shape[-1]), y.view(-1)
                    )

                    validation_loss = loss.item()

                    validation_perplexity = perplexity(
                        y_pred.view(-1, y_pred.shape[-1]), y.view(-1)
                    ).item()

                    print(
                        f"Iteration {iter}/{num_iters}, Validation Loss: {validation_loss:.4f}, "
                        f"Validation Perplexity: {validation_perplexity:.4f}, "
                    )

                    for temperature in [0.5, 0.8, 1.0]:
                        print("iter:", iter)
                        print("temperature:", temperature)
                        print("p:", 0.95)
                        print(
                            "output:",
                            generate(
                                model,
                                tokenizer,
                                "Once upon a time",
                                temperature=temperature,
                                p=0.95,
                                max_length=100,
                            ),
                        )

                    log_metric(
                        {
                            "iteration": iter,
                            "training_loss": training_loss,
                            "training_perplexity": training_perplexity,
                            "training_norm": training_norm,
                            "validation_loss": validation_loss,
                            "validation_perplexity": validation_perplexity,
                        }
                    )

            save_checkpoint(model, optimizer, iter, f"{checkpoint_dir}/checkpoint.pt")

        optimizer.param_groups[0]["lr"] = learning_rate_schedule(
            iter + 1, lr, lr / 10, num_iters // 2, num_iters
        )


if __name__ == "__main__":
    config = {
        "project": "train-tiny-stories",
        "job_type": "train",
        "name": "run-1",
        "vocab_size": 10000,
        "context_length": 256,
        "num_layers": 4,
        "d_model": 512,
        "num_heads": 16,
        "d_ff": 1344,
        "theta": 10000.0,
        "batch_size": 32,
        "num_iters": 5000,
    }

    experiment_log(config)

    train(
        "results/TinyStoriesV2-GPT4-train.npy",
        "results/TinyStoriesV2-GPT4-valid.npy",
        device="mps",
        **config,
    )
