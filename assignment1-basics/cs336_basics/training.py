import numpy as np
import torch

from cs336_basics.checkpointing import save_checkpoint
from cs336_basics.data_loading import data_loading
from cs336_basics.nn.learning_rate_schedule import learning_rate_schedule
from cs336_basics.nn.modules.adamw import AdamW
from cs336_basics.nn.modules.gradient_clipping import gradient_clipping
from cs336_basics.nn.modules.transformer_lm import TransformerLM


def train(
    training_data_file_path: str,
    validation_data_file_path: str,
    vocab_size: int = 50257,
    context_length: int = 5,
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
    num_iters: int = 10,
    checkpoint_dir: str = "checkpoints",
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
):
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

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate_schedule(1, lr, lr / 10, num_iters // 2, num_iters),
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )

    training_data = np.load(training_data_file_path, mmap_mode="r")
    validation_data = np.load(validation_data_file_path, mmap_mode="r")

    for iter in range(1, num_iters + 1):
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

        gradient_clipping(model.parameters(), maximum_norm=1.0)

        optimizer.step()

        if iter % 2 == 0:
            print(
                f"Iteration {iter}/{num_iters}, Training Loss: {loss.item():.4f}, "
                f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}"
            )

            if iter % 5 == 0:
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

                    print(
                        f"Iteration {iter}/{num_iters}, Validation Loss: {loss.item():.4f}, "
                    )

            save_checkpoint(model, optimizer, iter, f"{checkpoint_dir}/iter_{iter}.pt")

        optimizer.param_groups[0]["lr"] = learning_rate_schedule(
            iter + 1, lr, lr / 10, num_iters // 2, num_iters
        )


if __name__ == "__main__":
    train(
        "results/TinyStoriesV2-GPT4-train.npy", "results/TinyStoriesV2-GPT4-valid.npy"
    )
