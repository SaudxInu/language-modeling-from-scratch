import numpy as np
import torch

from cs336_basics.checkpointing import load_checkpoint
from cs336_basics.nn.modules.softmax import softmax
from cs336_basics.nn.modules.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def generate(model, tokenizer, prompt, temperature=1.0, p=0.99, max_length=50):
    tokens = (
        torch.tensor(
            np.array(tokenizer.encode(prompt)[:-1], dtype=np.int16), dtype=torch.int32
        )
        .unsqueeze(0)
        .to(model.device)
    )

    model.eval()
    while True:
        with torch.no_grad():
            y_pred = model(tokens).squeeze(0)

            y_pred = y_pred[-1, :] / temperature

            y_pred_normalized = softmax(y_pred)

            sorted_p, idxs = y_pred_normalized.sort(descending=True)

            sorted_p = sorted_p.cumsum(dim=-1)

            mask = sorted_p > p

            mask[0] = False

            y_pred_normalized[idxs[mask]] = 0.0

            y_pred_normalized = y_pred_normalized / y_pred_normalized.sum()

            next_token = torch.multinomial(y_pred_normalized, num_samples=1)

            tokens = torch.cat((tokens, next_token.unsqueeze(0)), dim=1)

            if tokenizer.reverse_vocab["<|endoftext|>".encode("utf-8")] == next_token:
                break

            if tokens.shape[1] >= max_length:
                break

    return tokenizer.decode(tokens.squeeze(0).tolist())


if __name__ == "__main__":
    vocab_filepath = "results/vocab_tinystories.json"
    merges_filepath = "results/merges_tinystories.json"
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)

    checkpoint = torch.load("checkpoints/checkpoint.pt")
    model = TransformerLM(
        10000,
        256,
        4,
        512,
        16,
        1344,
        10000.0,
    )

    state_dict = {}
    for name, data in checkpoint["model"].items():
        state_dict[name.replace("_orig_mod.", "")] = data

    model.load_state_dict(state_dict)

    prompt = "Once upon a time, there was a pretty girl named Lily."

    generated_text = generate(model, tokenizer, prompt, temperature=0.8, max_length=256)

    print(f"Generated text: {generated_text}")
