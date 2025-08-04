import torch
from transformers import PreTrainedTokenizer


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer
):
    prompt_tokens = tokenizer.batch_encode_plus(
        prompt_strs,
        add_special_tokens=False,
    )["input_ids"]
    response_tokens = tokenizer.batch_encode_plus(
        output_strs,
        add_special_tokens=False,
    )["input_ids"]
    response_tokens = [x + [tokenizer.eos_token_id] for x in response_tokens]
    prompt_lens = [len(x) for x in prompt_tokens]
    output_lens = [len(x) for x in response_tokens]
    prompt_and_output_lens = [x + y for x, y in zip(prompt_lens, output_lens)]
    max_len = max(prompt_and_output_lens)
    input_tokens = [x + y[:-1] for x, y in zip(prompt_tokens, response_tokens)]
    labels = [x[1:] + y for x, y in zip(prompt_tokens, response_tokens)]
    response_mask = [
        [False] * (x - 1)
        + [True] * (y - 1)
        + [False] * (max_len - 1 - (x - 1) - (y - 1))
        for x, y in zip(prompt_lens, output_lens)
    ]
    input_tokens = [
        x + [tokenizer.pad_token_id] * (max_len - 1 - len(x)) for x in input_tokens
    ]
    labels = [x + [tokenizer.pad_token_id] * (max_len - 1 - len(x)) for x in labels]
    input_tokens = torch.tensor(input_tokens, dtype=torch.long)[:, :-1]
    labels = torch.tensor(labels, dtype=torch.long)[:, :-1]
    response_mask = torch.tensor(response_mask, dtype=torch.bool)[:, :-1]
    return {
        "input_ids": input_tokens,
        "labels": labels,
        "response_mask": response_mask,
    }
