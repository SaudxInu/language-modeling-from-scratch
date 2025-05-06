# uv run scalene cs336_basics/train_bpe.py
# >> tinystories, parallelism 4 -> 24mins, 500MiB, ' accomplishment', LoC 147 (finding the most common pair takes the most time)
# >> owt, parallelism 4 -> OOS, resource constraints
from multiprocessing import Pool
from typing import BinaryIO
import json
import os
import regex as re


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"
    # Take file pointer to the end of the file
    file.seek(0, os.SEEK_END)
    # Get the size of the file in bytes
    file_size = file.tell()
    # Move the file pointer back to the beginning of the file
    file.seek(0)
    # Chunk size in bytes
    chunk_size = file_size // desired_num_chunks
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    # x x x x x x x x <|endoftext|> x x x x x <|endoftext|> x x x x x
    # [0, chunk_size, 2*chunk_size, ..., desired_num_chunks*chunk_size]
    # [0, chunk_size, 2*chunk_size, ..., file_size]
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(
    file_path: str, start: int, end: int, special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    # Read chunk
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    # Remove special tokens
    chunk = re.split(
        "|".join([re.escape(special_token) for special_token in special_tokens]), chunk
    )
    # Merge
    chunk = " ".join(chunk)
    # Pre-tokenize
    matches = re.finditer(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        chunk,
    )
    # Frequency count
    pre_token_counts = {}
    for match in matches:
        pre_token = []
        for c in tuple(match.group()):
            pre_token.append(c.encode("utf-8"))
        pre_token = tuple(pre_token)
        if pre_token in pre_token_counts:
            pre_token_counts[pre_token] += 1
        else:
            pre_token_counts[pre_token] = 1
    return pre_token_counts


def pre_tokenize(input_path: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 1000, special_tokens[0].encode("utf-8"))
    # outputs = []
    # for start, end in zip(boundaries[:-1], boundaries[1:]):
    #     output = pre_tokenize_chunk(input_path, start, end, special_tokens)
    #     outputs.append(output)
    #! To use multiprocessing convert the notebook to .py
    with Pool(processes=4) as pool:
        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            task = pool.apply_async(
                pre_tokenize_chunk,
                args=(
                    input_path,
                    start,
                    end,
                    special_tokens,
                ),
            )
            tasks.append(task)
        outputs = [t.get() for t in tasks]
    pre_token_counts = outputs[0]
    for output in outputs:
        for pre_token, count in output.items():
            if pre_token in pre_token_counts:
                pre_token_counts[pre_token] += count
            else:
                pre_token_counts[pre_token] = count
    return pre_token_counts


def get_pair_counts(
    pre_token_counts: dict[tuple[bytes], int],
) -> dict[tuple[bytes, bytes], tuple[int, dict[tuple[bytes], int]]]:
    pair_counts = {}
    for pre_token, count in pre_token_counts.items():
        for i in range(len(pre_token) - 1):
            pair = pre_token[i : i + 2]
            if pair in pair_counts:
                pair_counts[pair][0] += count
                pair_counts[pair][1][pre_token] = count
            else:
                pair_counts[pair] = [count, {pre_token: count}]
    return pair_counts


def merge_pre_token(
    pre_token: tuple[bytes], merge_pair: tuple[bytes, bytes]
) -> tuple[bytes]:
    new_pre_token = []
    i = 0
    while i < len(pre_token):
        pair = pre_token[i : i + 2]
        if pair == merge_pair:
            new_pre_token.append(pair[0] + pair[1])
            i += 2
        else:
            new_pre_token.append(pre_token[i])
            i += 1
    new_pre_token = tuple(new_pre_token)
    return new_pre_token


def merge(
    pair_counts: dict[tuple[bytes], tuple[int, dict[tuple[bytes], int]]],
) -> tuple[bytes, bytes]:
    # Find top pair
    top_pair = (
        sorted(
            pair_counts.items(), key=lambda item: (item[1][0], item[0]), reverse=True
        )
    )[0]
    merge_pair = top_pair[0]
    pre_tokens = dict(top_pair[1][1])
    # Update pair counts
    for pre_token, count in pre_tokens.items():
        for i in range(len(pre_token) - 1):
            pair = pre_token[i : i + 2]
            if pair in pair_counts:
                pair_counts[pair][0] -= count
                if pre_token in pair_counts[pair][1]:
                    del pair_counts[pair][1][pre_token]
    for pre_token, count in pre_tokens.items():
        pre_token = merge_pre_token(pre_token, merge_pair)
        for i in range(len(pre_token) - 1):
            pair = pre_token[i : i + 2]
            if pair in pair_counts:
                pair_counts[pair][0] += count
                pair_counts[pair][1][pre_token] = count
            else:
                pair_counts[pair] = [count, {pre_token: count}]
    return merge_pair


def train_bpe_tokenizer(
    input_path: str, vocab_size: int, special_tokens: list[int]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pre_token_counts = pre_tokenize(input_path, special_tokens)
    pair_counts = get_pair_counts(pre_token_counts)
    vocab = {}
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode("utf-8")
    k = 256 + len(vocab)
    merges = []
    while k < vocab_size:
        merge_pair = merge(pair_counts)
        vocab[k] = merge_pair[0] + merge_pair[1]
        k += 1
        merges.append(merge_pair)
    return vocab, merges


def train_bpe_development() -> None:
    vocab, merges = train_bpe_tokenizer(
        "data/Development.txt", 1_000, special_tokens=["<|endoftext|>"]
    )
    with open("results/vocab_development.json", "w") as f:
        json.dump({k: v.decode("utf-8") for k, v in vocab.items()}, f)
    with open("results/merges_development.json", "w") as f:
        json.dump([(x.decode("utf-8"), y.decode("utf-8")) for x, y in merges], f)


def train_bpe_tinystories() -> None:
    vocab, merges = train_bpe_tokenizer(
        "data/TinyStoriesV2-GPT4-train.txt", 10_000, special_tokens=["<|endoftext|>"]
    )
    with open("results/vocab_tinystories.json", "w") as f:
        json.dump({k: v.decode("utf-8") for k, v in vocab.items()}, f)
    with open("results/merges_tinystories.json", "w") as f:
        json.dump([(x.decode("utf-8"), y.decode("utf-8")) for x, y in merges], f)


def train_bpe_expts_owt() -> None:
    vocab, merges = train_bpe_tokenizer(
        "data/owt_train.txt.gz", 32_000, special_tokens=["<|endoftext|>"]
    )
    with open("results/vocab_owt.json", "w") as f:
        json.dump({k: v.decode("utf-8") for k, v in vocab.items()}, f)
    with open("results/merges_owt.json", "w") as f:
        json.dump([(x.decode("utf-8"), y.decode("utf-8")) for x, y in merges], f)


if __name__ == "__main__":
    os.makedirs("results/", exist_ok=True)
    # train_bpe_development()
    train_bpe_tinystories()
    # train_bpe_expts_owt()
