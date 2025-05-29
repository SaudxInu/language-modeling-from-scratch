# uv run cs336_basics/tokenizer.py
# >> tinystories -> 4
# >> owt -> OOS, resource constraints
# >> use tinystories tokenizer for owt -> compression ratio will be smaller because
#    tokenizer is optimized for pair counts in tinystories
# throughput ->  (4 cores, 22 MiB, 17 mins, 0.32 MiB/min/core) -> (32 cores, 825 GiB, 58 days)
# >> vocab size < 65536
from multiprocessing import Pool
from typing import Iterable, Iterator
import json
import random
import regex as re

from tqdm import tqdm
import numpy as np

from train_bpe import find_chunk_boundaries


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.vocab_size = 256 + len(vocab)
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
        vocab = {int(k): v.encode("utf-8") for k, v in vocab.items()}
        with open(merges_filepath, "r") as f:
            merges = json.load(f)
        merges = [
            (merge[0].encode("utf-8"), merge[1].encode("utf-8")) for merge in merges
        ]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pre_tokens = self._pre_tokenize(text, self.special_tokens)
        tokens = []
        for pre_token in pre_tokens:
            for merge in self.merges:
                for i in range(len(pre_token) - 1):
                    if merge == pre_token[i : i + 2]:
                        pre_token = self._merge_pre_token(pre_token, merge)
                        break
            for token in pre_token:
                if token in self.reverse_vocab:
                    tokens.append(self.reverse_vocab[token])
                else:
                    tokens += list(token)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            tokens = self.encode(text)
            yield tokens

    def decode(self, ids: list[int]) -> str:
        text = b""
        for token in ids:
            if token > 255 and token < self.vocab_size:
                text += self.vocab[token]
            elif token < 256:
                text += bytes([token])
            else:
                text += chr(65533).encode("utf-8")
        text = text.decode("utf-8", "replace")
        return text

    def _pre_tokenize(
        self, text: str, special_tokens: list[str] = None
    ) -> list[tuple[bytes]]:
        if special_tokens:
            text = re.split(
                "|".join(
                    [re.escape(special_token) for special_token in special_tokens]
                ),
                text,
            )
        if type(text) is str:
            text = [text]
        pre_tokens = []
        for t in text:
            matches = re.finditer(
                r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                str(t),
            )
            for match in matches:
                pre_token = []
                for c in tuple(match.group()):
                    pre_token.append(c.encode("utf-8"))
                pre_tokens.append(tuple(pre_token))
            if special_tokens:
                for special_token in special_tokens:
                    pre_tokens.append((special_token.encode("utf-8"),))
        return pre_tokens

    def _merge_pre_token(
        self, pre_token: tuple[bytes], merge_pair: tuple[bytes, bytes]
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


def development():
    vocab_filepath = "results/vocab_development.json"
    merges_filepath = "results/merges_development.json"
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    texts = [
        "hello my name is Saud 😊",
        "nice to meet you, Saud!",
        "Saud is learning Python.",
        "Have a great day, Saud 😊",
        "hello my name is Saud 😊 and I love coding!",
    ]
    for i, tokens in enumerate(tokenizer.encode_iterable(texts)):
        text = tokenizer.decode(tokens[:-1])
        print(
            f"Number of tokens: {len(tokens)} | Compression Ratio: {len(list(text.encode('utf-8')))/len(tokens):.2f} | Texts match: {text == texts[i]}"
        )


def tinystories():
    vocab_filepath = "results/vocab_tinystories.json"
    merges_filepath = "results/merges_tinystories.json"
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
        boundaries = find_chunk_boundaries(
            f, 1000, tokenizer.special_tokens[0].encode("utf-8")
        )
        boundaries = list(zip(boundaries[:-1], boundaries[1:]))
        boundaries = random.sample(boundaries, 10)
        texts = []
        for start, end in boundaries:
            f.seek(start)
            text = f.read(end - start).decode("utf-8", errors="ignore")
            texts.append(text)
    for i, tokens in enumerate(tokenizer.encode_iterable(texts)):
        og_text = b""
        for x in tokenizer._pre_tokenize(texts[i], tokenizer.special_tokens):
            for y in x:
                og_text += y
        og_text = og_text.decode("utf-8")
        text = tokenizer.decode(tokens)
        print(
            f"Number of tokens: {len(tokens)} | Compression Ratio: {len(list(og_text.encode('utf-8')))/len(tokens):.2f} | Texts match: {text == og_text}"
        )


def owt():
    vocab_filepath = "results/vocab_owt.json"
    merges_filepath = "results/merges_owt.json"
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    with open("data/owt_valid.txt.gz", "rb") as f:
        boundaries = find_chunk_boundaries(
            f, 1000, tokenizer.special_tokens[0].encode("utf-8")
        )
        boundaries = list(zip(boundaries[:-1], boundaries[1:]))
        boundaries = random.sample(boundaries, 10)
        texts = []
        for start, end in boundaries:
            f.seek(start)
            text = f.read(end - start).decode("utf-8", errors="ignore")
            texts.append(text)
    for i, tokens in enumerate(tokenizer.encode_iterable(texts)):
        og_text = b""
        for x in tokenizer._pre_tokenize(texts[i], tokenizer.special_tokens):
            for y in x:
                og_text += y
        og_text = og_text.decode("utf-8")
        text = tokenizer.decode(tokens)
        print(
            f"Number of tokens: {len(tokens)} | Compression Ratio: {len(list(og_text.encode('utf-8')))/len(tokens):.2f} | Texts match: {text == og_text}"
        )


def _run(input_path, start, end, tokenizer):
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    texts = [text[i : i + 1000] for i in range(0, len(text), 1000)]
    tokenized_text = []
    for tokens in tokenizer.encode_iterable(texts):
        tokenized_text += tokens
    return tokenized_text


def throughput_and_save():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_path = "results/TinyStoriesV2-GPT4-train.npy"
    vocab_filepath = "results/vocab_tinystories.json"
    merges_filepath = "results/merges_tinystories.json"
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, 100000, tokenizer.special_tokens[0].encode("utf-8")
        )
        boundaries = list(zip(boundaries[:-1], boundaries[1:]))
        boundaries = random.sample(boundaries, 10000)
    with Pool(processes=4) as pool:
        tasks = []
        for start, end in boundaries:
            task = pool.apply_async(
                _run,
                args=(
                    input_path,
                    start,
                    end,
                    tokenizer,
                ),
            )
            tasks.append(task)
        outputs = []
        for t in tqdm(tasks):
            outputs.append(t.get())
    tokenized_text = []
    for output in outputs:
        tokenized_text += output
        tokenized_text += [256]
    tokenized_text = np.array(tokenized_text, dtype=np.uint16)
    np.save(output_path, tokenized_text)


if __name__ == "__main__":
    # development()
    # tinystories()
    # owt()
    # throughput_and_save()
    pass
