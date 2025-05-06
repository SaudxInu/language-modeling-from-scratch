import json
from typing import Iterable, Iterator
import regex as re


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

    def _pre_tokenize(self, text: str, special_tokens: list[str]) -> list[tuple[bytes]]:
        text = re.split(
            "|".join([re.escape(special_token) for special_token in special_tokens]),
            text,
        )
        text = " ".join(text)
        matches = re.finditer(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            text,
        )
        pre_tokens = []
        for match in matches:
            pre_token = []
            for c in tuple(match.group()):
                pre_token.append(c.encode("utf-8"))
            pre_tokens.append(tuple(pre_token))
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


if __name__ == "__main__":
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
        text = tokenizer.decode(tokens)
        print(
            f"Length of text: {len(text)} | Number of tokens: {len(tokens)} | Ratio: {len(text)/len(tokens):.2f} | Texts match: {text == texts[i]}"
        )
