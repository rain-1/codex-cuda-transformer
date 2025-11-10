"""Utility functions for preparing text datasets."""
from __future__ import annotations

import math
import pathlib
import random
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)


_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
# Placeholder URL for future larger datasets.
_SIMPLE_ENGLISH_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"


def download_text(name: str, url: str) -> pathlib.Path:
    """Download a text file if it does not already exist."""
    path = _DATA_DIR / name
    if not path.exists():
        with urllib.request.urlopen(url) as response:
            data = response.read()
        path.write_bytes(data)
    return path


class CharacterTokenizer:
    """A simple character-level tokenizer."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab: List[str] = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(self.itos[idx] for idx in tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor


class TextDataset(Dataset[Batch]):
    """Dataset that returns contiguous token windows."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:  # type: ignore[override]
        return self.tokens.numel() - self.seq_len

    def __getitem__(self, idx: int) -> Batch:  # type: ignore[override]
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + 1 + self.seq_len]
        return Batch(x, y)


def collate_batch(items: Sequence[Batch]) -> Batch:
    xs = torch.stack([item.x for item in items], dim=0)
    ys = torch.stack([item.y for item in items], dim=0)
    return Batch(xs, ys)


def build_dataset(
    path: pathlib.Path,
    seq_len: int,
    split_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[TextDataset, TextDataset, CharacterTokenizer]:
    random.seed(seed)
    text = path.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split_idx = int(len(encoded) * split_ratio)
    train_tokens = encoded[:split_idx]
    val_tokens = encoded[split_idx:]

    train_dataset = TextDataset(train_tokens, seq_len)
    val_dataset = TextDataset(val_tokens, seq_len)
    return train_dataset, val_dataset, tokenizer


def cycle(iterable: Iterable[Batch]) -> Iterable[Batch]:
    while True:
        for item in iterable:
            yield item


def cosine_warmup(iteration: int, total_iters: int, warmup_iters: int) -> float:
    if iteration < warmup_iters:
        return iteration / max(1, warmup_iters)
    progress = (iteration - warmup_iters) / max(1, total_iters - warmup_iters)
    return 0.5 * (1 + math.cos(math.pi * progress))


__all__ = [
    "Batch",
    "CharacterTokenizer",
    "collate_batch",
    "TextDataset",
    "build_dataset",
    "download_text",
    "cycle",
    "cosine_warmup",
]

