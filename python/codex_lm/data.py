"""Utility functions for preparing text datasets."""
from __future__ import annotations

import io
import json
import math
import pathlib
import random
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, TextIO, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)


_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
# Placeholder URL for future larger datasets.
_SIMPLE_ENGLISH_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
_TINYSTORIES_BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
_TINYSTORIES_SOURCES = ("TinyStories-train.txt", "TinyStories-valid.txt")
_TINYSTORIES_SPECIAL_TOKENS = ("<|endoftext|>", "<|end_of_sequence|>")
_CHUNK_SIZE = 1 << 20  # 1 MiB of characters per streaming chunk
_MEMMAP_THRESHOLD_BYTES = 256 * 1024 * 1024  # Switch to streaming pipelines above 256 MiB


def download_text(name: str, url: str) -> pathlib.Path:
    """Download a text file if it does not already exist."""
    path = _DATA_DIR / name
    if not path.exists():
        with urllib.request.urlopen(url) as response:
            data = response.read()
        path.write_bytes(data)
    return path


@contextmanager
def _open_tinystories_source(path: pathlib.Path, url: str) -> Iterator[str]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            yield handle
        return
    with urllib.request.urlopen(url) as response:
        with io.TextIOWrapper(response, encoding="utf-8") as reader:
            yield reader


def _write_tinystories_line(dest: TextIO, line: str) -> None:
    stripped = line.strip()
    if stripped in _TINYSTORIES_SPECIAL_TOKENS:
        dest.write("\n\n")
        return
    normalized = line
    for token in _TINYSTORIES_SPECIAL_TOKENS:
        if token in normalized:
            normalized = normalized.replace(token, "\n\n")
    dest.write(normalized)


def download_tinystories() -> pathlib.Path:
    """Download and normalize the TinyStories dataset into a single text file."""
    output = _DATA_DIR / "tinystories.txt"
    if output.exists():
        return output

    tmp_output = output.with_suffix(output.suffix + ".tmp")
    with tmp_output.open("w", encoding="utf-8") as dest:
        for idx, filename in enumerate(_TINYSTORIES_SOURCES):
            url = f"{_TINYSTORIES_BASE_URL}/{filename}"
            source_path = _DATA_DIR / filename
            with _open_tinystories_source(source_path, url) as reader:
                for line in reader:
                    _write_tinystories_line(dest, line)
            if idx + 1 < len(_TINYSTORIES_SOURCES):
                dest.write("\n")
    tmp_output.replace(output)
    return output


class CharacterTokenizer:
    """A simple character-level tokenizer."""

    def __init__(self, text: str | None = None, vocab: Sequence[str] | None = None):
        if text is None and vocab is None:
            raise ValueError("Either text or vocab must be provided")
        if vocab is None:
            chars = sorted(set(text))
        else:
            chars = sorted(set(vocab))
        self.vocab: List[str] = list(chars)
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
    fraction: float = 1.0,
) -> Tuple[TextDataset, TextDataset, CharacterTokenizer]:
    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in the range (0, 1].")
    random.seed(seed)
    tokens, tokenizer = _load_or_cache_tokens(path)

    usable = int(tokens.numel() * fraction)
    usable = max(seq_len + 1, usable)
    usable = min(tokens.numel(), usable)
    tokens = tokens[:usable]
    if tokens.numel() <= seq_len:
        raise ValueError("Not enough tokens after applying fraction to build sequences.")

    split_idx = int(tokens.numel() * split_ratio)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

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
    "download_tinystories",
    "cycle",
    "cosine_warmup",
]


def _load_or_cache_tokens(path: pathlib.Path) -> Tuple[torch.Tensor, CharacterTokenizer]:
    file_size = path.stat().st_size
    if file_size <= _MEMMAP_THRESHOLD_BYTES:
        text = path.read_text(encoding="utf-8")
        tokenizer = CharacterTokenizer(text)
        encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        return encoded, tokenizer

    tokens_path, meta_path = _token_cache_paths(path)
    if tokens_path.exists() and meta_path.exists():
        return _load_cached_tokens(tokens_path, meta_path)
    return _build_token_cache(path, tokens_path, meta_path)


def _token_cache_paths(path: pathlib.Path) -> Tuple[pathlib.Path, pathlib.Path]:
    base = path.with_suffix(path.suffix + ".tokens")
    return base.with_suffix(base.suffix + ".npy"), base.with_suffix(base.suffix + ".json")


def _load_cached_tokens(tokens_path: pathlib.Path, meta_path: pathlib.Path) -> Tuple[torch.Tensor, CharacterTokenizer]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    tokenizer = CharacterTokenizer(vocab=meta["vocab"])
    dtype = np.dtype(meta["dtype"])
    memmap = np.memmap(tokens_path, dtype=dtype, mode="r+")
    tensor = torch.from_numpy(memmap)
    return tensor, tokenizer


def _build_token_cache(
    path: pathlib.Path,
    tokens_path: pathlib.Path,
    meta_path: pathlib.Path,
) -> Tuple[torch.Tensor, CharacterTokenizer]:
    vocab, total_chars = _collect_vocab(path)
    tokenizer = CharacterTokenizer(vocab=vocab)
    dtype = _select_dtype(len(tokenizer.vocab))

    tmp_tokens = tokens_path.with_suffix(tokens_path.suffix + ".tmp")
    memmap = np.memmap(tmp_tokens, dtype=dtype, mode="w+", shape=(total_chars,))

    with path.open("r", encoding="utf-8") as handle:
        offset = 0
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            arr = np.fromiter((tokenizer.stoi[ch] for ch in chunk), dtype=dtype, count=len(chunk))
            memmap[offset : offset + len(arr)] = arr
            offset += len(arr)
    memmap.flush()
    tmp_tokens.replace(tokens_path)

    meta = {"vocab": tokenizer.vocab, "dtype": str(dtype.name)}
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(meta), encoding="utf-8")
    meta_tmp.replace(meta_path)

    memmap = np.memmap(tokens_path, dtype=dtype, mode="r+")
    tensor = torch.from_numpy(memmap)
    return tensor, tokenizer


def _collect_vocab(path: pathlib.Path) -> Tuple[List[str], int]:
    vocab = set()
    total_chars = 0
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            total_chars += len(chunk)
            vocab.update(chunk)
    return sorted(vocab), total_chars


def _select_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= 256:
        return np.dtype(np.uint8)
    if vocab_size <= 32768:
        return np.dtype(np.int16)
    return np.dtype(np.int32)

