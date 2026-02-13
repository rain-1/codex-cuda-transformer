"""Utility functions for preparing text datasets."""
from __future__ import annotations

import io
import json
import math
import pathlib
import random
import re
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Literal, Protocol, Sequence, TextIO, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data"
_DATA_DIR.mkdir(exist_ok=True)


_TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
_TINYSTORIES_SOURCES = ("TinyStories-train.txt", "TinyStories-valid.txt")
_TINYSTORIES_SPECIAL_TOKENS = ("<|endoftext|>", "<|end_of_sequence|>")
_CHUNK_SIZE = 1 << 20  # 1 MiB
_MEMMAP_THRESHOLD_BYTES = 256 * 1024 * 1024
_WORD_PATTERN = re.compile(r"\n|\w+|[^\w\s]")
_PUNCT_PATTERN = re.compile(r"[^\w\s]+")
TokenizerKind = Literal["char", "word"]


class Tokenizer(Protocol):
    vocab: List[str]
    stoi: dict[str, int]
    itos: dict[int, str]

    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, tokens: Sequence[int]) -> str:
        ...

    @property
    def vocab_size(self) -> int:
        ...


def download_text(name: str, url: str) -> pathlib.Path:
    """Download a text file if it does not already exist."""
    path = _DATA_DIR / name
    if not path.exists():
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    return path


@contextmanager
def _open_tinystories_source(path: pathlib.Path, url: str) -> Iterator[TextIO]:
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
            url = f"{_TINYSTORIES_URL}/{filename}"
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
            raise ValueError("Either text or vocab must be provided.")
        if vocab is None:
            tokens = sorted(set(text))
        else:
            tokens = sorted(set(vocab))
        self.vocab: List[str] = list(tokens)
        self.stoi = {token: idx for idx, token in enumerate(tokens)}
        self.itos = {idx: token for idx, token in enumerate(tokens)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: Sequence[int]) -> str:
        return "".join(self.itos[idx] for idx in tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class WordTokenizer:
    """Tokenizer that splits text into words, punctuation, and newlines."""

    def __init__(self, vocab: Sequence[str]):
        tokens = sorted(set(vocab))
        self.vocab: List[str] = list(tokens)
        self.stoi = {token: idx for idx, token in enumerate(tokens)}
        self.itos = {idx: token for idx, token in enumerate(tokens)}

    def encode(self, text: str) -> List[int]:
        return [self.stoi[token] for token in _WORD_PATTERN.findall(text)]

    def decode(self, tokens: Sequence[int]) -> str:
        pieces: List[str] = []
        need_space = False
        for idx in tokens:
            token = self.itos[idx]
            if token == "\n":
                pieces.append("\n")
                need_space = False
                continue
            if _PUNCT_PATTERN.fullmatch(token):
                pieces.append(token)
                need_space = False
                continue
            if need_space:
                pieces.append(" ")
            pieces.append(token)
            need_space = True
        return "".join(pieces)

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


class DocumentAlignedDataset(Dataset[Batch]):
    """Dataset that respects document boundaries and truncates long documents.

    Each item corresponds to a contiguous slice within the first
    ``max_tokens_per_doc`` tokens of a single document. Windows never cross
    document boundaries, preserving the beginning-of-document token in the
    context and preventing any one document from dominating a training step.
    """

    def __init__(self, windows: Sequence[torch.Tensor], seq_len: int, max_tokens_per_doc: int):
        if not windows:
            raise ValueError("DocumentAlignedDataset requires at least one window.")
        self.windows = windows
        self.seq_len = seq_len
        self.max_tokens_per_doc = max_tokens_per_doc

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.windows)

    def __getitem__(self, idx: int) -> Batch:  # type: ignore[override]
        window = self.windows[idx]
        x = window[: self.seq_len]
        y = window[1 : self.seq_len + 1]
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
    tokenizer: TokenizerKind = "char",
    document_aligned: bool = False,
    max_document_tokens: int = 2048,
) -> Tuple[Dataset[Batch], Dataset[Batch], Tokenizer]:
    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in the range (0, 1].")
    if document_aligned and max_document_tokens <= 0:
        raise ValueError("max_document_tokens must be positive when document_aligned is True.")
    random.seed(seed)
    tokens, tokenizer_obj = _load_or_cache_tokens(path, tokenizer)

    if document_aligned:
        target_tokens = int(tokens.numel() * fraction)
        target_tokens = max(seq_len + 1, target_tokens)
        documents = _tokenize_documents(path, tokenizer_obj, target_tokens, max_document_tokens)
        windows = _build_document_aligned_windows(documents, seq_len, max_document_tokens)
        if len(windows) < 2:
            raise ValueError(
                "Not enough document-aligned windows to create train/validation splits."
            )
        random.shuffle(windows)
        split_idx = int(len(windows) * split_ratio)
        split_idx = min(max(1, split_idx), len(windows) - 1)
        train_dataset = DocumentAlignedDataset(windows[:split_idx], seq_len, max_document_tokens)
        val_dataset = DocumentAlignedDataset(windows[split_idx:], seq_len, max_document_tokens)
        return train_dataset, val_dataset, tokenizer_obj

    usable = int(tokens.numel() * fraction)
    usable = max(seq_len + 1, usable)
    usable = min(tokens.numel(), usable)
    if usable <= seq_len:
        raise ValueError("Not enough tokens after applying fraction to build sequences.")
    tokens = tokens[:usable]

    split_idx = int(tokens.numel() * split_ratio)
    train_dataset = TextDataset(tokens[:split_idx], seq_len)
    val_dataset = TextDataset(tokens[split_idx:], seq_len)
    return train_dataset, val_dataset, tokenizer_obj


def cycle(iterable: Iterable[Batch]) -> Iterable[Batch]:
    while True:
        for item in iterable:
            yield item


def cosine_warmup(iteration: int, total_iters: int, warmup_iters: int) -> float:
    if iteration < warmup_iters:
        return iteration / max(1, warmup_iters)
    progress = (iteration - warmup_iters) / max(1, total_iters - warmup_iters)
    return 0.5 * (1 + math.cos(math.pi * progress))


def _tokenize_documents(
    path: pathlib.Path,
    tokenizer: Tokenizer,
    target_tokens: int,
    max_tokens_per_doc: int,
) -> list[torch.Tensor]:
    documents: list[torch.Tensor] = []
    total_tokens = 0
    for document in _iter_documents(path):
        encoded = tokenizer.encode(document)
        if not encoded:
            continue
        tokens = torch.tensor(encoded, dtype=torch.long)
        tokens = tokens[: max_tokens_per_doc + 1]
        if tokens.numel() <= 1:
            continue
        documents.append(tokens)
        total_tokens += tokens.numel()
        if total_tokens >= target_tokens:
            break
    return documents


def _build_document_aligned_windows(
    documents: Sequence[torch.Tensor], seq_len: int, max_tokens_per_doc: int
) -> list[torch.Tensor]:
    windows: list[torch.Tensor] = []
    for doc in documents:
        usable = min(doc.numel(), max_tokens_per_doc + 1)
        if usable <= seq_len:
            continue
        start = 0
        while start + seq_len + 1 <= usable:
            windows.append(doc[start : start + seq_len + 1])
            start += seq_len
    return windows


def _iter_documents(path: pathlib.Path) -> Iterator[str]:
    buffer: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip() == "":
                if buffer:
                    yield "".join(buffer)
                    buffer = []
                continue
            buffer.append(line)
        if buffer:
            yield "".join(buffer)


__all__ = [
    "Batch",
    "CharacterTokenizer",
    "Tokenizer",
    "WordTokenizer",
    "DocumentAlignedDataset",
    "collate_batch",
    "TextDataset",
    "build_dataset",
    "download_text",
    "download_tinystories",
    "cycle",
    "cosine_warmup",
]


def _load_or_cache_tokens(path: pathlib.Path, kind: TokenizerKind) -> Tuple[torch.Tensor, Tokenizer]:
    file_size = path.stat().st_size
    if file_size <= _MEMMAP_THRESHOLD_BYTES:
        text = path.read_text(encoding="utf-8")
        if kind == "char":
            tokenizer = CharacterTokenizer(text)
            encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        else:
            tokens = _WORD_PATTERN.findall(text)
            tokenizer = WordTokenizer(tokens)
            encoded = torch.tensor([tokenizer.stoi[token] for token in tokens], dtype=torch.long)
        return encoded, tokenizer

    tokens_path, meta_path = _token_cache_paths(path, kind)
    if tokens_path.exists() and meta_path.exists():
        try:
            return _load_cached_tokens(tokens_path, meta_path, kind)
        except ValueError:
            pass

    if kind == "char":
        return _build_char_token_cache(path, tokens_path, meta_path)
    return _build_word_token_cache(path, tokens_path, meta_path)


def _token_cache_paths(path: pathlib.Path, kind: TokenizerKind) -> Tuple[pathlib.Path, pathlib.Path]:
    base = path.with_suffix(path.suffix + f".{kind}.tokens")
    return base.with_suffix(base.suffix + ".npy"), base.with_suffix(base.suffix + ".json")


def _load_cached_tokens(
    tokens_path: pathlib.Path, meta_path: pathlib.Path, kind: TokenizerKind
) -> Tuple[torch.Tensor, Tokenizer]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("tokenizer") != kind:
        raise ValueError("Tokenizer kind mismatch for cached tokens.")
    tokenizer = _tokenizer_from_vocab(meta["vocab"], kind)
    dtype = np.dtype(meta["dtype"])
    memmap = np.memmap(tokens_path, dtype=dtype, mode="r+")
    tensor = torch.from_numpy(memmap)
    return tensor, tokenizer


def _build_char_token_cache(
    path: pathlib.Path,
    tokens_path: pathlib.Path,
    meta_path: pathlib.Path,
) -> Tuple[torch.Tensor, CharacterTokenizer]:
    vocab, total_chars = _collect_char_vocab(path)
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

    _write_cache_meta(meta_path, tokenizer.vocab, dtype, "char")
    memmap = np.memmap(tokens_path, dtype=dtype, mode="r+")
    tensor = torch.from_numpy(memmap)
    return tensor, tokenizer


def _build_word_token_cache(
    path: pathlib.Path,
    tokens_path: pathlib.Path,
    meta_path: pathlib.Path,
) -> Tuple[torch.Tensor, WordTokenizer]:
    vocab, total_tokens = _collect_word_vocab(path)
    tokenizer = WordTokenizer(vocab=vocab)
    dtype = _select_dtype(len(tokenizer.vocab))

    tmp_tokens = tokens_path.with_suffix(tokens_path.suffix + ".tmp")
    memmap = np.memmap(tmp_tokens, dtype=dtype, mode="w+", shape=(total_tokens,))

    offset = 0
    for token in _iter_word_tokens(path):
        memmap[offset] = tokenizer.stoi[token]
        offset += 1
    memmap.flush()
    tmp_tokens.replace(tokens_path)

    _write_cache_meta(meta_path, tokenizer.vocab, dtype, "word")
    memmap = np.memmap(tokens_path, dtype=dtype, mode="r+")
    tensor = torch.from_numpy(memmap)
    return tensor, tokenizer


def _write_cache_meta(
    meta_path: pathlib.Path,
    vocab: Sequence[str],
    dtype: np.dtype,
    kind: TokenizerKind,
) -> None:
    meta = {"vocab": list(vocab), "dtype": dtype.name, "tokenizer": kind}
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(meta), encoding="utf-8")
    meta_tmp.replace(meta_path)


def _tokenizer_from_vocab(vocab: Sequence[str], kind: TokenizerKind) -> Tokenizer:
    if kind == "char":
        return CharacterTokenizer(vocab=vocab)
    return WordTokenizer(vocab=vocab)


def _collect_char_vocab(path: pathlib.Path) -> Tuple[List[str], int]:
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


def _iter_word_tokens(path: pathlib.Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            for token in _WORD_PATTERN.findall(line):
                yield token


def _collect_word_vocab(path: pathlib.Path) -> Tuple[List[str], int]:
    vocab = set()
    total_tokens = 0
    for token in _iter_word_tokens(path):
        vocab.add(token)
        total_tokens += 1
    return sorted(vocab), total_tokens


def _select_dtype(vocab_size: int) -> np.dtype:
    if vocab_size <= 256:
        return np.dtype(np.uint8)
    if vocab_size <= 32768:
        return np.dtype(np.int16)
    return np.dtype(np.int32)
