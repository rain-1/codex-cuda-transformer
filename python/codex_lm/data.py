"""Utility functions for preparing text datasets."""
from __future__ import annotations

import io
import json
import math
import pathlib
import random
import re
import urllib.request
from collections import Counter
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
TokenizerKind = Literal["char", "word", "bpe"]


_BPE_EOW = "</w>"
_BPE_SEGMENT_PATTERN = re.compile(r"\s+|\w+|[^\w\s]")
_DEFAULT_BPE_VOCAB_SIZE = 4096


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


class BPETokenizer:
    """Byte-Pair Encoding tokenizer operating on word segments."""

    def __init__(
        self,
        text: str | None = None,
        *,
        vocab: Sequence[str] | None = None,
        merges: Sequence[Sequence[str]] | None = None,
        whitespace_tokens: Sequence[str] | None = None,
        max_vocab_size: int = _DEFAULT_BPE_VOCAB_SIZE,
    ) -> None:
        if text is None and (vocab is None or merges is None):
            raise ValueError("Either text or (vocab and merges) must be provided.")
        self.max_vocab_size = max_vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.whitespace_tokens: set[str] = set(whitespace_tokens or [])
        if text is not None:
            self._train(text)
        else:
            self.vocab = list(vocab or [])
            self.merges = [tuple(pair) for pair in merges or []]
            if not self.whitespace_tokens:
                self.whitespace_tokens = {token for token in self.vocab if token.isspace()}
            self._build_lookup()

    def _train(self, text: str) -> None:
        segments = _BPE_SEGMENT_PATTERN.findall(text)
        word_freqs: dict[Tuple[str, ...], int] = {}
        base_vocab: set[str] = set()
        whitespace_tokens: set[str] = set()

        for segment in segments:
            if not segment:
                continue
            if segment.isspace():
                whitespace_tokens.add(segment)
                base_vocab.add(segment)
                for ch in set(segment):
                    whitespace_tokens.add(ch)
                    base_vocab.add(ch)
                continue
            word = tuple(segment) + (_BPE_EOW,)
            word_freqs[word] = word_freqs.get(word, 0) + 1
            base_vocab.update(word)

        base_vocab.add(_BPE_EOW)
        vocab: List[str] = sorted(base_vocab)
        merges: List[Tuple[str, str]] = []

        while len(vocab) < self.max_vocab_size:
            pair_freqs: Counter[Tuple[str, str]] = Counter()
            for word, freq in word_freqs.items():
                for idx in range(len(word) - 1):
                    pair = (word[idx], word[idx + 1])
                    pair_freqs[pair] += freq
            if not pair_freqs:
                break
            best_pair, best_freq = pair_freqs.most_common(1)[0]
            if best_freq < 2:
                break
            word_freqs = {self._merge_word(word, best_pair): freq for word, freq in word_freqs.items()}
            merged_token = "".join(best_pair)
            if merged_token not in vocab:
                vocab.append(merged_token)
            merges.append(best_pair)

        self.vocab = vocab
        self.merges = merges
        self.whitespace_tokens = whitespace_tokens
        self._build_lookup()

    def _build_lookup(self) -> None:
        self.stoi = {token: idx for idx, token in enumerate(self.vocab)}
        self.itos = {idx: token for idx, token in enumerate(self.vocab)}
        self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges)}

    def _merge_word(self, word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        merged: List[str] = []
        idx = 0
        while idx < len(word):
            if idx < len(word) - 1 and word[idx] == pair[0] and word[idx + 1] == pair[1]:
                merged.append(word[idx] + word[idx + 1])
                idx += 2
            else:
                merged.append(word[idx])
                idx += 1
        return tuple(merged)

    def _encode_segment(self, segment: str) -> List[int]:
        if not segment:
            return []
        symbols: List[str] = list(segment)
        symbols.append(_BPE_EOW)

        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked_pairs = [(self.merge_ranks[pair], pair) for pair in pairs if pair in self.merge_ranks]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs, key=lambda item: item[0])
            symbols = list(self._merge_word(tuple(symbols), best_pair))

        encoded: List[int] = []
        for symbol in symbols:
            token = symbol
            if token not in self.stoi:
                fallback = token.replace(_BPE_EOW, "")
                for ch in fallback:
                    if ch not in self.stoi:
                        raise KeyError(f"Unknown token '{ch}' encountered during BPE encoding.")
                    encoded.append(self.stoi[ch])
                continue
            encoded.append(self.stoi[token])
        return encoded

    def encode(self, text: str) -> List[int]:
        tokens: List[int] = []
        for segment in _BPE_SEGMENT_PATTERN.findall(text):
            if not segment:
                continue
            if segment in self.stoi and (segment in self.whitespace_tokens or segment.isspace()):
                tokens.append(self.stoi[segment])
                continue
            if segment.isspace():
                for ch in segment:
                    if ch not in self.stoi:
                        raise KeyError(f"Unknown whitespace character '{ch}' in BPE tokenizer.")
                    tokens.append(self.stoi[ch])
                continue
            tokens.extend(self._encode_segment(segment))
        return tokens

    def decode(self, tokens: Sequence[int]) -> str:
        pieces: List[str] = []
        for idx in tokens:
            token = self.itos[idx]
            if token in self.whitespace_tokens or token.isspace():
                pieces.append(token)
                continue
            pieces.append(token.replace(_BPE_EOW, ""))
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
) -> Tuple[TextDataset, TextDataset, Tokenizer]:
    if not (0 < fraction <= 1.0):
        raise ValueError("fraction must be in the range (0, 1].")
    random.seed(seed)
    tokens, tokenizer_obj = _load_or_cache_tokens(path, tokenizer)

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


__all__ = [
    "Batch",
    "CharacterTokenizer",
    "BPETokenizer",
    "Tokenizer",
    "WordTokenizer",
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
        elif kind == "word":
            tokens = _WORD_PATTERN.findall(text)
            tokenizer = WordTokenizer(tokens)
            encoded = torch.tensor([tokenizer.stoi[token] for token in tokens], dtype=torch.long)
        else:
            tokenizer = BPETokenizer(text)
            encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        return encoded, tokenizer

    tokens_path, meta_path = _token_cache_paths(path, kind)
    if tokens_path.exists() and meta_path.exists():
        try:
            return _load_cached_tokens(tokens_path, meta_path, kind)
        except ValueError:
            pass

    if kind == "char":
        return _build_char_token_cache(path, tokens_path, meta_path)
    if kind == "word":
        return _build_word_token_cache(path, tokens_path, meta_path)
    return _build_bpe_token_cache(path, tokens_path, meta_path)


def _token_cache_paths(path: pathlib.Path, kind: TokenizerKind) -> Tuple[pathlib.Path, pathlib.Path]:
    base = path.with_suffix(path.suffix + f".{kind}.tokens")
    return base.with_suffix(base.suffix + ".npy"), base.with_suffix(base.suffix + ".json")


def _load_cached_tokens(
    tokens_path: pathlib.Path, meta_path: pathlib.Path, kind: TokenizerKind
) -> Tuple[torch.Tensor, Tokenizer]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("tokenizer") != kind:
        raise ValueError("Tokenizer kind mismatch for cached tokens.")
    tokenizer = _tokenizer_from_meta(meta)
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


def _build_bpe_token_cache(
    path: pathlib.Path,
    tokens_path: pathlib.Path,
    meta_path: pathlib.Path,
) -> Tuple[torch.Tensor, BPETokenizer]:
    text = path.read_text(encoding="utf-8")
    tokenizer = BPETokenizer(text)
    dtype = _select_dtype(len(tokenizer.vocab))

    encoded = tokenizer.encode(text)
    tmp_tokens = tokens_path.with_suffix(tokens_path.suffix + ".tmp")
    memmap = np.memmap(tmp_tokens, dtype=dtype, mode="w+", shape=(len(encoded),))
    memmap[:] = np.fromiter(encoded, dtype=dtype, count=len(encoded))
    memmap.flush()
    tmp_tokens.replace(tokens_path)

    extra = {
        "merges": [list(pair) for pair in tokenizer.merges],
        "whitespace_tokens": sorted(tokenizer.whitespace_tokens),
        "max_vocab_size": tokenizer.max_vocab_size,
    }
    _write_cache_meta(meta_path, tokenizer.vocab, dtype, "bpe", extra)
    memmap = np.memmap(tokens_path, dtype=dtype, mode="r+")
    tensor = torch.from_numpy(memmap)
    return tensor, tokenizer


def _write_cache_meta(
    meta_path: pathlib.Path,
    vocab: Sequence[str],
    dtype: np.dtype,
    kind: TokenizerKind,
    extra: dict | None = None,
) -> None:
    meta = {"vocab": list(vocab), "dtype": dtype.name, "tokenizer": kind}
    if extra:
        meta.update(extra)
    meta_tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    meta_tmp.write_text(json.dumps(meta), encoding="utf-8")
    meta_tmp.replace(meta_path)


def _tokenizer_from_meta(meta: dict) -> Tokenizer:
    kind: TokenizerKind = meta["tokenizer"]
    vocab = meta["vocab"]
    if kind == "char":
        return CharacterTokenizer(vocab=vocab)
    if kind == "word":
        return WordTokenizer(vocab=vocab)
    merges = [tuple(pair) for pair in meta.get("merges", [])]
    whitespace_tokens = meta.get("whitespace_tokens", [])
    max_vocab_size = meta.get("max_vocab_size", _DEFAULT_BPE_VOCAB_SIZE)
    return BPETokenizer(
        vocab=vocab,
        merges=merges,
        whitespace_tokens=whitespace_tokens,
        max_vocab_size=max_vocab_size,
    )


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
