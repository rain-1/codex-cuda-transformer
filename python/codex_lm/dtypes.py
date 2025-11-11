"""Utilities for handling CLI dtype arguments."""
from __future__ import annotations

import torch

DTYPE_CHOICES: tuple[str, ...] = ("float32", "float16", "bfloat16")

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Return the torch.dtype that corresponds to a CLI dtype choice."""
    try:
        return _DTYPE_MAP[name]
    except KeyError as exc:  # pragma: no cover - guarded by CLI choices
        raise ValueError(f"Unsupported dtype option: {name}") from exc


__all__ = ["DTYPE_CHOICES", "resolve_dtype"]
