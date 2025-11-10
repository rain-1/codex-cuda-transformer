"""Configuration presets for Codex CUDA Transformer models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ModelConfig:
    """Dataclass holding transformer hyperparameters."""

    vocab_size: int
    seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float = 0.0
    rotary_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads


MODEL_PRESETS: Dict[str, ModelConfig] = {
    "pico": ModelConfig(
        vocab_size=512,
        seq_len=256,
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        dropout=0.1,
    ),
    "nano": ModelConfig(
        vocab_size=2048,
        seq_len=512,
        d_model=512,
        n_layers=12,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
    ),
    "micro": ModelConfig(
        vocab_size=4096,
        seq_len=1024,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_ff=8192,
        dropout=0.1,
    ),
}

