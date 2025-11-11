"""PyTorch implementation of a decoder-only Transformer language model."""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig

if TYPE_CHECKING:  # pragma: no cover - used for type checking only
    from .memory import MemoryAnalyzer


def _memory_section(analyzer: "MemoryAnalyzer" | None, name: str):
    return analyzer.section(name) if analyzer is not None else nullcontext()


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x


def rotary_emb(
    head_dim: int,
    seq_len: int,
    base: float,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even to use rotary embeddings")
    theta = torch.arange(head_dim // 2, device=device, dtype=torch.float32)
    theta = base ** (-2 * theta / head_dim)
    seq_idx = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(seq_idx, theta)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    if dtype != torch.float32:
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
    return cos, sin


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dtype != q.dtype:
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    # Broadcast rotary frequencies to [batch, heads, seq, head_dim / 2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).reshape_as(q)
    k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).reshape_as(k)
    return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.seq_len, config.seq_len)), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        memory_analyzer: "MemoryAnalyzer" | None = None,
        label: Optional[str] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        scope = label or "attention"
        with _memory_section(memory_analyzer, f"{scope}.qkv_proj"):
            qkv = self.qkv(x)
            qkv = qkv.view(batch, seq_len, 3, self.config.n_heads, self.config.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        with _memory_section(memory_analyzer, f"{scope}.rotary"):
            q, k = apply_rotary(q, k, cos[:seq_len], sin[:seq_len])

        with _memory_section(memory_analyzer, f"{scope}.scores"):
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.config.head_dim)
            mask = self.mask[:seq_len, :seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        with _memory_section(memory_analyzer, f"{scope}.softmax"):
            attn = torch.softmax(attn_scores, dim=-1)
            attn = self.dropout(attn)

        with _memory_section(memory_analyzer, f"{scope}.output"):
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.config.d_model)
            out = self.out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        memory_analyzer: "MemoryAnalyzer" | None = None,
        block_label: Optional[str] = None,
    ) -> torch.Tensor:
        prefix = block_label or "block"
        with _memory_section(memory_analyzer, f"{prefix}.norm1"):
            norm_x = self.norm1(x)
        with _memory_section(memory_analyzer, f"{prefix}.attn"):
            attn_out = self.attn(norm_x, cos, sin, memory_analyzer=memory_analyzer, label=f"{prefix}.attn")
        x = x + attn_out
        with _memory_section(memory_analyzer, f"{prefix}.norm2"):
            norm_x = self.norm2(x)
        with _memory_section(memory_analyzer, f"{prefix}.ff"):
            ff_out = self.ff(norm_x)
        return x + ff_out


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig, *, gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        memory_analyzer: "MemoryAnalyzer" | None = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, seq_len = idx.shape
        device = idx.device
        with _memory_section(memory_analyzer, "token_embedding"):
            x = self.token_emb(idx)
        with _memory_section(memory_analyzer, "rotary_emb"):
            cos, sin = rotary_emb(
                self.config.head_dim,
                self.config.seq_len,
                self.config.rotary_base,
                device,
                dtype=x.dtype,
            )
        for i, block in enumerate(self.blocks):
            label = f"block_{i}"
            if self.gradient_checkpointing and self.training:

                def block_forward(inp: torch.Tensor) -> torch.Tensor:
                    with _memory_section(memory_analyzer, label):
                        return block(
                            inp,
                            cos,
                            sin,
                            memory_analyzer=memory_analyzer,
                            block_label=label,
                        )

                x = checkpoint(block_forward, x)
            else:
                with _memory_section(memory_analyzer, label):
                    x = block(
                        x,
                        cos,
                        sin,
                        memory_analyzer=memory_analyzer,
                        block_label=label,
                    )
        with _memory_section(memory_analyzer, "final_norm"):
            x = self.norm(x)
        with _memory_section(memory_analyzer, "lm_head"):
            logits = self.lm_head(x)

        loss = None
        if targets is not None:
            with _memory_section(memory_analyzer, "loss"):
                logits_view = logits.view(batch * seq_len, -1)
                targets_view = targets.view(batch * seq_len)
                loss = F.cross_entropy(logits_view, targets_view)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


__all__ = ["TransformerLM"]

