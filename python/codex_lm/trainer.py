"""Training utilities for the Codex Transformer."""
from __future__ import annotations

import html
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import MODEL_PRESETS, ModelConfig
from .data import Batch, Tokenizer, cosine_warmup, cycle
from .dtypes import resolve_dtype
from .model import TransformerLM

try:  # Optional wandb logging
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore


@dataclass
class TrainingConfig:
    model_name: str
    batch_size: int
    micro_batch_size: int
    num_steps: int
    lr: float
    weight_decay: float
    warmup_ratio: float
    eval_interval: int
    eval_iters: int
    grad_clip: float
    device: str = "cuda"
    dtype: str = "float32"
    compile: bool = False
    use_wandb: bool = False
    wandb_project: str = "codex-transformer"
    wandb_run: Optional[str] = None
    override_model: Optional[ModelConfig] = None
    sample_prompts: tuple[str, ...] = ()
    sample_max_new_tokens: int = 200
    sample_dir: Optional[pathlib.Path] = None
    tokenizer: str = "char"

    def model_config(self) -> ModelConfig:
        return self.override_model or MODEL_PRESETS[self.model_name]


def _prepare_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        batch.x.to(device=device, dtype=torch.long),
        batch.y.to(device=device, dtype=torch.long),
    )


class Trainer:
    def __init__(
        self,
        model: TransformerLM,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
        config: TrainingConfig,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = resolve_dtype(config.dtype)
        self.tokenizer = tokenizer

        use_grad_scaler = self.device.type == "cuda" and self.dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

        self._wandb_samples_table = None
        if config.use_wandb and wandb is not None:
            wandb.init(project=config.wandb_project, name=config.wandb_run, config=config.__dict__)
            self._wandb_samples_table = wandb.Table(columns=["step", "sample", "prompt", "completion"])

    def train(
        self,
        train_loader: DataLoader[Batch],
        val_loader: DataLoader[Batch],
    ) -> None:
        device = self.device
        self.model.to(device=device)
        if self.config.compile:
            self.model = torch.compile(self.model)
        model = self.model

        model.train()
        train_iter: Iterator[Batch] = cycle(train_loader)
        accum_steps = max(1, self.config.batch_size // self.config.micro_batch_size)
        best_val = float("inf")
        for step in range(1, self.config.num_steps + 1):
            start = time.time()
            losses = []
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(accum_steps):
                batch = _prepare_batch(next(train_iter), device)
                with self._autocast_context():
                    _, loss = model(batch.x, batch.y)
                assert loss is not None
                losses.append(loss.detach())
                self.scaler.scale(loss / accum_steps).backward()
            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            loss_val = torch.stack(losses).mean().item()
            step_time = time.time() - start
            log_data = {
                "step": step,
                "train/loss": loss_val,
                "lr": self.optimizer.param_groups[0]["lr"],
                "step_time": step_time,
            }
            if self.config.use_wandb and wandb is not None:
                wandb.log(log_data)
            else:
                print(f"step {step:06d} loss={loss_val:.4f} lr={log_data['lr']:.3e} time={step_time:.2f}s")

            if step % self.config.eval_interval == 0:
                val_loss = self.evaluate(val_loader)
                sample_logs = self._log_samples(step)
                if self.config.use_wandb and wandb is not None:
                    log_payload: Dict[str, Any] = {
                        "val/loss": val_loss,
                        "val/perplexity": torch.exp(torch.tensor(val_loss)).item(),
                    }
                    log_payload.update(sample_logs)
                    wandb.log(log_payload, step=step)
                else:
                    print(f"val loss={val_loss:.4f} ppl={math.exp(val_loss):.2f}")
                if val_loss < best_val:
                    best_val = val_loss
                    self._save_checkpoint("best.pt")
        self._save_checkpoint("last.pt")

    def evaluate(self, loader: DataLoader[Batch]) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            iterator = iter(loader)
            for _ in range(self.config.eval_iters):
                try:
                    raw_batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    raw_batch = next(iterator)
                batch = _prepare_batch(raw_batch, self.device)
                with self._autocast_context():
                    _, loss = self.model(batch.x, batch.y)
                assert loss is not None
                losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def _save_checkpoint(self, name: str) -> None:
        path = pathlib.Path("checkpoints")
        path.mkdir(exist_ok=True)
        torch.save({"model": self.model.state_dict(), "config": self.config.__dict__}, path / name)

    def _log_samples(self, step: int) -> Dict[str, Any]:
        if not self.config.sample_prompts or self.tokenizer is None:
            return {}

        sample_dir = self.config.sample_dir
        if sample_dir is not None:
            sample_dir.mkdir(parents=True, exist_ok=True)

        was_training = self.model.training
        self.model.eval()
        outputs = []
        wandb_payload: Dict[str, Any] = {}
        wandb_rows_added = False
        with torch.no_grad():
            for idx, prompt in enumerate(self.config.sample_prompts):
                try:
                    prompt_tokens = self.tokenizer.encode(prompt)
                except KeyError as exc:
                    print(f"[warning] skipping sample prompt {idx} due to unknown character: {exc}")
                    continue
                if not prompt_tokens:
                    print(f"[warning] skipping empty prompt at index {idx}")
                    continue
                input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
                generated = self.model.generate(input_tensor, self.config.sample_max_new_tokens)
                text = self.tokenizer.decode(generated[0].tolist())
                outputs.append((idx, prompt, text))
                if self.config.use_wandb and wandb is not None:
                    html_prompt = html.escape(prompt)
                    html_text = html.escape(text)
                    wandb_payload[f"samples/{idx}"] = wandb.Html(
                        f"<div><h4>Prompt</h4><pre>{html_prompt}</pre><h4>Completion</h4><pre>{html_text}</pre></div>"
                    )
                    if self._wandb_samples_table is not None:
                        self._wandb_samples_table.add_data(step, idx, prompt, text)
                        wandb_rows_added = True
                if sample_dir is not None:
                    file_path = sample_dir / f"step_{step:06d}_sample_{idx}.txt"
                    file_path.write_text(text, encoding="utf-8")

        if was_training:
            self.model.train()

        if wandb_rows_added and self.config.use_wandb and wandb is not None and self._wandb_samples_table is not None:
            wandb_payload["eval/samples"] = self._wandb_samples_table

        for idx, prompt, text in outputs:
            separator = "-" * 80
            print(f"sample[{idx}] @ step {step}\nprompt: {prompt}\n{separator}\n{text}\n{separator}")

        return wandb_payload

    def _autocast_context(self):
        if self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16):
            return torch.amp.autocast("cuda", dtype=self.dtype)
        return nullcontext()


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.95))


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> torch.optim.lr_scheduler.LambdaLR:
    total_iters = config.num_steps
    warmup_iters = int(total_iters * config.warmup_ratio)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup(step, total_iters, warmup_iters),
    )


__all__ = ["TrainingConfig", "Trainer", "create_optimizer", "create_scheduler"]

