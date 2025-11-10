"""CLI for training Codex Transformer models."""
from __future__ import annotations

import argparse
import pathlib

import torch
from torch.utils.data import DataLoader

from codex_lm.config import MODEL_PRESETS, ModelConfig
from codex_lm.data import build_dataset, collate_batch, download_text
from codex_lm.model import TransformerLM
from codex_lm.trainer import TrainingConfig, Trainer, create_optimizer, create_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Transformer language model")
    parser.add_argument("model", choices=MODEL_PRESETS.keys(), help="Model size preset")
    parser.add_argument("data", choices=["tinyshakespeare", "custom"], help="Dataset choice")
    parser.add_argument("--data-path", type=pathlib.Path, default=None, help="Path to custom dataset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="codex-transformer")
    parser.add_argument("--wandb-run", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MODEL_PRESETS[args.model]

    if args.data == "tinyshakespeare":
        path = download_text("tinyshakespeare.txt", "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    else:
        if args.data_path is None:
            raise ValueError("--data-path must be provided when using custom dataset")
        path = args.data_path

    train_dataset, val_dataset, tokenizer = build_dataset(path, config.seq_len)
    if tokenizer.vocab_size != config.vocab_size:
        print(
            f"[warning] tokenizer vocab_size={tokenizer.vocab_size} differs from preset {config.vocab_size}; adjusting model.")
        config = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            seq_len=config.seq_len,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            rotary_base=config.rotary_base,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_batch,
    )

    train_config = TrainingConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        num_steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        grad_clip=args.grad_clip,
        device=args.device,
        compile=args.compile,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        override_model=config,
    )

    model = TransformerLM(train_config.model_config())
    optimizer = create_optimizer(model, train_config)
    scheduler = create_scheduler(optimizer, train_config)
    trainer = Trainer(model, optimizer, scheduler, train_config)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()

