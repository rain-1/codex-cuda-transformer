"""CLI for training and sampling Codex Transformer models."""
from __future__ import annotations

import argparse
import pathlib

import torch
from torch.utils.data import DataLoader, Dataset

from codex_lm.config import MODEL_PRESETS, ModelConfig
from codex_lm.data import CharacterTokenizer, build_dataset, collate_batch, download_text
from codex_lm.model import TransformerLM
from codex_lm.trainer import TrainingConfig, Trainer, create_optimizer, create_scheduler


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data", choices=["tinyshakespeare", "custom"], help="Dataset choice")
    parser.add_argument("--data-path", type=pathlib.Path, default=None, help="Path to custom dataset")


def _resolve_dataset(choice: str, data_path: pathlib.Path | None) -> pathlib.Path:
    if choice == "tinyshakespeare":
        return download_text(
            "tinyshakespeare.txt",
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        )
    if data_path is None:
        raise ValueError("--data-path must be provided when using custom dataset")
    return data_path


def _load_tokenizer(path: pathlib.Path, seq_len: int) -> tuple[Dataset, Dataset, CharacterTokenizer]:
    return build_dataset(path, seq_len)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Codex Transformer utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a Transformer language model")
    train_parser.add_argument("model", choices=MODEL_PRESETS.keys(), help="Model size preset")
    _add_data_args(train_parser)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--micro-batch-size", type=int, default=8)
    train_parser.add_argument("--steps", type=int, default=1000)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.1)
    train_parser.add_argument("--warmup-ratio", type=float, default=0.03)
    train_parser.add_argument("--eval-interval", type=int, default=100)
    train_parser.add_argument("--eval-iters", type=int, default=10)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--device", type=str, default=_default_device())
    train_parser.add_argument("--compile", action="store_true")
    train_parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    train_parser.add_argument("--wandb-project", type=str, default="codex-transformer")
    train_parser.add_argument("--wandb-run", type=str, default=None)
    train_parser.add_argument(
        "--sample-prompt",
        action="append",
        default=[],
        help="Prompt text to decode during evaluation. Can be provided multiple times.",
    )
    train_parser.add_argument(
        "--sample-max-new-tokens",
        type=int,
        default=200,
        help="Number of new tokens to generate for evaluation samples.",
    )
    train_parser.add_argument(
        "--sample-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory to store decoded samples from evaluation.",
    )

    generate_parser = subparsers.add_parser("generate", help="Run inference with a saved checkpoint")
    generate_parser.add_argument("checkpoint", type=pathlib.Path, help="Path to the saved model checkpoint")
    _add_data_args(generate_parser)
    generate_parser.add_argument("--prompt", type=str, required=True, help="Prompt text used to start generation")
    generate_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of new tokens to sample from the model.",
    )
    generate_parser.add_argument("--device", type=str, default=_default_device())

    return parser.parse_args()


def _maybe_adjust_config(config: ModelConfig, tokenizer: CharacterTokenizer) -> ModelConfig:
    if tokenizer.vocab_size == config.vocab_size:
        return config
    print(
        f"[warning] tokenizer vocab_size={tokenizer.vocab_size} differs from preset {config.vocab_size}; adjusting model."
    )
    return ModelConfig(
        vocab_size=tokenizer.vocab_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        rotary_base=config.rotary_base,
    )


def _run_training(args: argparse.Namespace) -> None:
    preset_config = MODEL_PRESETS[args.model]
    data_path = _resolve_dataset(args.data, args.data_path)
    train_dataset, val_dataset, tokenizer = _load_tokenizer(data_path, preset_config.seq_len)
    model_config = _maybe_adjust_config(preset_config, tokenizer)

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
        override_model=model_config,
        sample_prompts=tuple(args.sample_prompt),
        sample_max_new_tokens=args.sample_max_new_tokens,
        sample_dir=args.sample_dir,
    )

    model = TransformerLM(train_config.model_config())
    optimizer = create_optimizer(model, train_config)
    scheduler = create_scheduler(optimizer, train_config)
    trainer = Trainer(model, optimizer, scheduler, train_config, tokenizer=tokenizer)
    trainer.train(train_loader, val_loader)


def _run_generation(args: argparse.Namespace) -> None:
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_dict = checkpoint["config"]
    override_model = config_dict.get("override_model")
    if override_model is not None:
        model_config = override_model
    else:
        model_name = config_dict["model_name"]
        model_config = MODEL_PRESETS[model_name]

    data_path = _resolve_dataset(args.data, args.data_path)
    _, _, tokenizer = _load_tokenizer(data_path, model_config.seq_len)

    model = TransformerLM(model_config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens:
        raise ValueError("Prompt must contain at least one known character to tokenize.")
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(prompt_tensor, args.max_new_tokens)
    decoded = tokenizer.decode(output[0].tolist())
    print(decoded)


def main() -> None:
    args = parse_args()
    if args.command == "train":
        _run_training(args)
    elif args.command == "generate":
        _run_generation(args)
    else:  # pragma: no cover - safety catch for argparse
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

