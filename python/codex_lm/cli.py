"""CLI for training and sampling Codex Transformer models."""
from __future__ import annotations

import argparse
import pathlib
import pickle
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from codex_lm.config import MODEL_PRESETS, ModelConfig
from codex_lm.data import (
    Tokenizer,
    build_dataset,
    collate_batch,
    download_text,
    download_tinystories,
)
from codex_lm.dtypes import DTYPE_CHOICES, resolve_dtype
from codex_lm.memory import MemoryAnalyzer
from codex_lm.model import TransformerLM
from codex_lm.trainer import TrainingConfig, Trainer, create_optimizer, create_scheduler

try:  # PyTorch >= 2.6 exposes safe serialization helpers
    from torch.serialization import safe_globals as _torch_safe_globals
except (ImportError, AttributeError):  # pragma: no cover - older PyTorch versions
    _torch_safe_globals = None


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


DATASET_CHOICES = ("tinyshakespeare", "tinystories", "custom")

DEFAULT_SAMPLE_PROMPTS: dict[str, tuple[str, ...]] = {
    "tinyshakespeare": (
        "ROMEO: ",
        "HAMLET: ",
    ),
    "tinystories": (
        "Once upon a time",
        "The robot ",
    ),
}

FALLBACK_SAMPLE_PROMPTS: tuple[str, ...] = ("Once upon a time",)


def _add_data_args(
    parser: argparse.ArgumentParser,
    *,
    include_positional: bool = True,
    default_tokenizer: str | None = "char",
) -> None:
    if include_positional:
        parser.add_argument("data", choices=DATASET_CHOICES, help="Dataset choice")
    parser.add_argument("--data-path", type=pathlib.Path, default=None, help="Path to custom dataset")
    parser.add_argument(
        "--data-frac",
        type=float,
        default=1.0,
        help="Fraction of the dataset to keep (0 < frac <= 1).",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["char", "word", "bpe"],
        default=default_tokenizer,
        help="Tokenizer granularity (character, word/punctuation, or byte-pair encoding).",
    )


def _resolve_dataset(choice: str, data_path: pathlib.Path | None) -> pathlib.Path:
    if choice == "tinyshakespeare":
        return download_text(
            "tinyshakespeare.txt",
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        )
    if choice == "tinystories":
        return download_tinystories()
    if data_path is None:
        raise ValueError("--data-path must be provided when using custom dataset")
    return data_path


def _load_tokenizer(
    path: pathlib.Path,
    seq_len: int,
    fraction: float,
    tokenizer_kind: str,
) -> tuple[Dataset, Dataset, Tokenizer]:
    return build_dataset(path, seq_len, fraction=fraction, tokenizer=tokenizer_kind)  # type: ignore[arg-type]


def _coerce_model_config(config: Any) -> ModelConfig:
    if isinstance(config, ModelConfig):
        return config
    if isinstance(config, dict):
        return ModelConfig(**config)
    raise TypeError(f"Unsupported model config type in checkpoint: {type(config)!r}")


def _load_checkpoint(path: pathlib.Path) -> dict[str, Any]:
    load_kwargs = {"map_location": "cpu"}
    context = (
        _torch_safe_globals([ModelConfig])  # type: ignore[misc]
        if _torch_safe_globals is not None
        else nullcontext()
    )
    with context:
        try:
            checkpoint = torch.load(path, **load_kwargs)
        except pickle.UnpicklingError:
            try:
                checkpoint = torch.load(path, weights_only=False, **load_kwargs)
            except TypeError as error:  # pragma: no cover - PyTorch < 2.6 fallback
                raise RuntimeError(
                    "Failed to load checkpoint. Upgrade PyTorch or re-save the checkpoint with"
                    " a compatible version."
                ) from error
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint file did not contain a state dict dictionary.")
    return checkpoint


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
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override the number of gradient accumulation steps (defaults to batch_size/micro_batch_size).",
    )
    train_parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to trade extra compute for lower activation memory.",
    )
    train_parser.add_argument("--device", type=str, default=_default_device())
    train_parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="float32",
        help="Computation precision for the forward pass (weights remain float32).",
    )
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
    train_parser.set_defaults(print_samples=False)
    train_parser.add_argument(
        "--print-samples",
        dest="print_samples",
        action="store_true",
        help="Print generated samples to stdout during evaluation steps.",
    )
    train_parser.add_argument(
        "--no-print-samples",
        dest="print_samples",
        action="store_false",
        help="Disable printing generated samples to stdout during evaluation steps.",
    )

    generate_parser = subparsers.add_parser("generate", help="Run inference with a saved checkpoint")
    generate_parser.add_argument(
        "generate_args",
        nargs=2,
        metavar=("checkpoint", "data"),
        help=(
            "Checkpoint path and dataset choice (tinyshakespeare, tinystories, or custom). The order "
            "of these two arguments is flexible for backwards compatibility."
        ),
    )
    _add_data_args(generate_parser, include_positional=False, default_tokenizer=None)
    generate_parser.add_argument("--prompt", type=str, required=True, help="Prompt text used to start generation")
    generate_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of new tokens to sample from the model.",
    )
    generate_parser.add_argument("--device", type=str, default=_default_device())

    info_parser = subparsers.add_parser("info", help="Print information about a model preset")
    info_parser.add_argument("model", choices=MODEL_PRESETS.keys(), help="Model size preset")

    analyze_parser = subparsers.add_parser(
        "analyze-memory", help="Profile CUDA memory usage for a model forward pass"
    )
    analyze_parser.add_argument("model", choices=MODEL_PRESETS.keys(), help="Model size preset")
    analyze_parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size to use for the synthetic input"
    )
    analyze_parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Sequence length (tokens) for the synthetic batch; defaults to the model preset",
    )
    analyze_parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override the model's maximum sequence length before profiling.",
    )
    analyze_parser.add_argument(
        "--dtype",
        choices=DTYPE_CHOICES,
        default="float16",
        help="Precision to cast the model parameters to during profiling.",
    )
    analyze_parser.add_argument(
        "--device",
        type=str,
        default=_default_device(),
        help="Device on which to run the memory analysis (requires CUDA).",
    )
    analyze_parser.add_argument(
        "--no-backward",
        action="store_true",
        help="Skip the backward() call when profiling (forward pass only).",
    )

    args = parser.parse_args()

    if args.command == "generate":
        first, second = args.generate_args
        dataset_options = set(DATASET_CHOICES)

        if first in dataset_options and second in dataset_options:
            parser.error(
                "Provide exactly one dataset choice and one checkpoint path when using the generate command."
            )

        if first in dataset_options:
            data_choice = first
            checkpoint_str = second
        elif second in dataset_options:
            data_choice = second
            checkpoint_str = first
        else:
            parser.error(
                "Could not determine the dataset choice. Expected one of tinyshakespeare, tinystories, or custom."
            )

        args.checkpoint = pathlib.Path(checkpoint_str)
        args.data = data_choice
        delattr(args, "generate_args")

    return args


def _maybe_adjust_config(config: ModelConfig, tokenizer: Tokenizer) -> ModelConfig:
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
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    if device.type != "cuda" and dtype in (torch.float16, torch.bfloat16):
        raise ValueError("--dtype float16/bfloat16 requires a CUDA device.")
    tokenizer_choice = args.tokenizer or "char"
    data_path = _resolve_dataset(args.data, args.data_path)
    train_dataset, val_dataset, tokenizer = _load_tokenizer(
        data_path,
        preset_config.seq_len,
        args.data_frac,
        tokenizer_choice,
    )
    model_config = _maybe_adjust_config(preset_config, tokenizer)

    if args.sample_prompt:
        sample_prompts = tuple(args.sample_prompt)
    else:
        sample_prompts = DEFAULT_SAMPLE_PROMPTS.get(args.data, FALLBACK_SAMPLE_PROMPTS)
        prompt_preview = ", ".join(repr(prompt) for prompt in sample_prompts)
        print(f"[info] using default sample prompts for {args.data}: {prompt_preview}")

    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be a positive integer when provided")

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

    if args.sample_prompt:
        sample_prompts = tuple(args.sample_prompt)
    else:
        sample_prompts = DEFAULT_SAMPLE_PROMPTS.get(args.data, FALLBACK_SAMPLE_PROMPTS)
        prompt_preview = ", ".join(repr(prompt) for prompt in sample_prompts)
        print(f"[info] using default sample prompts for {args.data}: {prompt_preview}")

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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run=args.wandb_run,
        override_model=model_config,
        sample_prompts=sample_prompts,
        sample_max_new_tokens=args.sample_max_new_tokens,
        sample_dir=args.sample_dir,
        tokenizer=tokenizer_choice,
        print_samples=args.print_samples,
    )

    model = TransformerLM(
        train_config.model_config(), gradient_checkpointing=train_config.gradient_checkpointing
    )
    optimizer = create_optimizer(model, train_config)
    scheduler = create_scheduler(optimizer, train_config)
    trainer = Trainer(model, optimizer, scheduler, train_config, tokenizer=tokenizer)
    trainer.train(train_loader, val_loader)


def _run_generation(args: argparse.Namespace) -> None:
    checkpoint = _load_checkpoint(args.checkpoint)
    config_dict = checkpoint["config"]
    override_model = config_dict.get("override_model")
    if override_model is not None:
        model_config = _coerce_model_config(override_model)
    else:
        model_name = config_dict["model_name"]
        model_config = MODEL_PRESETS[model_name]

    config_tokenizer = config_dict.get("tokenizer")
    tokenizer_choice = args.tokenizer or config_tokenizer or "char"
    if config_tokenizer and args.tokenizer and args.tokenizer != config_tokenizer:
        print(
            f"[warning] CLI tokenizer '{args.tokenizer}' differs from checkpoint tokenizer '{config_tokenizer}'; "
            "using CLI selection."
        )
    elif not args.tokenizer and config_tokenizer:
        tokenizer_choice = config_tokenizer

    data_path = _resolve_dataset(args.data, args.data_path)
    _, _, tokenizer = _load_tokenizer(
        data_path,
        model_config.seq_len,
        args.data_frac,
        tokenizer_choice,
    )

    model = TransformerLM(model_config)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens:
        raise ValueError("Prompt must contain at least one known token to tokenize.")
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(prompt_tensor, args.max_new_tokens)
    decoded = tokenizer.decode(output[0].tolist())
    print(decoded)



def _run_info(args: argparse.Namespace) -> None:
    config = MODEL_PRESETS[args.model]
    model = TransformerLM(config)
    param_count = sum(param.numel() for param in model.parameters())
    print(f"Preset: {args.model}")
    for field in ("vocab_size", "seq_len", "d_model", "n_layers", "n_heads", "d_ff", "dropout", "rotary_base"):
        value = getattr(config, field)
        print(f"  {field}: {value}")
    print(f"Parameters: {param_count:,}")


def _run_memory_analysis(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("Memory analysis requires a CUDA device.")

    dtype = resolve_dtype(args.dtype)
    preset_config = MODEL_PRESETS[args.model]
    config = preset_config
    if args.seq_len is not None:
        if args.seq_len <= 0:
            raise ValueError("--seq-len must be a positive integer")
        config = replace(preset_config, seq_len=args.seq_len)

    context_length = args.context_length or config.seq_len
    if context_length <= 0:
        raise ValueError("--context-length must be positive")
    if context_length > config.seq_len:
        raise ValueError(
            f"context-length ({context_length}) cannot exceed the model sequence length ({config.seq_len})."
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    model = TransformerLM(config)
    model.to(device=device, dtype=dtype)
    model.train()
    model.zero_grad(set_to_none=True)

    analyzer = MemoryAnalyzer(device)
    if device.index is not None:
        torch.cuda.set_device(device.index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    input_tokens = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, context_length),
        dtype=torch.long,
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, context_length),
        dtype=torch.long,
        device=device,
    )

    with analyzer.section("forward_pass"):
        _, loss = model(input_tokens, targets=targets, memory_analyzer=analyzer)

    if loss is not None and not args.no_backward:
        with analyzer.section("backward"):
            loss.backward()

    torch.cuda.synchronize(device)
    total_peak = torch.cuda.max_memory_allocated(device)
    print(analyzer.format_report())
    print(f"\nTotal peak allocation: {total_peak / (1024 ** 3):.2f} GiB")

    param_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    print(f"Parameter storage (dtype={args.dtype}): {param_bytes / (1024 ** 2):.2f} MiB")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        _run_training(args)
    elif args.command == "generate":
        _run_generation(args)
    elif args.command == "info":
        _run_info(args)
    elif args.command == "analyze-memory":
        _run_memory_analysis(args)
    else:  # pragma: no cover - safety catch for argparse
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
