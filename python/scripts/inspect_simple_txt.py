#!/usr/bin/env python3
"""Display random samples from the Simple Wikipedia text corpus."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def sample_files(root: Path, count: int) -> list[Path]:
    files = [path for path in root.rglob("*.txt") if path.is_file()]
    if not files:
        print(f"[error] No .txt files found under {root}", file=sys.stderr)
        sys.exit(1)
    if count >= len(files):
        return files
    return random.sample(files, count)


def print_samples(files: list[Path], root: Path) -> None:
    sep = "=" * 80
    for idx, path in enumerate(files, start=1):
        relative = path.relative_to(root)
        print(sep)
        print(f"Sample {idx}: {relative}")
        print(sep)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="replace")
        print(content.rstrip())
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing plain-text articles (simple_txt).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of random files to display (default: 10).",
    )
    args = parser.parse_args(argv)

    files = sample_files(args.root, args.count)
    random.shuffle(files)
    print_samples(files, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
