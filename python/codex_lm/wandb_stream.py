"""Simple helper to stream JSON metrics to Weights & Biases."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream stdin JSON lines into wandb")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config: Dict[str, Any] = {}
    if args.config:
        config = json.loads(args.config)
    run = wandb.init(project=args.project, name=args.name, config=config)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        step = payload.pop("step", None)
        wandb.log(payload, step=step)
    run.finish()


if __name__ == "__main__":
    main()

