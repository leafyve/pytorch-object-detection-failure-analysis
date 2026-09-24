"""Train a configured Torchvision detector with local MLflow tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podfa.training.pipeline import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(run_training(args.config, args.resume), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
