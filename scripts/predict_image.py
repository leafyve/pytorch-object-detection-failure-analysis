"""Run a trained detector on one image and save an annotated copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podfa.inference.image import predict_image
from podfa.models.loading import load_trained_model
from podfa.utils.config import load_config
from podfa.utils.environment import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = select_device()
    model, _ = load_trained_model(config["model"], args.checkpoint, device)
    summary = predict_image(model, args.input, args.output, args.threshold, device)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
