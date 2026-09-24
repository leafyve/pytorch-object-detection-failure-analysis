"""Annotate a local video with a trained person detector and measured FPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podfa.inference.video import predict_video
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
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = select_device()
    model, _ = load_trained_model(config["model"], args.checkpoint, device)
    summary = predict_video(model, args.input, args.output, args.threshold, device)
    payload = summary.to_dict()
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
