"""Create the deterministic Penn-Fudan train/validation/test manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podfa.data.splits import create_split_manifest, save_split_manifest
from podfa.data.validation import validate_dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/PennFudanPed"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/splits/penn_fudan_seed42.json")
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate_dataset_root(args.dataset_root)
    manifest = create_split_manifest(summary.records, args.seed)
    save_split_manifest(manifest, args.output)
    print(json.dumps({"manifest": str(args.output), "counts": manifest.counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
