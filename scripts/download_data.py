"""Download and validate Penn-Fudan without adding it to Git."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podfa.data.download import PENN_FUDAN_URL, download_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--url", default=PENN_FUDAN_URL)
    parser.add_argument("--expected-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = download_dataset(args.url, args.data_dir, args.expected_sha256)
    print(json.dumps({"root": str(summary.root), "image_mask_pairs": summary.count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
