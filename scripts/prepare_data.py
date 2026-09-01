from __future__ import annotations

import argparse
from pathlib import Path

from fedrec.config import ExperimentConfig
from fedrec.data import prepare_movielens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MovieLens 100K for the experiment.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/movielens_100k.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    metadata_path = prepare_movielens(args.raw_dir, args.output_dir, config.data)
    print(metadata_path.read_text(encoding="utf-8"))

