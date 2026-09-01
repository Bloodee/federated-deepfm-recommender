from __future__ import annotations

import argparse
from pathlib import Path

from fedrec.config import ExperimentConfig
from fedrec.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run federated movie recommendation.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/movielens_100k"))
    parser.add_argument("--config", type=Path, default=Path("configs/movielens_100k.json"))
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = ExperimentConfig.from_json(args.config)
    report_path = run_experiment(
        raw_dir=args.raw_dir,
        work_dir=args.work_dir,
        config=config,
        prepare_only=args.prepare_only,
    )
    print(report_path.read_text(encoding="utf-8"))

