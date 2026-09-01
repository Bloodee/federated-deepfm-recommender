from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig


MOVIE_GENRES = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)


def stable_integer(seed: int, *parts: object) -> int:
    value = ":".join(str(part) for part in (seed, *parts))
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8], "big"
    )


def load_movielens_100k(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only the three MovieLens files required by the experiment."""
    raw_dir = Path(raw_dir)
    required = ("u.data", "u.user", "u.item")
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"MovieLens 100K is incomplete under {raw_dir}: missing {missing}"
        )

    interactions = pd.read_csv(
        raw_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": "int32", "item_id": "int32", "rating": "float32"},
    )
    users = pd.read_csv(
        raw_dir / "u.user",
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
    )
    item_columns = [
        "item_id",
        "item_title",
        "release_date",
        "video_release_date",
        "imdb_url",
        *MOVIE_GENRES,
    ]
    items = pd.read_csv(
        raw_dir / "u.item",
        sep="|",
        names=item_columns,
        encoding="latin-1",
    )
    release_date = pd.to_datetime(
        items["release_date"], format="%d-%b-%Y", errors="coerce"
    )
    items["release_year"] = release_date.dt.year.fillna(0).astype("int16")
    items = items[["item_id", "item_title", "release_year", *MOVIE_GENRES]].copy()
    return interactions, users, items


def select_users(interactions: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Select the high-intent cohort used by the retained 270-user experiment."""
    frame = interactions.copy()
    frame["label"] = frame["rating"].ge(config.positive_rating).astype("int8")
    grouped = frame.groupby("user_id")["label"]
    stats = pd.DataFrame(
        {
            "interaction_count": grouped.size(),
            "positive_count": grouped.sum(),
            "positive_rate": grouped.mean(),
        }
    )
    stats["negative_count"] = stats["interaction_count"] - stats["positive_count"]
    eligible = stats[
        stats["interaction_count"].ge(config.min_user_interactions)
        & stats["positive_rate"].ge(config.min_user_positive_rate)
        & stats["positive_count"].ge(config.min_user_positives)
        & stats["negative_count"].ge(config.min_user_negatives)
    ].index
    return frame[frame["user_id"].isin(eligible)].copy()


def _split_one_user(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    validation_size = config.validation_holdout_size
    test_size = config.test_holdout_size
    holdout_size = validation_size + test_size
    positive_count = int(frame["label"].eq(1).sum())
    negative_count = int(frame["label"].eq(0).sum())
    if len(frame) < holdout_size + 2 or min(positive_count, negative_count) < 3:
        raise ValueError(f"User {int(frame['user_id'].iloc[0])} cannot support the fixed holdout.")

    minimum_positive = max(2, holdout_size - negative_count + 1)
    maximum_positive = min(positive_count - 1, holdout_size - 2)
    desired_positive = round(holdout_size * positive_count / len(frame))
    positive_holdout = int(np.clip(desired_positive, minimum_positive, maximum_positive))
    negative_holdout = holdout_size - positive_holdout

    minimum_validation_positive = max(
        1,
        positive_holdout - (test_size - 1),
        validation_size - negative_holdout + 1,
    )
    maximum_validation_positive = min(validation_size - 1, positive_holdout - 1)
    desired_validation_positive = round(
        validation_size * positive_holdout / holdout_size
    )
    validation_positive = int(
        np.clip(
            desired_validation_positive,
            minimum_validation_positive,
            maximum_validation_positive,
        )
    )
    counts = {
        1: (validation_positive, positive_holdout - validation_positive),
        0: (
            validation_size - validation_positive,
            test_size - (positive_holdout - validation_positive),
        ),
    }

    parts: list[pd.DataFrame] = []
    user_id = int(frame["user_id"].iloc[0])
    for label in (1, 0):
        part = frame[frame["label"].eq(label)].copy()
        part["_order"] = [
            stable_integer(config.seed, "holdout", user_id, label, item_id, timestamp)
            for item_id, timestamp in part[["item_id", "timestamp"]].itertuples(index=False)
        ]
        part = part.sort_values("_order", kind="stable").drop(columns="_order")
        validation_count, test_count = counts[label]
        split = np.full(len(part), "train", dtype=object)
        split[:validation_count] = "validation"
        split[validation_count : validation_count + test_count] = "test"
        part["split"] = split
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fixed_holdout_split(interactions: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    output = pd.concat(
        [_split_one_user(frame, config) for _, frame in interactions.groupby("user_id")],
        ignore_index=True,
    )
    output["party"] = [
        f"party_{stable_integer(config.seed, 'party', user_id) % config.party_count + 1}"
        for user_id in output["user_id"]
    ]
    return output.sort_values(
        ["party", "user_id", "split", "timestamp", "item_id"], kind="stable"
    ).reset_index(drop=True)


def prepare_movielens(raw_dir: str | Path, output_dir: str | Path, config: DataConfig) -> Path:
    """Create the minimal processed tables consumed by retrieval and training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions, users, items = load_movielens_100k(raw_dir)
    selected = select_users(interactions, config)
    prepared = fixed_holdout_split(selected, config)
    selected_users = users[users["user_id"].isin(prepared["user_id"].unique())].copy()

    prepared.to_csv(output_dir / "interactions.csv", index=False)
    selected_users.to_csv(output_dir / "users.csv", index=False)
    items.to_csv(output_dir / "items.csv", index=False)
    metadata = {
        "dataset": "MovieLens 100K",
        "user_count": int(prepared["user_id"].nunique()),
        "catalog_item_count": int(items["item_id"].nunique()),
        "interaction_count": int(len(prepared)),
        "party_users": {
            party: int(frame["user_id"].nunique())
            for party, frame in prepared.groupby("party")
        },
        "split_rows": prepared["split"].value_counts().sort_index().to_dict(),
        "config": config.__dict__,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata_path

