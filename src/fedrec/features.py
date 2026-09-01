from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .config import ModelConfig
from .data import MOVIE_GENRES, stable_integer
from .retrieval import HybridRetriever, build_genre_map


CATEGORICAL_FEATURES = (
    "item_feature_id",
    "age_bucket",
    "gender_id",
    "occupation_id",
    "primary_genre_id",
    "secondary_genre_id",
    "preferred_genre_id",
    "disliked_genre_id",
)

CONTINUOUS_FEATURES = (
    "user_history_log",
    "item_positive_rate",
    "item_history_log",
    "content_affinity",
    "negative_content_affinity",
    "preference_concentration",
)

MODEL_FEATURES = (*CATEGORICAL_FEATURES, *CONTINUOUS_FEATURES)


def _ranked_sample(
    values: set[int], count: int, seed: int, namespace: str, user_id: int
) -> list[int]:
    return sorted(
        values,
        key=lambda item_id: stable_integer(seed, namespace, user_id, item_id),
    )[:count]


def _largest_remainder_counts(total: int) -> dict[str, int]:
    weights = {
        "retrieval_hard": 0.50,
        "same_interest": 0.30,
        "local_popularity": 0.10,
        "catalog_random": 0.10,
    }
    raw = {name: total * weight for name, weight in weights.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(weights, key=lambda name: (-(raw[name] - counts[name]), name))
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def build_training_rows(
    history: pd.DataFrame,
    all_explicit: pd.DataFrame,
    retriever: HybridRetriever,
    model_config: ModelConfig,
    seed: int,
) -> pd.DataFrame:
    """Build balanced explicit + label-free sampled training rows.

    Held-out explicit interactions are excluded from negative sampling. The
    50/30/10/10 source mix mirrors the retained robust experiment.
    """
    catalog = set(retriever.catalog)
    known_by_user = {
        int(user_id): set(frame["item_id"].astype(int))
        for user_id, frame in all_explicit.groupby("user_id")
    }
    output: list[pd.DataFrame] = []
    for user_id, frame in history.groupby("user_id", sort=False):
        user_id = int(user_id)
        positives = frame[frame["label"].eq(1)].copy()
        explicit_negatives = frame[frame["label"].eq(0)].copy()
        positives = positives.assign(
            _order=[
                stable_integer(seed, "positive-cap", user_id, item_id)
                for item_id in positives["item_id"]
            ]
        ).nsmallest(64, "_order").drop(columns="_order")
        explicit_negatives = explicit_negatives.assign(
            _order=[
                stable_integer(seed, "explicit-negative-cap", user_id, item_id)
                for item_id in explicit_negatives["item_id"]
            ]
        ).nsmallest(32, "_order").drop(columns="_order")
        observed = pd.concat([positives, explicit_negatives], ignore_index=True)
        observed["negative_source"] = np.where(
            observed["label"].eq(1), "explicit_positive", "explicit_negative"
        )
        output.append(observed)

        desired = min(384, model_config.negatives_per_positive * len(positives))
        if desired <= 0:
            continue
        legal = catalog - known_by_user.get(user_id, set())
        scored = retriever.score(user_id, legal)
        hard_order = list(scored["item_id"].astype(int))
        preferred_genres: Counter[str] = Counter()
        for item_id in positives["item_id"].astype(int):
            preferred_genres.update(retriever.genre_map.get(item_id, ()))
        preferred = {genre for genre, _ in preferred_genres.most_common(3)}
        same_interest = {
            item_id
            for item_id in legal
            if preferred.intersection(retriever.genre_map.get(item_id, ()))
        }
        popularity_order = sorted(
            legal,
            key=lambda item_id: (-retriever.popularity.get(item_id, 0.0), item_id),
        )
        random_order = _ranked_sample(legal, len(legal), seed, "random-negative", user_id)
        source_order = {
            "retrieval_hard": hard_order,
            "same_interest": _ranked_sample(
                same_interest, len(same_interest), seed, "same-interest", user_id
            ),
            "local_popularity": popularity_order,
            "catalog_random": random_order,
        }
        counts = _largest_remainder_counts(desired)
        chosen: set[int] = set()
        sampled: list[dict] = []
        for source in (
            "retrieval_hard",
            "same_interest",
            "local_popularity",
            "catalog_random",
        ):
            need = counts[source]
            for item_id in source_order[source]:
                if item_id in chosen:
                    continue
                chosen.add(item_id)
                sampled.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "rating": np.nan,
                        "timestamp": 0,
                        "label": 0,
                        "negative_source": source,
                    }
                )
                need -= 1
                if need == 0:
                    break
        if len(sampled) < desired:
            for item_id in random_order:
                if item_id in chosen:
                    continue
                chosen.add(item_id)
                sampled.append(
                    {
                        "user_id": user_id,
                        "item_id": item_id,
                        "rating": np.nan,
                        "timestamp": 0,
                        "label": 0,
                        "negative_source": "catalog_random_fallback",
                    }
                )
                if len(sampled) == desired:
                    break
        output.append(pd.DataFrame(sampled))
    return pd.concat(output, ignore_index=True)


class FeatureBuilder:
    """Fit party-local statistics and convert user-item pairs to DeepFM features."""

    def __init__(self, history: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame):
        self.history = history.copy()
        self.users = users.set_index("user_id", drop=False)
        self.items = items.copy()
        self.genre_map = build_genre_map(items)
        self.genre_id = {genre: index + 1 for index, genre in enumerate(MOVIE_GENRES)}
        occupations = sorted(users["occupation"].astype(str).unique())
        self.occupation_id = {value: index + 1 for index, value in enumerate(occupations)}
        self._fit_profiles()
        self.cardinalities = {
            "item_feature_id": int(items["item_id"].max()) + 2,
            "age_bucket": 12,
            "gender_id": 3,
            "occupation_id": len(self.occupation_id) + 1,
            "primary_genre_id": len(self.genre_id) + 1,
            "secondary_genre_id": len(self.genre_id) + 1,
            "preferred_genre_id": len(self.genre_id) + 1,
            "disliked_genre_id": len(self.genre_id) + 1,
        }

    def _fit_profiles(self) -> None:
        item_counts = self.history.groupby("item_id").size()
        item_positive = self.history[self.history["label"].eq(1)].groupby("item_id").size()
        global_rate = float(self.history["label"].mean())
        self.item_count = item_counts.to_dict()
        self.item_positive_rate = {
            int(item_id): float((item_positive.get(item_id, 0) + 10 * global_rate) / (count + 10))
            for item_id, count in item_counts.items()
        }
        self.user_profiles: dict[int, dict] = {}
        for user_id, frame in self.history.groupby("user_id"):
            positive_genres: Counter[str] = Counter()
            negative_genres: Counter[str] = Counter()
            for item_id, label in frame[["item_id", "label"]].itertuples(index=False):
                target = positive_genres if int(label) == 1 else negative_genres
                target.update(self.genre_map.get(int(item_id), ()))
            preferred = positive_genres.most_common()
            disliked = negative_genres.most_common()
            total_positive = max(1, sum(positive_genres.values()))
            self.user_profiles[int(user_id)] = {
                "history_count": len(frame),
                "positive_genres": positive_genres,
                "negative_genres": negative_genres,
                "preferred": preferred[0][0] if preferred else "unknown",
                "disliked": disliked[0][0] if disliked else "unknown",
                "concentration": preferred[0][1] / total_positive if preferred else 0.0,
            }

    def transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for record in pairs.to_dict("records"):
            user_id, item_id = int(record["user_id"]), int(record["item_id"])
            profile = self.user_profiles[user_id]
            demographic = self.users.loc[user_id]
            genres = self.genre_map.get(item_id, ("unknown",))
            primary = genres[0] if genres else "unknown"
            secondary = genres[1] if len(genres) > 1 else "unknown"
            positive = profile["positive_genres"]
            negative = profile["negative_genres"]
            denominator = max(1, sum(positive.values()))
            negative_denominator = max(1, sum(negative.values()))
            row = dict(record)
            row.update(
                {
                    "item_feature_id": item_id + 1,
                    "age_bucket": min(11, max(1, int(demographic["age"]) // 10 + 1)),
                    "gender_id": 1 if str(demographic["gender"]).upper() == "M" else 2,
                    "occupation_id": self.occupation_id.get(str(demographic["occupation"]), 0),
                    "primary_genre_id": self.genre_id.get(primary, 0),
                    "secondary_genre_id": self.genre_id.get(secondary, 0),
                    "preferred_genre_id": self.genre_id.get(profile["preferred"], 0),
                    "disliked_genre_id": self.genre_id.get(profile["disliked"], 0),
                    "user_history_log": float(np.log1p(profile["history_count"])),
                    "item_positive_rate": self.item_positive_rate.get(item_id, 0.5),
                    "item_history_log": float(np.log1p(self.item_count.get(item_id, 0))),
                    "content_affinity": sum(positive.get(genre, 0) for genre in genres) / denominator,
                    "negative_content_affinity": sum(negative.get(genre, 0) for genre in genres) / negative_denominator,
                    "preference_concentration": float(profile["concentration"]),
                }
            )
            rows.append(row)
        output = pd.DataFrame(rows)
        for name in CATEGORICAL_FEATURES:
            output[name] = output[name].astype("int32")
        for name in CONTINUOUS_FEATURES:
            output[name] = output[name].astype("float32")
        return output


def apply_item_id_dropout(
    training: pd.DataFrame, rate: float, seed: int
) -> pd.DataFrame:
    output = training.copy()
    if rate <= 0:
        return output
    mask = np.asarray(
        [
            stable_integer(seed, "item-dropout", user_id, item_id, index) % 10_000
            < round(rate * 10_000)
            for index, (user_id, item_id) in enumerate(
                output[["user_id", "item_id"]].itertuples(index=False)
            )
        ]
    )
    output.loc[mask, "item_feature_id"] = 0
    return output

