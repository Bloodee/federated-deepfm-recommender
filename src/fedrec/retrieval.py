from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import RetrievalConfig
from .data import MOVIE_GENRES, stable_integer


RETRIEVAL_SCORE_COLUMNS = (
    "popularity_score",
    "itemcf_score",
    "content_score",
    "signed_itemcf_score",
    "usercf_score",
    "retrieval_score",
)


def build_genre_map(items: pd.DataFrame) -> dict[int, tuple[str, ...]]:
    return {
        int(row.item_id): tuple(
            genre for genre in MOVIE_GENRES if int(getattr(row, genre.replace("-", "_"), 0)) == 1
        )
        for row in items.rename(columns={"Film-Noir": "Film_Noir", "Sci-Fi": "Sci_Fi"}).itertuples()
    }


def _minmax(raw: dict[int, float], allowed: list[int]) -> dict[int, float]:
    if not allowed:
        return {}
    values = np.asarray([raw.get(item_id, 0.0) for item_id in allowed], dtype="float64")
    minimum, maximum = float(values.min()), float(values.max())
    if maximum <= minimum:
        return {item_id: 0.0 for item_id in allowed}
    normalized = (values - minimum) / (maximum - minimum)
    return dict(zip(allowed, normalized.astype(float)))


def build_candidate_universe(
    user_id: int,
    catalog_items: Iterable[int],
    history_items: set[int],
    target_items: set[int],
    other_explicit_items: set[int],
    config: RetrievalConfig,
    seed: int,
) -> list[int]:
    """Build either a controlled 500-item pool or the complete unseen catalog.

    In controlled mode, held-out items are guaranteed only in the mother pool.
    Their labels never affect retrieval scoring and they are not forced into the
    recalled Top-N set.
    """
    catalog = {int(item_id) for item_id in catalog_items}
    unseen = catalog - {int(item_id) for item_id in history_items}
    if config.candidate_mode == "full_catalog":
        return sorted(unseen - {int(item_id) for item_id in other_explicit_items})
    if config.candidate_mode != "controlled_500":
        raise ValueError(f"Unknown candidate mode: {config.candidate_mode}")
    targets = sorted(unseen & {int(item_id) for item_id in target_items})
    if len(targets) > config.mother_pool_size:
        raise ValueError("Held-out targets exceed the configured mother-pool size.")
    filler = unseen - set(targets) - {int(item_id) for item_id in other_explicit_items}
    ranked_filler = sorted(
        filler,
        key=lambda item_id: stable_integer(seed, "mother-pool", user_id, item_id),
    )
    selected = targets + ranked_filler[: config.mother_pool_size - len(targets)]
    if len(selected) != config.mother_pool_size:
        raise ValueError(
            f"User {user_id} has only {len(selected)} legal controlled candidates."
        )
    return selected


class HybridRetriever:
    """Five-channel retriever fitted on one party's local training history."""

    def __init__(
        self,
        history: pd.DataFrame,
        items: pd.DataFrame,
        config: RetrievalConfig,
        seed: int = 2026,
    ) -> None:
        required = {"user_id", "item_id", "label"}
        missing = required - set(history.columns)
        if missing:
            raise ValueError(f"History is missing columns: {sorted(missing)}")
        self.history = history.copy()
        self.items = items.copy()
        self.config = config
        self.seed = seed
        self.catalog = sorted(items["item_id"].astype(int).unique())
        self.genre_map = build_genre_map(items)
        self.seen = {
            int(user_id): set(frame["item_id"].astype(int))
            for user_id, frame in history.groupby("user_id")
        }
        positive = history[history["label"].eq(1)].sort_values(
            ["user_id", "timestamp", "item_id"]
        )
        self.liked = {
            int(user_id): list(
                frame["item_id"].drop_duplicates(keep="last").astype(int).tail(
                    config.positive_history_limit
                )
            )
            for user_id, frame in positive.groupby("user_id")
        }
        self._build_popularity()
        self._build_content_index()
        self._build_itemcf()
        self._build_signed_cf()

    def _build_popularity(self) -> None:
        counts = self.history.groupby("item_id").size()
        positives = self.history[self.history["label"].eq(1)].groupby("item_id").size()
        global_rate = float(self.history["label"].mean())
        smoothing = 6.0
        self.popularity = {
            item_id: float(
                ((positives.get(item_id, 0) + smoothing * global_rate)
                 / (counts.get(item_id, 0) + smoothing))
                * np.log1p(counts.get(item_id, 0))
            )
            for item_id in self.catalog
        }

    def _build_content_index(self) -> None:
        token_items: dict[str, list[int]] = defaultdict(list)
        for item_id, genres in self.genre_map.items():
            for genre in genres:
                token_items[genre].append(item_id)
        total = max(1, len(self.catalog))
        self.genre_items = dict(token_items)
        self.genre_idf = {
            genre: float(np.log1p(total / max(1, len(item_ids))))
            for genre, item_ids in token_items.items()
        }

    def _build_itemcf(self) -> None:
        support: Counter[int] = Counter()
        cooccurrence: Counter[tuple[int, int]] = Counter()
        for liked in self.liked.values():
            values = sorted(set(liked))
            support.update(values)
            for index, left in enumerate(values):
                for right in values[index + 1 :]:
                    cooccurrence[(left, right)] += 1
        neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (left, right), count in cooccurrence.items():
            similarity = count / np.sqrt(max(1, support[left]) * max(1, support[right]))
            neighbors[left].append((right, float(similarity)))
            neighbors[right].append((left, float(similarity)))
        self.item_neighbors = {
            item_id: sorted(values, key=lambda pair: (-pair[1], pair[0]))[
                : self.config.itemcf_neighbors
            ]
            for item_id, values in neighbors.items()
        }

    def _build_signed_cf(self) -> None:
        user_ids = sorted(self.history["user_id"].astype(int).unique())
        user_index = {user_id: index for index, user_id in enumerate(user_ids)}
        item_index = {item_id: index for index, item_id in enumerate(self.catalog)}
        signed = np.zeros((len(user_ids), len(self.catalog)), dtype="float32")
        positive = np.zeros_like(signed)
        negative = np.zeros_like(signed)
        for user_id, item_id, label in self.history[
            ["user_id", "item_id", "label"]
        ].itertuples(index=False):
            if int(item_id) not in item_index:
                continue
            row, column = user_index[int(user_id)], item_index[int(item_id)]
            if int(label) == 1:
                signed[row, column] = 1.0
                positive[row, column] = 1.0
            else:
                signed[row, column] = -0.65
                negative[row, column] = 1.0

        item_norm = np.linalg.norm(signed, axis=0)
        item_denominator = np.outer(item_norm, item_norm)
        item_similarity = signed.T @ signed
        np.divide(
            item_similarity,
            item_denominator,
            out=item_similarity,
            where=item_denominator > 0,
        )
        np.fill_diagonal(item_similarity, 0.0)
        signed_prediction = signed @ item_similarity

        user_norm = np.linalg.norm(positive, axis=1)
        user_denominator = np.outer(user_norm, user_norm)
        user_similarity = positive @ positive.T
        np.divide(
            user_similarity,
            user_denominator,
            out=user_similarity,
            where=user_denominator > 0,
        )
        np.fill_diagonal(user_similarity, 0.0)
        user_similarity = np.maximum(user_similarity, 0.0)
        similarity_sum = np.maximum(user_similarity.sum(axis=1, keepdims=True), 1e-9)
        user_prediction = (
            user_similarity @ positive - 0.35 * (user_similarity @ negative)
        ) / similarity_sum
        self.signed_itemcf = {
            user_id: dict(zip(self.catalog, signed_prediction[user_index[user_id]].astype(float)))
            for user_id in user_ids
        }
        self.usercf = {
            user_id: dict(zip(self.catalog, user_prediction[user_index[user_id]].astype(float)))
            for user_id in user_ids
        }

    def _positive_itemcf(self, liked: list[int], excluded: set[int]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for source in liked:
            for item_id, similarity in self.item_neighbors.get(source, ()):
                if item_id not in excluded:
                    scores[item_id] += similarity
        return dict(scores)

    def _content(self, liked: list[int], excluded: set[int]) -> dict[int, float]:
        profile: Counter[str] = Counter()
        for item_id in liked:
            for genre in self.genre_map.get(item_id, ()):
                profile[genre] += self.genre_idf.get(genre, 0.0)
        scores: dict[int, float] = defaultdict(float)
        for genre, strength in profile.most_common(20):
            contribution = strength * self.genre_idf.get(genre, 0.0)
            for item_id in self.genre_items.get(genre, ()):
                if item_id not in excluded:
                    scores[item_id] += contribution
        return dict(scores)

    def score(self, user_id: int, allowed_items: Iterable[int]) -> pd.DataFrame:
        user_id = int(user_id)
        allowed = sorted({int(item_id) for item_id in allowed_items})
        excluded = self.seen.get(user_id, set())
        allowed = [item_id for item_id in allowed if item_id not in excluded]
        liked = self.liked.get(user_id, [])
        channels = {
            "popularity": _minmax(self.popularity, allowed),
            "itemcf": _minmax(self._positive_itemcf(liked, excluded), allowed),
            "content": _minmax(self._content(liked, excluded), allowed),
            "signed_itemcf": _minmax(self.signed_itemcf.get(user_id, {}), allowed),
            "usercf": _minmax(self.usercf.get(user_id, {}), allowed),
        }
        weights = self.config.weights()
        rows = []
        for item_id in allowed:
            values = {name: float(scores.get(item_id, 0.0)) for name, scores in channels.items()}
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    **{f"{name}_score": value for name, value in values.items()},
                    "retrieval_score": sum(weights[name] * values[name] for name in weights),
                }
            )
        output = pd.DataFrame(rows)
        if output.empty:
            return output
        output["_tie"] = [
            stable_integer(self.seed, "retrieval", user_id, item_id)
            for item_id in output["item_id"]
        ]
        output = output.sort_values(
            ["retrieval_score", "_tie"], ascending=[False, True], kind="stable"
        ).drop(columns="_tie")
        output["retrieval_rank"] = np.arange(1, len(output) + 1)
        return output.reset_index(drop=True)

    def retrieve(self, user_id: int, allowed_items: Iterable[int]) -> pd.DataFrame:
        return self.score(user_id, allowed_items).head(self.config.recall_size).copy()

