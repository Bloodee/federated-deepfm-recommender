from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    positive_rating: float = 4.0
    min_user_interactions: int = 50
    min_user_positive_rate: float = 0.60
    min_user_positives: int = 5
    min_user_negatives: int = 5
    validation_holdout_size: int = 5
    test_holdout_size: int = 10
    party_count: int = 2
    seed: int = 2026


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_mode: str = "controlled_500"
    mother_pool_size: int = 500
    recall_size: int = 100
    popularity_weight: float = 0.15
    itemcf_weight: float = 0.35
    content_weight: float = 0.15
    signed_itemcf_weight: float = 0.05
    usercf_weight: float = 0.30
    positive_history_limit: int = 50
    itemcf_neighbors: int = 160

    def weights(self) -> dict[str, float]:
        raw = {
            "popularity": self.popularity_weight,
            "itemcf": self.itemcf_weight,
            "content": self.content_weight,
            "signed_itemcf": self.signed_itemcf_weight,
            "usercf": self.usercf_weight,
        }
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("At least one retrieval weight must be positive.")
        return {name: value / total for name, value in raw.items()}


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 16
    hidden_units: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.14
    learning_rate: float = 5e-4
    l2_regularization: float = 2e-5
    epochs: int = 16
    batch_size: int = 256
    aggregate_freq: int = 1
    negatives_per_positive: int = 12
    positive_class_weight_cap: float = 2.5
    focal_gamma: float = 1.5
    bce_weight: float = 0.85
    item_id_dropout_rate: float = 0.05


@dataclass(frozen=True)
class RankingConfig:
    top_k: int = 10
    fusion_plans: tuple[tuple[float, float], ...] = (
        (1.00, 0.00),
        (0.75, 0.25),
        (0.55, 0.45),
        (0.50, 0.50),
        (0.45, 0.55),
        (0.25, 0.75),
    )


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model_payload = dict(payload.get("model", {}))
        if "hidden_units" in model_payload:
            model_payload["hidden_units"] = tuple(model_payload["hidden_units"])
        ranking_payload = dict(payload.get("ranking", {}))
        if "fusion_plans" in ranking_payload:
            ranking_payload["fusion_plans"] = tuple(
                tuple(plan) for plan in ranking_payload["fusion_plans"]
            )
        return cls(
            data=DataConfig(**payload.get("data", {})),
            retrieval=RetrievalConfig(**payload.get("retrieval", {})),
            model=ModelConfig(**model_payload),
            ranking=RankingConfig(**ranking_payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

