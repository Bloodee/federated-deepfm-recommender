from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import evaluate_topk


def rank_fusion(
    predictions: pd.DataFrame,
    deepfm_weight: float,
    retrieval_weight: float,
) -> pd.DataFrame:
    """Fuse user-local ranks; labels and raw candidate positions are never read."""
    total = deepfm_weight + retrieval_weight
    if deepfm_weight < 0 or retrieval_weight < 0 or total <= 0:
        raise ValueError("Fusion weights must be non-negative and sum above zero.")
    deepfm_weight, retrieval_weight = deepfm_weight / total, retrieval_weight / total
    outputs = []
    for _, frame in predictions.groupby(["party", "user_id"], sort=False):
        frame = frame.copy()
        count = len(frame)
        denominator = max(1, count - 1)
        for name, score_column in (
            ("deepfm", "deepfm_score"),
            ("retrieval", "retrieval_score"),
        ):
            order = frame.sort_values(
                [score_column, "item_id"], ascending=[False, True], kind="stable"
            ).index
            ranks = pd.Series(
                np.arange(1, count + 1, dtype="float64"), index=order
            )
            frame[f"{name}_rank_component"] = (
                1.0 - (ranks.reindex(frame.index) - 1.0) / denominator
            )
        frame["final_score"] = (
            deepfm_weight * frame["deepfm_rank_component"]
            + retrieval_weight * frame["retrieval_rank_component"]
        )
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def select_fusion_plan(
    validation_predictions: pd.DataFrame,
    plans: tuple[tuple[float, float], ...],
    top_k: int = 10,
) -> tuple[dict, pd.DataFrame]:
    """Select on validation only and freeze the winner for test."""
    trials = []
    for deepfm_weight, retrieval_weight in plans:
        fused = rank_fusion(
            validation_predictions, deepfm_weight, retrieval_weight
        )
        metrics, _ = evaluate_topk(fused, "final_score", top_k)
        trials.append(
            {
                "deepfm_weight": deepfm_weight,
                "retrieval_weight": retrieval_weight,
                **metrics,
            }
        )
    audit = pd.DataFrame(trials)
    pure = audit[
        np.isclose(audit["deepfm_weight"], 1.0)
        & np.isclose(audit["retrieval_weight"], 0.0)
    ].iloc[0]
    improved = audit[audit["hit_rate_at_10"].gt(float(pure["hit_rate_at_10"]))]
    if improved.empty:
        selected = pure
        reason = "pure_deepfm_fallback"
    else:
        selected = improved.sort_values(
            [
                "hit_rate_at_10",
                "mean_user_auc",
                "candidate_auc",
                "ndcg_at_10",
                "precision_at_10",
                "deepfm_weight",
            ],
            ascending=[False, False, False, False, False, False],
            kind="stable",
        ).iloc[0]
        reason = "strict_validation_hit_rate_gain"
    plan = {
        "deepfm_weight": float(selected["deepfm_weight"]),
        "retrieval_weight": float(selected["retrieval_weight"]),
        "selection_reason": reason,
        "validation_hit_rate_at_10": float(selected["hit_rate_at_10"]),
    }
    audit["selected"] = (
        np.isclose(audit["deepfm_weight"], plan["deepfm_weight"])
        & np.isclose(audit["retrieval_weight"], plan["retrieval_weight"])
    )
    return plan, audit

