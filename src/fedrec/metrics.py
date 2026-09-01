from __future__ import annotations

import numpy as np
import pandas as pd


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype="int8")
    scores = np.asarray(scores, dtype="float64")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = pd.Series(scores).rank(method="average", ascending=True).to_numpy()
    positive_rank_sum = float(ranks[labels == 1].sum())
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def evaluate_topk(
    predictions: pd.DataFrame,
    score_column: str,
    k: int = 10,
) -> tuple[dict, pd.DataFrame]:
    required = {"party", "user_id", "item_id", "target_relevant", score_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    per_user = []
    for (party, user_id), frame in predictions.groupby(["party", "user_id"]):
        ranked = frame.sort_values(
            [score_column, "item_id"], ascending=[False, True], kind="stable"
        )
        top = ranked.head(k)
        hits = int(top["target_relevant"].sum())
        target_positive_count = int(frame["target_positive_count"].iloc[0])
        candidate_positive_count = int(frame["target_relevant"].sum())
        gains = top["target_relevant"].to_numpy(dtype="float64")
        discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
        dcg = float(np.sum(gains * discounts))
        ideal_count = min(k, target_positive_count)
        idcg = float(np.sum(discounts[:ideal_count])) if ideal_count else 0.0
        per_user.append(
            {
                "party": party,
                "user_id": int(user_id),
                "candidate_count": len(frame),
                "target_positive_count": target_positive_count,
                "candidate_positive_count": candidate_positive_count,
                "precision_at_k": hits / k,
                "recall_at_k": hits / max(1, target_positive_count),
                "hit_rate_at_k": float(hits > 0),
                "ndcg_at_k": dcg / idcg if idcg else 0.0,
                "candidate_recall": candidate_positive_count / max(1, target_positive_count),
                "candidate_hit_upper_bound": float(candidate_positive_count > 0),
                "user_auc": binary_auc(
                    frame["target_relevant"].to_numpy(), frame[score_column].to_numpy()
                ),
            }
        )
    per_user_frame = pd.DataFrame(per_user)
    metrics = {
        "user_count": int(len(per_user_frame)),
        "precision_at_10": float(per_user_frame["precision_at_k"].mean()),
        "recall_at_10": float(per_user_frame["recall_at_k"].mean()),
        "hit_rate_at_10": float(per_user_frame["hit_rate_at_k"].mean()),
        "ndcg_at_10": float(per_user_frame["ndcg_at_k"].mean()),
        "candidate_auc": binary_auc(
            predictions["target_relevant"].to_numpy(), predictions[score_column].to_numpy()
        ),
        "mean_user_auc": float(per_user_frame["user_auc"].mean()),
        "candidate_recall": float(per_user_frame["candidate_recall"].mean()),
        "candidate_hit_upper_bound": float(
            per_user_frame["candidate_hit_upper_bound"].mean()
        ),
        "average_candidate_count": float(per_user_frame["candidate_count"].mean()),
        "candidate_positive_rate": float(predictions["target_relevant"].mean()),
    }
    return metrics, per_user_frame

