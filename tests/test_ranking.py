import pandas as pd

from fedrec.ranking import rank_fusion, select_fusion_plan


def make_predictions() -> pd.DataFrame:
    rows = []
    for user_id in (1, 2):
        for item_id in range(1, 21):
            relevant = int(item_id == 1)
            rows.append(
                {
                    "party": "party_1",
                    "user_id": user_id,
                    "item_id": item_id,
                    "deepfm_score": float(item_id),
                    "retrieval_score": float(21 - item_id),
                    "target_relevant": relevant,
                    "target_positive_count": 1,
                }
            )
    return pd.DataFrame(rows)


def test_rank_fusion_is_label_free_and_validation_selects_gain():
    predictions = make_predictions()
    fused = rank_fusion(predictions, 0.25, 0.75)
    assert "final_score" in fused
    plan, audit = select_fusion_plan(
        predictions,
        plans=((1.0, 0.0), (0.25, 0.75), (0.0, 1.0)),
    )
    assert plan["retrieval_weight"] > 0
    assert plan["selection_reason"] == "strict_validation_hit_rate_gain"
    assert audit["selected"].sum() == 1

