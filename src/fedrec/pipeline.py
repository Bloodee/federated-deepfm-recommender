from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .data import prepare_movielens
from .features import (
    MODEL_FEATURES,
    FeatureBuilder,
    apply_item_id_dropout,
    build_training_rows,
)
from .federated import FederatedTrainer
from .metrics import evaluate_topk
from .ranking import rank_fusion, select_fusion_plan
from .retrieval import HybridRetriever, build_candidate_universe


def _history_splits(stage: str) -> tuple[str, ...]:
    if stage == "validation":
        return ("train",)
    if stage == "test":
        return ("train", "validation")
    raise ValueError("Stage must be validation or test.")


def prepare_stage(
    processed_dir: str | Path,
    output_dir: str | Path,
    stage: str,
    config: ExperimentConfig,
) -> tuple[dict[str, str], dict[str, pd.DataFrame], dict[str, int]]:
    processed_dir, output_dir = Path(processed_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    interactions = pd.read_csv(processed_dir / "interactions.csv")
    users = pd.read_csv(processed_dir / "users.csv")
    items = pd.read_csv(processed_dir / "items.csv")
    history_names = _history_splits(stage)
    training_paths: dict[str, str] = {}
    candidates: dict[str, pd.DataFrame] = {}
    cardinalities: dict[str, int] | None = None

    for party, party_all in interactions.groupby("party", sort=True):
        history = party_all[party_all["split"].isin(history_names)].copy()
        targets = party_all[party_all["split"].eq(stage)].copy()
        retriever = HybridRetriever(
            history, items, config.retrieval, seed=config.data.seed
        )
        feature_builder = FeatureBuilder(history, users, items)
        cardinalities = feature_builder.cardinalities
        training = build_training_rows(
            history,
            party_all,
            retriever,
            config.model,
            seed=config.data.seed,
        )
        training = feature_builder.transform(training)
        training = apply_item_id_dropout(
            training, config.model.item_id_dropout_rate, config.data.seed
        )
        training_path = output_dir / f"{stage}_{party}_train.csv"
        training[[*MODEL_FEATURES, "label"]].to_csv(training_path, index=False)
        training_paths[str(party)] = str(training_path.resolve())

        all_explicit_by_user = {
            int(user_id): set(frame["item_id"].astype(int))
            for user_id, frame in party_all.groupby("user_id")
        }
        history_by_user = {
            int(user_id): set(frame["item_id"].astype(int))
            for user_id, frame in history.groupby("user_id")
        }
        target_by_user = {
            int(user_id): frame.set_index("item_id")["label"].astype(int).to_dict()
            for user_id, frame in targets.groupby("user_id")
        }
        recalled_frames = []
        for user_id, target_labels in target_by_user.items():
            target_items = set(map(int, target_labels))
            other_explicit = all_explicit_by_user[user_id] - target_items
            universe = build_candidate_universe(
                user_id=user_id,
                catalog_items=items["item_id"],
                history_items=history_by_user.get(user_id, set()),
                target_items=target_items,
                other_explicit_items=other_explicit,
                config=config.retrieval,
                seed=config.data.seed,
            )
            recalled = retriever.retrieve(user_id, universe)
            recalled["target_relevant"] = recalled["item_id"].map(target_labels).fillna(0).astype("int8")
            recalled["target_positive_count"] = sum(target_labels.values())
            recalled["candidate_source"] = recalled["item_id"].isin(target_items).map(
                {True: "heldout_explicit", False: "unobserved"}
            )
            recalled_frames.append(recalled)
        party_candidates = pd.concat(recalled_frames, ignore_index=True)
        party_candidates = feature_builder.transform(party_candidates)
        candidate_path = output_dir / f"{stage}_{party}_candidates.csv"
        party_candidates.to_csv(candidate_path, index=False)
        candidates[str(party)] = party_candidates

    if cardinalities is None:
        raise ValueError("No parties were prepared.")
    return training_paths, candidates, cardinalities


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(
    raw_dir: str | Path,
    work_dir: str | Path,
    config: ExperimentConfig,
    prepare_only: bool = False,
) -> Path:
    """Run data preparation, validation selection and frozen test evaluation."""
    work_dir = Path(work_dir)
    processed_dir = work_dir / "processed"
    prepared_dir = work_dir / "prepared"
    results_dir = work_dir / "results"
    prepare_movielens(raw_dir, processed_dir, config.data)
    stage_inputs = {}
    for stage in ("validation", "test"):
        stage_inputs[stage] = prepare_stage(
            processed_dir, prepared_dir, stage, config
        )
    if prepare_only:
        marker = results_dir / "prepare_only.json"
        _write_json(marker, {"status": "prepared", "config": config.to_dict()})
        return marker

    validation_paths, validation_candidates, cardinalities = stage_inputs["validation"]
    validation_predictions = FederatedTrainer(
        config.model, cardinalities, seed=config.data.seed
    ).fit_predict(validation_paths, validation_candidates, results_dir / "validation")
    plan, trials = select_fusion_plan(
        validation_predictions, config.ranking.fusion_plans, config.ranking.top_k
    )
    trials.to_csv(results_dir / "validation_fusion_trials.csv", index=False)
    _write_json(results_dir / "selected_fusion.json", plan)

    test_paths, test_candidates, cardinalities = stage_inputs["test"]
    test_predictions = FederatedTrainer(
        config.model, cardinalities, seed=config.data.seed + 1
    ).fit_predict(test_paths, test_candidates, results_dir / "test")
    final_predictions = rank_fusion(
        test_predictions, plan["deepfm_weight"], plan["retrieval_weight"]
    )
    metrics, per_user = evaluate_topk(
        final_predictions, "final_score", config.ranking.top_k
    )
    final_predictions.to_csv(results_dir / "test_predictions.csv", index=False)
    per_user.to_csv(results_dir / "test_per_user_metrics.csv", index=False)
    report = {
        "protocol": "controlled pool -> natural five-channel recall -> federated DeepFM -> validation-frozen rank fusion",
        "config": config.to_dict(),
        "selected_fusion": plan,
        "test_metrics": metrics,
        "evaluation_boundary": (
            "Controlled mode guarantees held-out items only in the mother pool; "
            "labels never enter retrieval, DeepFM features, fusion selection, or test ranking."
        ),
    }
    report_path = results_dir / "test_metrics.json"
    _write_json(report_path, report)
    return report_path

