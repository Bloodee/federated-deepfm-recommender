from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig
from .deepfm import build_model_factory
from .features import MODEL_FEATURES


def load_secretflow_runtime():
    try:
        import secretflow as sf
        from secretflow import reveal
        from secretflow.security.aggregation import SecureAggregator
    except ImportError as error:
        raise RuntimeError(
            "SecretFlow is required for federated training. Use the validated "
            "Python 3.10 / TensorFlow 2.12 / SecretFlow 1.11 environment."
        ) from error
    try:
        from secretflow_fl.ml.nn.fl.fl_model import FLModel
    except ImportError:
        from secretflow.ml.nn import FLModel
    return sf, reveal, SecureAggregator, FLModel


def _prediction_inputs(
    candidates: dict[str, pd.DataFrame],
    work_dir: Path,
    batch_size: int,
) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    target_rows = math.ceil(max(map(len, candidates.values())) / batch_size) * batch_size
    paths: dict[str, str] = {}
    for party, frame in candidates.items():
        padding = target_rows - len(frame)
        padded = frame
        if padding:
            padded = pd.concat(
                [frame, frame.iloc[np.arange(padding) % len(frame)]], ignore_index=True
            )
        path = work_dir / f"{party}_predict.csv"
        padded[list(MODEL_FEATURES)].to_csv(path, index=False)
        paths[party] = str(path.resolve())
    return paths, candidates


class FederatedTrainer:
    """Two-party horizontal FedAvgW training backed by SecretFlow."""

    def __init__(
        self,
        model_config: ModelConfig,
        cardinalities: dict[str, int],
        seed: int = 2026,
        ray_num_cpus: int = 8,
    ) -> None:
        self.model_config = model_config
        self.cardinalities = cardinalities
        self.seed = seed
        self.ray_num_cpus = ray_num_cpus

    def fit_predict(
        self,
        training_paths: dict[str, str | Path],
        candidates: dict[str, pd.DataFrame],
        output_dir: str | Path,
    ) -> pd.DataFrame:
        if set(training_paths) != set(candidates):
            raise ValueError("Training and candidate parties must match.")
        sf, reveal, SecureAggregator, FLModel = load_secretflow_runtime()
        parties = sorted(training_paths)
        server_name = "server"
        try:
            sf.shutdown(barrier_on_shutdown=True)
        except Exception:
            pass
        sf.init(
            [*parties, server_name],
            address="local",
            num_cpus=self.ray_num_cpus,
            omp_num_threads=1,
            include_dashboard=False,
            log_to_driver=True,
        )
        devices = {party: sf.PYU(party) for party in parties}
        server = sf.PYU(server_name)
        model = FLModel(
            server=server,
            device_list=list(devices.values()),
            model=build_model_factory(
                self.cardinalities, self.model_config, self.seed
            ),
            aggregator=SecureAggregator(server, list(devices.values())),
            strategy="fed_avg_w",
            backend="tensorflow",
        )
        training_frames = {
            party: pd.read_csv(path) for party, path in training_paths.items()
        }
        positive_rows = sum(int(frame["label"].sum()) for frame in training_frames.values())
        negative_rows = sum(int(frame["label"].eq(0).sum()) for frame in training_frames.values())
        positive_weight = float(
            np.clip(
                negative_rows / max(1, positive_rows),
                1.0,
                self.model_config.positive_class_weight_cap,
            )
        )
        device_train_paths = {
            devices[party]: str(Path(path).resolve())
            for party, path in training_paths.items()
        }
        output_dir = Path(output_dir)
        prediction_paths, original = _prediction_inputs(
            candidates, output_dir / "prediction_inputs", self.model_config.batch_size
        )
        device_prediction_paths = {
            devices[party]: path for party, path in prediction_paths.items()
        }
        try:
            model.fit(
                x=device_train_paths,
                y="label",
                epochs=self.model_config.epochs,
                batch_size=self.model_config.batch_size,
                aggregate_freq=self.model_config.aggregate_freq,
                sampler_method="batch",
                shuffle=True,
                class_weight={0: 1.0, 1: positive_weight},
                random_seed=self.seed,
                verbose=1,
            )
            remote = model.predict(
                x=device_prediction_paths, batch_size=self.model_config.batch_size
            )
            outputs = []
            for party, device in devices.items():
                values = np.asarray(reveal(remote[device]), dtype="float64").reshape(-1)
                frame = original[party].copy().reset_index(drop=True)
                frame["party"] = party
                frame["deepfm_score"] = np.clip(values[: len(frame)], 0.0, 1.0)
                outputs.append(frame)
            return pd.concat(outputs, ignore_index=True)
        finally:
            try:
                sf.shutdown(barrier_on_shutdown=True)
            except Exception:
                pass

