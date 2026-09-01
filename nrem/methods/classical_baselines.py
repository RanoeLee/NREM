"""Factory-compatible classical product-level anomaly baselines."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from ..classical.features import ProductFeatureExtractor


class ClassicalBaselineMethod(nn.Module):
    """OCSVM and LOF behind the common method contract.

    The fitted sklearn pipeline is persisted as a method artifact next to the
    normal PyTorch checkpoint. A small registered buffer keeps checkpoint
    metadata compatible with the common runner; it is not an optimised model
    parameter.
    """

    SUPPORTED = {"ocsvm", "lof"}
    requires_reference = False
    uses_custom_training = True

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.variant = str((cfg.get("method") or {}).get("name", "")).lower()
        if self.variant not in self.SUPPORTED:
            raise ValueError(f"Unknown classical baseline: {self.variant!r}")
        self.channels = int((cfg.get("data") or {}).get("channels", 6))
        self.extractor = ProductFeatureExtractor(channels=self.channels)
        self.estimator = None
        self.register_buffer("fitted", torch.tensor(False, dtype=torch.bool))

    def _build_estimator(self, seed: int) -> Pipeline:
        params = self.cfg.get("classical", {}) or {}
        if self.variant == "ocsvm":
            detector = OneClassSVM(
                kernel=str(params.get("kernel", "rbf")),
                gamma=params.get("gamma", "scale"),
                nu=float(params.get("nu", 0.1)),
            )
        elif self.variant == "lof":
            detector = LocalOutlierFactor(
                n_neighbors=int(params.get("n_neighbors", 35)),
                metric=str(params.get("metric", "minkowski")),
                novelty=True,
                contamination="auto",
                n_jobs=int(params.get("n_jobs", 1)),
            )
        else:
            raise ValueError(f"Unknown classical baseline: {self.variant!r}")
        return Pipeline([("scale", RobustScaler()), ("detector", detector)])

    def _batch_features(self, batch: Dict) -> np.ndarray:
        x = batch["x"].detach().cpu().numpy()
        return self.extractor.transform([record for record in x])

    def training_epoch(self, loader, cfg: dict, device: str, epoch: int, optimizers=None):
        del device, optimizers
        if epoch != 0:
            raise RuntimeError("Classical baseline configs must set train.epochs to 1")
        features = []
        for batch in loader:
            features.append(self._batch_features(batch))
        matrix = np.concatenate(features, axis=0)
        seed = int((cfg.get("runtime") or {}).get("seed", 2025))
        self.estimator = self._build_estimator(seed)
        self.estimator.fit(matrix)
        self.fitted.fill_(True)
        return {
            "loss": 0.0,
            "n_train": int(matrix.shape[0]),
            "feature_dim": int(matrix.shape[1]),
            "labels_used_for_fit": False,
        }

    def save_artifacts(self, out_dir: str | Path) -> None:
        if self.estimator is None or not bool(self.fitted.item()):
            raise RuntimeError("Cannot save an unfitted classical baseline")
        joblib.dump(self.estimator, Path(out_dir) / "classical_model.joblib", compress=3)

    def load_artifacts(self, run_dir: str | Path) -> None:
        path = Path(run_dir) / "classical_model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing fitted classical model: {path}")
        self.estimator = joblib.load(path)
        self.fitted.fill_(True)

    @torch.no_grad()
    def score_batch(self, batch: Dict, score_mode=None, resa_ref=None) -> Tuple[torch.Tensor, Dict]:
        del score_mode, resa_ref
        if self.estimator is None or not bool(self.fitted.item()):
            raise RuntimeError("Classical baseline has not been fitted or loaded")
        features = self._batch_features(batch)
        scores = -np.asarray(self.estimator.decision_function(features), dtype=np.float32).reshape(-1)
        return torch.from_numpy(scores).to(batch["x"].device), {"score_mode": self.variant}
