"""DeepSVDD baseline adapted to the NREM engine interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.backbone import ConvWindowAE, MLPWindowAE
from ..modules.windowing import windowize


@dataclass
class DeepSVDDForward:
    loss: Optional[torch.Tensor]
    sample_score: torch.Tensor
    window_energy: torch.Tensor
    rec_window: torch.Tensor
    u1_window: torch.Tensor
    u2_window: torch.Tensor
    logs: Dict[str, float]


class DeepSVDDMethod(nn.Module):
    """One-class DeepSVDD over window-level time-series embeddings.

    The class intentionally follows the small method interface used by
    `NREMMethod`: `training_step()` returns a scalar loss and logs, while
    `score_batch()` returns sample-level anomaly scores where larger means more
    anomalous.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        ncfg = cfg.get("nrem", {}) or {}
        dcfg = cfg.get("data", {}) or {}
        mcfg = cfg.get("model", {}) or {}
        svdd = cfg.get("deepsvdd", {}) or {}

        self.channels = int(dcfg.get("channels", mcfg.get("channels", 6)))
        self.window_size = int(ncfg.get("window_size", svdd.get("window_size", 1024)))
        self.stride = int(ncfg.get("stride", svdd.get("stride", 512)))
        self.input_dim = self.channels * self.window_size
        self.score_aggregation = str(svdd.get("score_aggregation", "mean")).lower()
        self.center_eps = float(svdd.get("eps", 0.1))

        backbone = str(svdd.get("backbone", ncfg.get("backbone", "conv"))).lower()
        dropout = float(ncfg.get("dropout", svdd.get("dropout", 0.0)))
        if backbone == "conv":
            ae = ConvWindowAE(
                channels=self.channels,
                window_size=self.window_size,
                hidden_channels=tuple(ncfg.get("conv_hidden", svdd.get("conv_hidden", [32, 64, 128]))),
                latent_channels=int(ncfg.get("conv_latent", svdd.get("conv_latent", 64))),
                kernel_size=int(ncfg.get("conv_kernel", svdd.get("conv_kernel", 7))),
                dropout=dropout,
            )
            self.encoder = ae.encoder
            self.feature_dim = int(ncfg.get("conv_latent", svdd.get("conv_latent", 64)))
            self.backbone_type = "conv"
        elif backbone == "mlp":
            ae = MLPWindowAE(
                input_dim=self.input_dim,
                latent_dim=int(ncfg.get("latent_dim", svdd.get("latent_dim", 256))),
                hidden_dim=int(ncfg.get("hidden_dim", svdd.get("hidden_dim", 1024))),
                dropout=dropout,
            )
            self.encoder = ae.encoder
            self.feature_dim = int(ncfg.get("latent_dim", svdd.get("latent_dim", 256)))
            self.backbone_type = "mlp"
        else:
            raise ValueError(f"DeepSVDD backbone must be 'conv' or 'mlp', got {backbone!r}")

        if self.score_aggregation not in {"mean", "max", "topk_mean"}:
            raise ValueError(
                "deepsvdd.score_aggregation must be one of: mean, max, topk_mean"
            )
        self.topk = int(svdd.get("topk", 5))
        self.register_buffer("center", torch.zeros(self.feature_dim))
        self.register_buffer("center_initialized", torch.tensor(False, dtype=torch.bool))

    def set_epoch(self, epoch: int) -> None:
        pass

    def _encode_flat(self, flat: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == "conv":
            x3 = flat.view(-1, self.channels, self.window_size)
            z = self.encoder(x3)
            return F.adaptive_avg_pool1d(z, 1).squeeze(-1)
        return self.encoder(flat)

    def encode_windows(self, x: torch.Tensor) -> torch.Tensor:
        win, _ = windowize(x, self.window_size, self.stride)
        b, m, c, w = win.shape
        flat = win.reshape(b * m, c * w)
        z = self._encode_flat(flat)
        return z.view(b, m, -1)

    @torch.no_grad()
    def before_training(self, loader, device: str) -> None:
        self.to(device)
        self.eval()
        sums = None
        count = 0
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            z = self.encode_windows(x).reshape(-1, self.feature_dim)
            batch_sum = z.sum(dim=0)
            sums = batch_sum if sums is None else sums + batch_sum
            count += int(z.shape[0])
        if sums is None or count == 0:
            raise RuntimeError("Cannot initialize DeepSVDD center from an empty loader")
        c = sums / float(count)
        eps = self.center_eps
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        self.center.copy_(c)
        self.center_initialized.fill_(True)

    def _window_distance(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_windows(x)
        if not bool(self.center_initialized.item()):
            with torch.no_grad():
                c = z.reshape(-1, z.shape[-1]).mean(dim=0)
                self.center.copy_(c)
                self.center_initialized.fill_(True)
        return ((z - self.center.view(1, 1, -1)) ** 2).sum(dim=-1)

    def _aggregate(self, dist: torch.Tensor) -> torch.Tensor:
        if self.score_aggregation == "mean":
            return dist.mean(dim=1)
        if self.score_aggregation == "max":
            return dist.max(dim=1).values
        k = max(1, min(self.topk, dist.shape[1]))
        return dist.topk(k, dim=1).values.mean(dim=1)

    def forward(self, batch: Dict, training: bool = True) -> DeepSVDDForward:
        dist = self._window_distance(batch["x"])
        sample_score = self._aggregate(dist)
        loss = dist.mean() if training else None
        zeros = torch.zeros_like(dist)
        logs = {
            "svdd_loss": float(dist.mean().detach().cpu()),
            "svdd_center_norm": float(self.center.norm().detach().cpu()),
        }
        return DeepSVDDForward(
            loss=loss,
            sample_score=sample_score,
            window_energy=dist,
            rec_window=dist,
            u1_window=zeros,
            u2_window=zeros,
            logs=logs,
        )

    def training_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(batch, training=True)
        assert out.loss is not None, "Loss is None during DeepSVDD training_step"
        return out.loss, out.logs

    @torch.no_grad()
    def score_batch(
        self,
        batch: Dict,
        score_mode: str | None = None,
        resa_ref: dict | None = None,
    ) -> Tuple[torch.Tensor, Dict]:
        out = self.forward(batch, training=False)
        return out.sample_score, {"score_mode": score_mode or "deepsvdd"}
