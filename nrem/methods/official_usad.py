"""USAD faithful baseline adapted to the NREM engine.

This implementation keeps the central USAD training mechanics separate from
the in-repository reconstruction baselines:

- shared MLP encoder with two decoders;
- epoch-dependent USAD losses;
- separate optimizer updates for AE1 and AE2;
- reconstruction score from decoder1 and decoder2(encoder(decoder1(...))).

The surrounding data loading, checkpointing, and AUROC/AUPRC evaluation are
provided by the local NREM runner so that Midea-Set comparisons use the same
train/eval split and metric code.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..modules.windowing import windowize


class _Encoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        h1 = max(latent_dim * 4, in_dim // 2)
        h2 = max(latent_dim * 2, in_dim // 4)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, out_dim: int, latent_dim: int):
        super().__init__()
        h2 = max(latent_dim * 2, out_dim // 4)
        h1 = max(latent_dim * 4, out_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class OfficialUSADMethod(nn.Module):
    """Official-style USAD baseline for multivariate time-series windows."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        dcfg = cfg.get("data", {}) or {}
        mcfg = cfg.get("model", {}) or {}
        ncfg = cfg.get("nrem", {}) or {}
        bcfg = cfg.get("baseline", {}) or {}

        self.channels = int(dcfg.get("channels", mcfg.get("channels", 6)))
        self.window_size = int(ncfg.get("window_size", bcfg.get("window_size", 1024)))
        self.stride = int(ncfg.get("stride", bcfg.get("stride", 512)))
        self.input_dim = self.channels * self.window_size
        self.latent_dim = int(bcfg.get("latent_dim", max(1, self.input_dim // 10)))
        self.score_alpha = float(bcfg.get("score_alpha", 0.5))
        self.score_beta = float(bcfg.get("score_beta", 0.5))
        self.score_aggregation = str(bcfg.get("score_aggregation", "mean")).lower()
        if self.score_aggregation not in {"mean", "max"}:
            raise ValueError("baseline.score_aggregation must be 'mean' or 'max'")

        self.encoder = _Encoder(self.input_dim, self.latent_dim)
        self.decoder1 = _Decoder(self.input_dim, self.latent_dim)
        self.decoder2 = _Decoder(self.input_dim, self.latent_dim)

    def _windows(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        win, _ = windowize(x, self.window_size, self.stride)
        b, m, c, w = win.shape
        return win.reshape(b * m, c * w), b, m

    def _reconstruct(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(flat)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        return w1, w2, w3

    def _aggregate(self, rec_w: torch.Tensor) -> torch.Tensor:
        if self.score_aggregation == "max":
            return rec_w.max(dim=1).values
        return rec_w.mean(dim=1)

    def configure_optimizers(self, cfg: dict):
        tcfg = cfg.get("train", {}) or {}
        lr = float(tcfg.get("lr", 1e-3))
        wd = float(tcfg.get("weight_decay", 0.0))
        opt1 = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder1.parameters()),
            lr=lr,
            weight_decay=wd,
        )
        opt2 = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder2.parameters()),
            lr=lr,
            weight_decay=wd,
        )
        return {"ae1": opt1, "ae2": opt2}

    def training_epoch(self, loader, cfg: dict, device: str, epoch: int, optimizers=None) -> Dict[str, float]:
        opts = optimizers if optimizers is not None else self.configure_optimizers(cfg)
        grad_clip = float((cfg.get("train") or {}).get("grad_clip", 0.0))
        n = float(epoch + 1)
        self.train()
        meters: Dict[str, list[float]] = {"loss": [], "loss_ae1": [], "loss_ae2": []}
        pbar = tqdm(loader, desc=f"official_usad epoch {epoch+1}", leave=False)

        for batch in pbar:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            flat, _, _ = self._windows(batch["x"])

            w1, _w2, w3 = self._reconstruct(flat)
            loss1 = (1.0 / n) * F.mse_loss(w1, flat) + (1.0 - 1.0 / n) * F.mse_loss(w3, flat)
            opts["ae1"].zero_grad(set_to_none=True)
            loss1.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder1.parameters()),
                    grad_clip,
                )
            opts["ae1"].step()

            _w1, w2, w3 = self._reconstruct(flat)
            loss2 = (1.0 / n) * F.mse_loss(w2, flat) - (1.0 - 1.0 / n) * F.mse_loss(w3, flat)
            opts["ae2"].zero_grad(set_to_none=True)
            loss2.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder2.parameters()),
                    grad_clip,
                )
            opts["ae2"].step()

            total = loss1.detach() + loss2.detach()
            meters["loss"].append(float(total.cpu()))
            meters["loss_ae1"].append(float(loss1.detach().cpu()))
            meters["loss_ae2"].append(float(loss2.detach().cpu()))
            pbar.set_postfix({k: f"{sum(v) / max(len(v), 1):.4f}" for k, v in meters.items()})

        return {k: float(sum(v) / max(len(v), 1)) for k, v in meters.items()}

    def training_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        flat, _, _ = self._windows(batch["x"])
        w1, _w2, w3 = self._reconstruct(flat)
        loss = self.score_alpha * F.mse_loss(w1, flat) + self.score_beta * F.mse_loss(w3, flat)
        return loss, {"loss": float(loss.detach().cpu())}

    def forward(self, batch: Dict, training: bool = True):
        flat, b, m = self._windows(batch["x"])
        w1, _w2, w3 = self._reconstruct(flat)
        rec = (
            self.score_alpha * F.mse_loss(w1, flat, reduction="none").mean(dim=1)
            + self.score_beta * F.mse_loss(w3, flat, reduction="none").mean(dim=1)
        ).view(b, m)
        zeros = torch.zeros_like(rec)
        return type("OfficialUSADForward", (), {
            "loss": rec.mean() if training else None,
            "sample_score": self._aggregate(rec),
            "window_energy": rec,
            "rec_window": rec,
            "u1_window": zeros,
            "u2_window": zeros,
            "logs": {"loss": float(rec.mean().detach().cpu()), "rec_loss": float(rec.mean().detach().cpu())},
        })()

    @torch.no_grad()
    def score_batch(
        self,
        batch: Dict,
        score_mode: str | None = None,
        resa_ref: dict | None = None,
    ) -> Tuple[torch.Tensor, Dict]:
        del score_mode, resa_ref
        out = self.forward(batch, training=False)
        return out.sample_score, {"window_energy": out.window_energy}
