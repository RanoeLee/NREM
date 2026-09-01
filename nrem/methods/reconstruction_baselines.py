"""Conventional time-series reconstruction baselines for external comparison."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.windowing import windowize


@dataclass
class BaselineForward:
    loss: Optional[torch.Tensor]
    sample_score: torch.Tensor
    window_energy: torch.Tensor
    rec_window: torch.Tensor
    u1_window: torch.Tensor
    u2_window: torch.Tensor
    logs: Dict[str, float]


class _MLPAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, latent_dim), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


class _TCNAE(nn.Module):
    def __init__(self, channels: int, hidden: int = 64, latent: int = 64, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel, padding=pad), nn.GELU(),
            nn.Conv1d(hidden, latent, kernel, padding=pad), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent, hidden, kernel, padding=pad), nn.GELU(),
            nn.Conv1d(hidden, channels, kernel, padding=pad),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z.mean(dim=-1)


class _LSTMAE(nn.Module):
    def __init__(self, channels: int, hidden: int = 64, latent: int = 64, layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(channels, hidden, num_layers=layers, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [N, C, W]
        seq = x.transpose(1, 2)
        _, (h, _) = self.encoder(seq)
        z = self.to_latent(h[-1])
        dec_in = self.from_latent(z).unsqueeze(1).expand(-1, seq.shape[1], -1)
        y, _ = self.decoder(dec_in)
        return self.out(y).transpose(1, 2), z


class _TransformerAE(nn.Module):
    def __init__(self, channels: int, window_size: int, d_model: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window_size, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out = nn.Linear(d_model, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = x.transpose(1, 2)
        h = self.in_proj(seq) + self.pos[:, :seq.shape[1]]
        zseq = self.encoder(h)
        return self.out(zseq).transpose(1, 2), zseq.mean(dim=1)


class _OmniLite(nn.Module):
    def __init__(self, channels: int, hidden: int = 64, latent: int = 32):
        super().__init__()
        self.encoder = nn.GRU(channels, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec_in = nn.Linear(latent, hidden)
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = x.transpose(1, 2)
        _, h = self.encoder(seq)
        h = h[-1]
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 8.0)
        eps = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        dec_in = self.dec_in(z).unsqueeze(1).expand(-1, seq.shape[1], -1)
        y, _ = self.decoder(dec_in)
        return self.out(y).transpose(1, 2), z, mu, logvar


class ReconstructionBaselineMethod(nn.Module):
    """A family of conventional reconstruction baselines.

    Supported names are intentionally scoped to the implementation used here:
    `mlp_ae`, `lstm_ae`, `tcn_ae`, `usad`, `dagmm_style`, `omni_lite`, and
    `transformer_ae`.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        dcfg = cfg.get("data", {}) or {}
        mcfg = cfg.get("model", {}) or {}
        ncfg = cfg.get("nrem", {}) or {}
        bcfg = cfg.get("baseline", {}) or {}

        self.variant = str((cfg.get("method") or {}).get("name", "lstm_ae")).lower()
        self.channels = int(dcfg.get("channels", mcfg.get("channels", 6)))
        self.window_size = int(ncfg.get("window_size", bcfg.get("window_size", 1024)))
        self.stride = int(ncfg.get("stride", bcfg.get("stride", 512)))
        self.input_dim = self.channels * self.window_size
        self.score_aggregation = str(bcfg.get("score_aggregation", "mean")).lower()
        hidden = int(bcfg.get("hidden_dim", 64))
        latent = int(bcfg.get("latent_dim", 64))

        if self.variant == "mlp_ae":
            self.model = _MLPAE(self.input_dim, latent_dim=latent, hidden_dim=int(bcfg.get("mlp_hidden_dim", hidden)))
        elif self.variant == "lstm_ae":
            self.model = _LSTMAE(self.channels, hidden=hidden, latent=latent, layers=int(bcfg.get("layers", 1)))
        elif self.variant == "tcn_ae":
            self.model = _TCNAE(self.channels, hidden=hidden, latent=latent, kernel=int(bcfg.get("kernel", 5)))
        elif self.variant == "usad":
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, hidden), nn.GELU(),
                nn.Linear(hidden, latent), nn.GELU(),
            )
            self.decoder1 = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, self.input_dim))
            self.decoder2 = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, self.input_dim))
        elif self.variant == "dagmm_style":
            self.ae = _MLPAE(self.input_dim, latent_dim=latent, hidden_dim=int(bcfg.get("mlp_hidden_dim", 512)))
            self.estimation = nn.Sequential(
                nn.Linear(latent + 2, hidden), nn.Tanh(),
                nn.Linear(hidden, int(bcfg.get("components", 4))),
                nn.Softmax(dim=-1),
            )
        elif self.variant == "omni_lite":
            self.model = _OmniLite(self.channels, hidden=hidden, latent=latent)
            self.beta = float(bcfg.get("kl_weight", 0.001))
        elif self.variant == "transformer_ae":
            self.model = _TransformerAE(
                self.channels,
                self.window_size,
                d_model=hidden,
                heads=int(bcfg.get("heads", 4)),
                layers=int(bcfg.get("layers", 2)),
            )
        else:
            raise ValueError(f"Unknown reconstruction baseline: {self.variant!r}")

        if self.score_aggregation not in {"mean", "max"}:
            raise ValueError("baseline.score_aggregation must be 'mean' or 'max'")
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _aggregate(self, rec_w: torch.Tensor) -> torch.Tensor:
        if self.score_aggregation == "max":
            return rec_w.max(dim=1).values
        return rec_w.mean(dim=1)

    def _windows(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        win, _ = windowize(x, self.window_size, self.stride)
        b, m, c, w = win.shape
        return win.reshape(b * m, c, w), b, m

    def _forward_windows(self, xw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = xw.flatten(1)
        extra = flat.new_tensor(0.0)

        if self.variant == "mlp_ae":
            # The feed-forward autoencoder consumes flattened windows. Passing
            # [N, C, W] directly makes nn.Linear operate on W only and caused
            # the external MLP-AE jobs to fail with a 128-vs-768 dimension
            # mismatch for six-channel, 128-sample windows.
            recon, z = self.model(flat)
            return recon, z, extra

        if self.variant in {"lstm_ae", "tcn_ae", "transformer_ae"}:
            recon, z = self.model(xw)
            return recon.flatten(1), z, extra

        if self.variant == "usad":
            z = self.encoder(flat)
            w1 = self.decoder1(z)
            w2 = self.decoder2(z)
            w3 = self.decoder2(self.encoder(w1))
            rec1 = F.mse_loss(w1, flat, reduction="none").mean(dim=1)
            rec2 = F.mse_loss(w3, flat, reduction="none").mean(dim=1)
            alpha = 1.0 / float(self._epoch + 1)
            loss_extra = alpha * rec1.mean() + (1.0 - alpha) * rec2.mean()
            return 0.5 * (w1 + w3), z, loss_extra

        if self.variant == "dagmm_style":
            recon, z = self.ae(flat)
            rel_euclid = torch.norm(flat - recon, dim=1, keepdim=True) / torch.clamp(torch.norm(flat, dim=1, keepdim=True), min=1e-6)
            cos = F.cosine_similarity(flat, recon, dim=1).unsqueeze(1)
            gamma = self.estimation(torch.cat([z, rel_euclid, cos], dim=1))
            entropy = -(gamma * torch.log(gamma.clamp_min(1e-8))).sum(dim=1).mean()
            return recon, z, 0.001 * entropy

        recon, z, mu, logvar = self.model(xw)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return recon.flatten(1), z, self.beta * kl

    def forward(self, batch: Dict, training: bool = True) -> BaselineForward:
        xw, b, m = self._windows(batch["x"])
        flat = xw.flatten(1)
        recon, _z, extra = self._forward_windows(xw)
        rec = F.mse_loss(recon, flat, reduction="none").mean(dim=1).view(b, m)
        loss = rec.mean() + extra if training else None
        sample_score = self._aggregate(rec)
        zeros = torch.zeros_like(rec)
        logs = {"rec_loss": float(rec.mean().detach().cpu())}
        if training:
            logs["loss"] = float((rec.mean() + extra).detach().cpu())
        return BaselineForward(
            loss=loss,
            sample_score=sample_score,
            window_energy=rec,
            rec_window=rec,
            u1_window=zeros,
            u2_window=zeros,
            logs=logs,
        )

    def training_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(batch, training=True)
        assert out.loss is not None
        return out.loss, out.logs

    @torch.no_grad()
    def score_batch(
        self,
        batch: Dict,
        score_mode: str | None = None,
        resa_ref: dict | None = None,
    ) -> Tuple[torch.Tensor, Dict]:
        out = self.forward(batch, training=False)
        return out.sample_score, {"score_mode": score_mode or self.variant}
