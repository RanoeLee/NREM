"""Official-faithful external anomaly detection baselines.

These adapters keep method-specific objectives separate from the earlier
engineering reconstruction baselines. They are not drop-in vendored upstream
repositories; instead they preserve the core published training/scoring
mechanics while using the local Midea-Set loader and AUROC/AUPRC evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.windowing import windowize


@dataclass
class OfficialBaselineForward:
    loss: Optional[torch.Tensor]
    sample_score: torch.Tensor
    window_energy: torch.Tensor
    rec_window: torch.Tensor
    u1_window: torch.Tensor
    u2_window: torch.Tensor
    logs: Dict[str, float]


class _MLPAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, sigmoid_out: bool = False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, latent_dim)), nn.Tanh(),
            nn.Linear(max(hidden_dim // 2, latent_dim), latent_dim),
        )
        dec = [
            nn.Linear(latent_dim, max(hidden_dim // 2, latent_dim)), nn.Tanh(),
            nn.Linear(max(hidden_dim // 2, latent_dim), hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
        ]
        if sigmoid_out:
            dec.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


class _GRUVAE(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.GRU(channels, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec_in = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = x.transpose(1, 2)
        _, h = self.encoder(seq)
        h = h[-1]
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 8.0)
        eps = torch.randn_like(mu) if self.training else torch.zeros_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        dec = self.dec_in(z).unsqueeze(1).expand(-1, seq.shape[1], -1)
        y, _ = self.decoder(dec)
        return self.out(y).transpose(1, 2), mu, logvar


class _AssociationLayer(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.heads = heads
        self.d_head = d_model // heads
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.sigma = nn.Linear(d_model, heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        q = self.q(x).view(b, t, self.heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(b, t, self.heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(b, t, self.heads, self.d_head).transpose(1, 2)
        series = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head), dim=-1)
        sigma = F.softplus(self.sigma(x)).transpose(1, 2).unsqueeze(-1) + 1e-4
        idx = torch.arange(t, device=x.device, dtype=x.dtype)
        dist = (idx[None, :] - idx[:, None]).abs().view(1, 1, t, t)
        prior = torch.exp(-0.5 * (dist / sigma).pow(2))
        prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        y = torch.matmul(series, v).transpose(1, 2).contiguous().view(b, t, d)
        x = self.norm1(x + self.out(y))
        x = self.norm2(x + self.ffn(x))
        return x, series, prior


class _AssociationAE(nn.Module):
    def __init__(self, channels: int, window_size: int, d_model: int, heads: int, layers: int):
        super().__init__()
        self.in_proj = nn.Linear(channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window_size, d_model))
        self.layers = nn.ModuleList([_AssociationLayer(d_model, heads) for _ in range(layers)])
        self.out = nn.Linear(d_model, channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x.transpose(1, 2)) + self.pos[:, : x.shape[-1]]
        penalties = []
        for layer in self.layers:
            h, series, prior = layer(h)
            s = series.clamp_min(1e-8)
            p = prior.clamp_min(1e-8)
            penalties.append((s * (s.log() - p.log())).sum(dim=-1).mean(dim=(1, 2)))
        rec = self.out(h).transpose(1, 2)
        penalty = torch.stack(penalties, dim=0).mean(dim=0)
        return rec, penalty


class _TranADAE(nn.Module):
    def __init__(self, channels: int, window_size: int, d_model: int, heads: int, layers: int):
        super().__init__()
        self.in_proj = nn.Linear(channels * 2, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window_size, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.decoder1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, channels),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, channels),
        )

    def _encode(self, x: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        seq = torch.cat([x.transpose(1, 2), focus.transpose(1, 2)], dim=-1)
        return self.encoder(self.in_proj(seq) + self.pos[:, : seq.shape[1]])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        focus0 = torch.zeros_like(x)
        h1 = self._encode(x, focus0)
        out1 = self.decoder1(h1).transpose(1, 2)
        focus1 = (out1.detach() - x).pow(2)
        h2 = self._encode(x, focus1)
        out2 = self.decoder2(h2).transpose(1, 2)
        return out1, out2


class OfficialBaselineMethod(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        dcfg = cfg.get("data", {}) or {}
        mcfg = cfg.get("model", {}) or {}
        ncfg = cfg.get("nrem", {}) or {}
        bcfg = cfg.get("baseline", {}) or {}
        self.variant = str((cfg.get("method") or {}).get("name", "")).lower()
        self.channels = int(dcfg.get("channels", mcfg.get("channels", 6)))
        self.window_size = int(ncfg.get("window_size", bcfg.get("window_size", 128)))
        self.stride = int(ncfg.get("stride", bcfg.get("stride", self.window_size // 2)))
        self.input_dim = self.channels * self.window_size
        self.score_aggregation = str(bcfg.get("score_aggregation", "mean")).lower()
        hidden = int(bcfg.get("hidden_dim", 256))
        latent = int(bcfg.get("latent_dim", 32))
        self._epoch = 0
        if self.score_aggregation not in {"mean", "max"}:
            raise ValueError("baseline.score_aggregation must be 'mean' or 'max'")

        if self.variant == "dagmm_official":
            self.ae = _MLPAE(self.input_dim, latent, hidden, sigmoid_out=False)
            self.estimation = nn.Sequential(
                nn.Linear(latent + 2, hidden), nn.Tanh(),
                nn.Dropout(float(bcfg.get("dropout", 0.1))),
                nn.Linear(hidden, int(bcfg.get("components", 4))),
                nn.Softmax(dim=-1),
            )
            self.lambda_energy = float(bcfg.get("lambda_energy", 0.1))
            self.lambda_cov = float(bcfg.get("lambda_cov", 0.005))
        elif self.variant == "omnianomaly_official":
            self.model = _GRUVAE(self.channels, hidden, latent)
            self.beta = float(bcfg.get("kl_weight", 0.001))
        elif self.variant == "anomaly_transformer_official":
            self.model = _AssociationAE(
                self.channels,
                self.window_size,
                d_model=hidden,
                heads=int(bcfg.get("heads", 4)),
                layers=int(bcfg.get("layers", 2)),
            )
            self.lambda_assoc = float(bcfg.get("lambda_assoc", 0.1))
        elif self.variant == "tranad_official":
            self.model = _TranADAE(
                self.channels,
                self.window_size,
                d_model=hidden,
                heads=int(bcfg.get("heads", 4)),
                layers=int(bcfg.get("layers", 1)),
            )
            self.score_alpha = float(bcfg.get("score_alpha", 0.5))
            self.score_beta = float(bcfg.get("score_beta", 0.5))
        else:
            raise ValueError(f"Unknown official baseline: {self.variant!r}")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _windows(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        win, _ = windowize(x, self.window_size, self.stride)
        b, m, c, w = win.shape
        return win.reshape(b * m, c, w), b, m, c * w

    def _aggregate(self, rec_w: torch.Tensor) -> torch.Tensor:
        if self.score_aggregation == "max":
            return rec_w.max(dim=1).values
        return rec_w.mean(dim=1)

    def _dagmm_energy(self, z: torch.Tensor, gamma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = z.shape[0]
        phi = gamma.mean(dim=0)
        mu = gamma.t().matmul(z) / gamma.sum(dim=0).unsqueeze(1).clamp_min(1e-8)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        cov = torch.einsum("nk,nkd,nke->kde", gamma, diff, diff) / gamma.sum(dim=0).view(-1, 1, 1).clamp_min(1e-8)
        eye = torch.eye(cov.shape[-1], device=z.device).unsqueeze(0)
        cov = cov + 1e-4 * eye
        inv = torch.linalg.pinv(cov)
        exponent = -0.5 * torch.einsum("nkd,kde,nke->nk", diff, inv, diff)
        det = torch.linalg.det(cov).abs().clamp_min(1e-12)
        coef = phi / torch.sqrt(((2.0 * math.pi) ** z.shape[1]) * det)
        prob = (coef.unsqueeze(0) * torch.exp(exponent)).sum(dim=1).clamp_min(1e-12)
        energy = -torch.log(prob)
        cov_diag_penalty = (1.0 / cov.diagonal(dim1=-2, dim2=-1).clamp_min(1e-8)).sum()
        return energy, cov_diag_penalty / float(max(n, 1))

    def forward(self, batch: Dict, training: bool = True) -> OfficialBaselineForward:
        xw, b, m, _ = self._windows(batch["x"])
        flat = xw.flatten(1)
        extra = flat.new_tensor(0.0)

        if self.variant == "dagmm_official":
            recon, z = self.ae(flat)
            rec_vec = F.mse_loss(recon, flat, reduction="none")
            rec = rec_vec.mean(dim=1)
            rel = torch.norm(flat - recon, dim=1, keepdim=True) / torch.norm(flat, dim=1, keepdim=True).clamp_min(1e-8)
            cos = F.cosine_similarity(flat, recon, dim=1).unsqueeze(1)
            z_full = torch.cat([z, rel, cos], dim=1)
            gamma = self.estimation(z_full)
            energy, cov_pen = self._dagmm_energy(z_full, gamma)
            window_score = (rec + self.lambda_energy * energy).view(b, m)
            if training:
                extra = self.lambda_energy * energy.mean() + self.lambda_cov * cov_pen
            loss = rec.mean() + extra if training else None
        elif self.variant == "omnianomaly_official":
            recon, mu, logvar = self.model(xw)
            rec = F.mse_loss(recon, xw, reduction="none").mean(dim=(1, 2))
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            window_score = (rec + self.beta * (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean(dim=1))).view(b, m)
            loss = rec.mean() + self.beta * kl if training else None
        elif self.variant == "anomaly_transformer_official":
            recon, assoc = self.model(xw)
            rec = F.mse_loss(recon, xw, reduction="none").mean(dim=(1, 2))
            window_score = (rec + self.lambda_assoc * assoc).view(b, m)
            loss = rec.mean() + self.lambda_assoc * assoc.mean() if training else None
        else:
            recon1, recon2 = self.model(xw)
            rec1 = F.mse_loss(recon1, xw, reduction="none").mean(dim=(1, 2))
            rec2 = F.mse_loss(recon2, xw, reduction="none").mean(dim=(1, 2))
            alpha = 1.0 / float(self._epoch + 1)
            window_score = (self.score_alpha * rec1 + self.score_beta * rec2).view(b, m)
            loss = alpha * rec1.mean() + (1.0 - alpha) * rec2.mean() if training else None

        zeros = torch.zeros_like(window_score)
        sample_score = self._aggregate(window_score)
        logs = {
            "loss": float(loss.detach().cpu()) if loss is not None else 0.0,
            "rec_loss": float(window_score.mean().detach().cpu()),
        }
        return OfficialBaselineForward(
            loss=loss,
            sample_score=sample_score,
            window_energy=window_score,
            rec_window=window_score,
            u1_window=zeros,
            u2_window=zeros,
            logs=logs,
        )

    def training_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(batch, training=True)
        assert out.loss is not None
        return out.loss, out.logs

    @torch.no_grad()
    def score_batch(self, batch: Dict, score_mode: str | None = None, resa_ref: dict | None = None):
        del score_mode, resa_ref
        out = self.forward(batch, training=False)
        return out.sample_score, {"score_mode": self.variant}
