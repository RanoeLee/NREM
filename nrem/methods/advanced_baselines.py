"""Published-core adapters for recent multivariate time-series baselines.

These implementations share the repository's data loader and product-level
evaluator. They preserve the mechanism that defines each compared method, but
are deliberately named adapters because they are not vendored upstream code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.windowing import windowize


@dataclass
class AdvancedBaselineForward:
    loss: Optional[torch.Tensor]
    sample_score: torch.Tensor
    window_energy: torch.Tensor
    rec_window: torch.Tensor
    u1_window: torch.Tensor
    u2_window: torch.Tensor
    logs: Dict[str, float]


class _TemporalEncoder(nn.Module):
    def __init__(self, channels: int, hidden: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, latent, 3, padding=1), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).mean(dim=-1)


class _GDNCore(nn.Module):
    """Learned top-k sensor graph with graph-conditioned next-step prediction."""

    def __init__(self, channels: int, hidden: int, embedding: int, topk: int):
        super().__init__()
        self.channels = channels
        self.topk = min(max(int(topk), 1), channels)
        self.node_embeddings = nn.Parameter(torch.randn(channels, embedding) * 0.1)
        self.temporal = nn.Sequential(
            nn.Conv1d(1, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.predictor = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def adjacency(self) -> torch.Tensor:
        embeddings = F.normalize(self.node_embeddings, dim=-1)
        similarity = embeddings @ embeddings.t()
        values, indices = torch.topk(similarity, k=self.topk, dim=-1)
        sparse = similarity.new_full(similarity.shape, float("-inf"))
        sparse.scatter_(1, indices, values)
        return torch.softmax(sparse, dim=-1)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, c, t = history.shape
        local = self.temporal(history.reshape(n * c, 1, t)).mean(dim=-1).view(n, c, -1)
        graph = torch.einsum("ij,njh->nih", self.adjacency(), local)
        pred = self.predictor(torch.cat([local, graph], dim=-1)).squeeze(-1)
        return pred, self.adjacency()


class _MTGFlowCore(nn.Module):
    """Dynamic sensor graph followed by a conditional affine normalizing flow."""

    def __init__(self, channels: int, hidden: int, latent: int):
        super().__init__()
        self.channels = channels
        self.entity = nn.Embedding(channels, latent)
        self.temporal = nn.Sequential(
            nn.Conv1d(1, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, latent, 3, padding=1), nn.GELU(),
        )
        split = latent // 2
        if split < 1 or latent - split < 1:
            raise ValueError("MTGFlow latent dimension must be at least two")
        self.split = split
        self.coupling_12 = nn.Sequential(
            nn.Linear(split + latent, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * (latent - split)),
        )
        self.coupling_21 = nn.Sequential(
            nn.Linear((latent - split) + latent, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * split),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, c, t = x.shape
        h = self.temporal(x.reshape(n * c, 1, t)).mean(dim=-1).view(n, c, -1)
        entities = self.entity.weight.unsqueeze(0).expand(n, -1, -1)
        query = F.normalize(h + entities, dim=-1)
        adjacency = torch.softmax(torch.matmul(query, query.transpose(1, 2)), dim=-1)
        graph_context = torch.matmul(adjacency, h)
        condition = graph_context + entities
        first, second = h[..., :self.split], h[..., self.split:]
        shift_2, log_scale_2 = self.coupling_12(torch.cat([first, condition], dim=-1)).chunk(2, dim=-1)
        log_scale_2 = 2.0 * torch.tanh(log_scale_2 / 2.0)
        z_second = second * torch.exp(log_scale_2) + shift_2
        shift_1, log_scale_1 = self.coupling_21(
            torch.cat([z_second, condition], dim=-1)
        ).chunk(2, dim=-1)
        log_scale_1 = 2.0 * torch.tanh(log_scale_1 / 2.0)
        z_first = first * torch.exp(log_scale_1) + shift_1
        z = torch.cat([z_first, z_second], dim=-1)
        log_det = log_scale_1.sum(dim=-1) + log_scale_2.sum(dim=-1)
        nll = 0.5 * z.square().sum(dim=-1) - log_det
        return nll, adjacency


class _HybridNormalizer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.mix_logits = nn.Parameter(torch.zeros(channels))
        self.z_gain = nn.Parameter(torch.ones(channels))
        self.z_bias = nn.Parameter(torch.zeros(channels))
        self.mm_gain = nn.Parameter(torch.ones(channels))
        self.mm_bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True).clamp_min(1e-5)
        zscore = (x - mean) / std
        minimum, maximum = x.amin(dim=-1, keepdim=True), x.amax(dim=-1, keepdim=True)
        minmax = 2.0 * (x - minimum) / (maximum - minimum).clamp_min(1e-5) - 1.0
        zscore = zscore * self.z_gain[None, :, None] + self.z_bias[None, :, None]
        minmax = minmax * self.mm_gain[None, :, None] + self.mm_bias[None, :, None]
        mix = torch.sigmoid(self.mix_logits)[None, :, None]
        return mix * zscore + (1.0 - mix) * minmax


class _FilterAECore(nn.Module):
    """Learnable hybrid normalization, filtering, and clean-signal reconstruction."""

    def __init__(self, channels: int, hidden: int, latent: int):
        super().__init__()
        self.normalizer = _HybridNormalizer(channels)
        self.filter = nn.Sequential(
            nn.Conv1d(channels, channels, 9, padding=4, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1),
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, latent, 3, padding=1), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, channels, 5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.normalizer(x)
        clean = self.filter(normalized)
        reconstruction = self.decoder(self.encoder(clean))
        residual = normalized - clean
        return normalized, clean, reconstruction + 0.0 * residual.mean()


class _CARLACore(nn.Module):
    """Contrastive representation learning with deliberately injected anomalies."""

    def __init__(self, channels: int, hidden: int, latent: int):
        super().__init__()
        self.encoder = _TemporalEncoder(channels, hidden, latent)
        self.classifier = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, 1))

    @staticmethod
    def positive_view(x: torch.Tensor) -> torch.Tensor:
        return torch.roll(x, shifts=1, dims=-1) + 0.01 * torch.randn_like(x)

    @staticmethod
    def injected_anomaly(x: torch.Tensor) -> torch.Tensor:
        n, _c, t = x.shape
        ramp = torch.linspace(0.0, 1.0, t, device=x.device, dtype=x.dtype)[None, None, :]
        scale = x.std(dim=-1, keepdim=True).clamp_min(1e-4)
        negative = x + 2.5 * scale * ramp
        start = max(t // 3, 1)
        width = max(t // 12, 1)
        negative[:, :, start:start + width] += 4.0 * scale
        return negative

    def pretext_loss(self, x: torch.Tensor) -> torch.Tensor:
        positive = self.positive_view(x)
        negative = self.injected_anomaly(x)
        anchor_z = self.encoder(x)
        positive_z = self.encoder(positive)
        negative_z = self.encoder(negative)
        return F.triplet_margin_loss(anchor_z, positive_z, negative_z, margin=1.0)

    def representations(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def logits(self, representation: torch.Tensor) -> torch.Tensor:
        return self.classifier(representation).squeeze(-1)


class AdvancedBaselineMethod(nn.Module):
    SUPPORTED = {"gdn", "mtgflow", "normfaae", "carla"}
    requires_reference = False

    def __init__(self, cfg: dict):
        super().__init__()
        data = cfg.get("data", {}) or {}
        model = cfg.get("model", {}) or {}
        nrem = cfg.get("nrem", {}) or {}
        baseline = cfg.get("baseline", {}) or {}
        self.variant = str((cfg.get("method") or {}).get("name", "")).lower()
        if self.variant not in self.SUPPORTED:
            raise ValueError(f"Unknown advanced baseline: {self.variant!r}")
        self.uses_custom_training = self.variant in {"carla", "normfaae"}
        self.channels = int(data.get("channels", model.get("channels", 6)))
        self.window_size = int(nrem.get("window_size", 128))
        self.stride = int(nrem.get("stride", 64))
        self.score_aggregation = str(baseline.get("score_aggregation", "mean"))
        hidden = int(baseline.get("hidden_dim", 64))
        latent = int(baseline.get("latent_dim", 32))
        if self.variant == "gdn":
            self.model = _GDNCore(self.channels, hidden, latent, int(baseline.get("topk", 3)))
        elif self.variant == "mtgflow":
            self.model = _MTGFlowCore(self.channels, hidden, latent)
        elif self.variant == "normfaae":
            self.model = _FilterAECore(self.channels, hidden, latent)
            self.filter_weight = float(baseline.get("filter_weight", 0.05))
            self.uses_custom_training = True
        else:
            self.model = _CARLACore(self.channels, hidden, latent)
            self.triplet_weight = float(baseline.get("triplet_weight", 1.0))
            self.pretext_epochs = int(baseline.get("pretext_epochs", 15))
            self.neighbor_fraction = float(baseline.get("neighbor_fraction", 0.1))

    def _windows(self, x: torch.Tensor):
        windows, _ = windowize(x, self.window_size, self.stride)
        b, m, c, w = windows.shape
        return windows.reshape(b * m, c, w), b, m

    def _aggregate(self, score: torch.Tensor) -> torch.Tensor:
        return score.max(dim=1).values if self.score_aggregation == "max" else score.mean(dim=1)

    def forward(self, batch: Dict, training: bool = True) -> AdvancedBaselineForward:
        xw, b, m = self._windows(batch["x"])
        if self.variant == "gdn":
            prediction, _adjacency = self.model(xw[:, :, :-1])
            energy = (prediction - xw[:, :, -1]).square().mean(dim=-1)
            loss = energy.mean() if training else None
        elif self.variant == "mtgflow":
            entity_nll, _adjacency = self.model(xw)
            energy = entity_nll.mean(dim=-1)
            loss = energy.mean() if training else None
        elif self.variant == "normfaae":
            normalized, clean, reconstruction = self.model(xw)
            rec = (reconstruction - clean.detach()).square().mean(dim=(1, 2))
            separation = (normalized - clean).abs().mean(dim=(1, 2))
            energy = (normalized - reconstruction).square().mean(dim=(1, 2))
            loss = rec.mean() + self.filter_weight * separation.mean() if training else None
        else:
            representation = self.model.representations(xw)
            energy = torch.sigmoid(self.model.logits(representation))
            loss = self.model.pretext_loss(xw) if training else None

        window_energy = energy.view(b, m)
        zeros = torch.zeros_like(window_energy)
        logs = {
            "loss": float(loss.detach().cpu()) if loss is not None else 0.0,
            "window_score": float(window_energy.mean().detach().cpu()),
        }
        return AdvancedBaselineForward(
            loss=loss,
            sample_score=self._aggregate(window_energy),
            window_energy=window_energy,
            rec_window=window_energy,
            u1_window=zeros,
            u2_window=zeros,
            logs=logs,
        )

    def training_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        output = self.forward(batch, training=True)
        assert output.loss is not None
        return output.loss, output.logs

    def configure_optimizers(self, cfg: dict):
        if self.variant not in {"carla", "normfaae"}:
            return None
        train = cfg.get("train", {}) or {}
        lr = float(train.get("lr", 5e-4))
        wd = float(train.get("weight_decay", 1e-5))
        if self.variant == "carla":
            return {
                "pretext": torch.optim.AdamW(self.model.encoder.parameters(), lr=lr, weight_decay=wd),
                "classifier": torch.optim.AdamW(self.model.classifier.parameters(), lr=lr, weight_decay=wd),
            }
        ae_parameters = list(self.model.normalizer.parameters())
        ae_parameters += list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        return {
            "autoencoder": torch.optim.AdamW(ae_parameters, lr=lr, weight_decay=wd),
            "filter": torch.optim.AdamW(self.model.filter.parameters(), lr=lr, weight_decay=wd),
        }

    def training_epoch(self, loader, cfg: dict, device: str, epoch: int, optimizers=None):
        """CARLA's pretext stage followed by nearest/furthest pseudo classification."""
        if self.variant not in {"carla", "normfaae"}:
            raise AttributeError("training_epoch is only defined for custom two-stage methods")
        if optimizers is None:
            raise RuntimeError("Custom training requires method-specific optimizers")
        self.train()
        losses = []
        if self.variant == "normfaae":
            for batch in loader:
                windows, _b, _m = self._windows(batch["x"].to(device))
                # Step A: fit normalization and AE to the filtered signal while
                # the filter is held fixed.
                for parameter in self.model.filter.parameters():
                    parameter.requires_grad_(False)
                normalized, clean, reconstruction = self.model(windows)
                ae_loss = F.mse_loss(reconstruction, clean.detach())
                optimizers["autoencoder"].zero_grad(set_to_none=True)
                ae_loss.backward()
                optimizers["autoencoder"].step()
                for parameter in self.model.filter.parameters():
                    parameter.requires_grad_(True)

                # Step B: fit the filter to retain the AE-explainable component
                # and penalise unnecessary distortion of the normalized input.
                with torch.no_grad():
                    normalized = self.model.normalizer(windows)
                clean = self.model.filter(normalized.detach())
                with torch.no_grad():
                    reconstruction = self.model.decoder(self.model.encoder(clean))
                filter_loss = F.mse_loss(clean, reconstruction.detach())
                filter_loss += self.filter_weight * F.l1_loss(clean, normalized.detach())
                optimizers["filter"].zero_grad(set_to_none=True)
                filter_loss.backward()
                optimizers["filter"].step()
                losses.append(float((ae_loss + filter_loss).detach().cpu()))
            return {"loss": sum(losses) / max(len(losses), 1), "stage": "alternating_filter_ae"}

        if epoch < self.pretext_epochs:
            for batch in loader:
                x = batch["x"].to(device)
                windows, _b, _m = self._windows(x)
                loss = self.model.pretext_loss(windows)
                optimizers["pretext"].zero_grad(set_to_none=True)
                loss.backward()
                optimizers["pretext"].step()
                losses.append(float(loss.detach().cpu()))
            return {"loss": sum(losses) / max(len(losses), 1), "stage": "pretext"}

        # Represent the current training population without labels. Samples
        # nearest to the robust representation centre form the normal anchor
        # set; equally many furthest samples form the contrasting set.
        representations = []
        with torch.no_grad():
            for batch in loader:
                windows, _b, _m = self._windows(batch["x"].to(device))
                representations.append(self.model.representations(windows))
        representation = torch.cat(representations, dim=0)
        centre = representation.median(dim=0).values
        distance = torch.norm(representation - centre, dim=-1)
        count = max(1, int(round(distance.numel() * self.neighbor_fraction)))
        count = min(count, max(distance.numel() // 2, 1))
        order = torch.argsort(distance)
        selected = torch.cat([order[:count], order[-count:]])
        targets = torch.cat([
            torch.zeros(count, device=device), torch.ones(count, device=device),
        ])
        logits = self.model.logits(representation[selected].detach())
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        optimizers["classifier"].zero_grad(set_to_none=True)
        loss.backward()
        optimizers["classifier"].step()
        return {"loss": float(loss.detach().cpu()), "stage": "pseudo_classification"}

    @torch.no_grad()
    def score_batch(self, batch: Dict, score_mode=None, resa_ref=None):
        del score_mode, resa_ref
        output = self.forward(batch, training=False)
        return output.sample_score, {"score_mode": self.variant}
