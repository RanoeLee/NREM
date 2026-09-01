"""Sample Residual Evidence (SRE) — Eqs (3)-(5) in the paper.

Given the AE residual r_m for window m, this module:
  1. Applies two complementary temporal masks to r_m              [Eq 3]
  2. Encodes each masked residual with a shared conv encoder
  3. Computes the BYOL-style cross-view prediction energy u1_m    [Eq 4]
  4. Returns the consensus descriptor e_m = 0.5*(ea + eb)

For the M2 ablation (view_mode='symmetric'), step 3 uses the direct
symmetric distance instead of the BYOL predictor:
    u1_m = ρ_ν(ea − eb)
"""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn

from ..modules.student_t import StudentTEnergy
from ..modules.masking import complementary_element_mask


# ── Residual encoder ─────────────────────────────────────────────────────────

class ConvResidualEncoder(nn.Module):
    """Map a masked residual window [N, C, W] → descriptor [N, d_model].

    Applies a small stride-2 conv stack then global-average-pools over the
    time axis. Output is L2-normalised via LayerNorm (matches prior work on
    self-supervised descriptor learning).
    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        d_model: int = 128,
        hidden_channels: Tuple[int, ...] = (64, 128),
        kernel_size: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        from .backbone import _gn   # reuse helper
        self.channels = int(channels)
        self.window_size = int(window_size)
        pad = kernel_size // 2
        layers: list[nn.Module] = []
        in_c = channels
        for h in hidden_channels:
            layers += [nn.Conv1d(in_c, h, kernel_size, stride=2, padding=pad),
                       _gn(h), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_c = h
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_channels[-1], d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """r: [N, C, W]  →  descriptor [N, d_model]"""
        h = self.net(r).mean(dim=-1)   # global average pool over time
        return self.norm(self.proj(h))


# ── Within-sample module ─────────────────────────────────────────────────────

class SampleResidualEvidence(nn.Module):
    """Dual-view residual consistency — within-sample anomaly evidence.

    Parameters
    ----------
    channels, window_size : int
        Signal dimensions.
    d_model : int
        Descriptor dimension d.
    residual_enc_hidden : tuple
        Hidden channel widths for ConvResidualEncoder.
    predictor_hidden_dim : int
        Hidden dim of the 2-layer MLP predictor π.
    mask_ratio : float
        Fraction of residual elements assigned to the second complementary view
        during training (κ₁ = 0.30 in the reported model); the first view
        receives 1-κ₁. Evaluation uses a deterministic 50/50 alternating mask.
    nu : float
        Student-t degrees of freedom for ρ_ν.
    view_mode : str
        'byol'      — asymmetric predictor + stop-gradient   [Eq 4, default]
        'symmetric' — direct distance ρ_ν(ea−eb)              [M2 ablation]
    kernel_size : int
        Conv kernel for the residual encoder.
    dropout : float
    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        d_model: int = 128,
        residual_enc_hidden: Tuple[int, ...] = (64, 128),
        predictor_hidden_dim: int = 256,
        mask_ratio: float = 0.30,
        nu: float = 4.0,
        view_mode: str = "byol",
        kernel_size: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        self.mask_ratio = float(mask_ratio)
        self.view_mode = str(view_mode).lower()
        if self.view_mode not in {"byol", "symmetric"}:
            raise ValueError(f"view_mode must be 'byol' or 'symmetric', got {view_mode}")

        self.residual_encoder = ConvResidualEncoder(
            channels=channels,
            window_size=window_size,
            d_model=d_model,
            hidden_channels=residual_enc_hidden,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        if self.view_mode == "byol":
            # π: R^d → R^d   (2-layer MLP, wider hidden)
            self.predictor = nn.Sequential(
                nn.Linear(d_model, predictor_hidden_dim),
                nn.GELU(),
                nn.Linear(predictor_hidden_dim, d_model),
            )
        else:
            self.predictor = None   # not used in symmetric mode

        self.energy = StudentTEnergy(nu=nu)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        r_sg: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute within-sample energy for a batch of detached residuals.

        Parameters
        ----------
        r_sg : Tensor [B*M, C, W]
            Stop-gradient residuals from the backbone.
        training : bool
            If True, masks are sampled randomly; if False, a fixed
            alternating mask is used (deterministic at inference).

        Returns
        -------
        u1      : Tensor [B*M]   per-window within-sample energy
        ea, eb  : Tensor [B*M, d]  per-view descriptors
        e       : Tensor [B*M, d]  consensus descriptor (ea+eb)/2
        """
        # Eq 3: complementary masks on the residual
        ma, mb = complementary_element_mask(
            r_sg, mask_ratio=self.mask_ratio, training=training
        )
        ea = self.residual_encoder(ma * r_sg)   # [B*M, d]
        eb = self.residual_encoder(mb * r_sg)

        # Eq 4: view consistency energy
        if self.view_mode == "byol":
            pa = self.predictor(ea)
            pb = self.predictor(eb)
            u1 = 0.5 * (
                self.energy(pa - eb.detach()) +
                self.energy(pb - ea.detach())
            )
        else:   # symmetric — M2 ablation
            u1 = self.energy(ea - eb)

        e = 0.5 * (ea + eb)   # consensus descriptor
        return u1, ea, eb, e


# Backward-compatible import name for checkpoints and third-party scripts.
IntraSampleModule = SampleResidualEvidence
