"""Contextual Trajectory Modeling (CTM) — Eqs (6)-(8) in the paper.

Given the descriptor trajectory E = (e_1, …, e_M) of a single run, this
module predicts each descriptor from the trajectory context of its neighbours
using a small stack of dilated 1-D convolutions over the M axis.

Training  : a random subset M_mask ⊂ {1,…,M} is zeroed out; the model
            predicts (μ_m, log σ_m) at those positions and the Student-t
            surprise  u2_m = ρ_ν((e_m − μ_m)/σ_m) is the training loss.

Inference : leave-group-out with K=4 groups so every position gets a
            prediction from K independent context sets.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..modules.student_t import StudentTEnergy
from ..modules.masking import make_window_mask


class ContextualTrajectoryModel(nn.Module):
    """Masked-trajectory dynamics for within-condition anomaly evidence.

    Parameters
    ----------
    d_model : int
        Descriptor dimension d.
    hidden : int
        Channel width of the dilated conv stack.
    layers : int
        Number of dilated conv layers (dilation doubles each layer).
    dropout : float
    nu : float
        Student-t degrees of freedom.
    dyn_mask_ratio : float
        Fraction of trajectory positions masked during training (κ₂ = 0.15).
    eval_groups : int
        Leave-group-out groups at inference (K = 4).
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden: int = 256,
        layers: int = 4,
        dropout: float = 0.0,
        nu: float = 4.0,
        dyn_mask_ratio: float = 0.15,
        eval_groups: int = 4,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.dyn_mask_ratio = float(dyn_mask_ratio)
        self.eval_groups = int(eval_groups)

        # Dilated 1-D conv over the M (window-index) axis
        # Input / output: [B, d_model, M]
        blocks: list[nn.Module] = []
        for i in range(int(layers)):
            dil = 2 ** i
            blocks += [
                nn.Conv1d(d_model, d_model, kernel_size=3,
                          padding=dil, dilation=dil),
                nn.GELU(),
            ]
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*blocks)
        self.to_mu        = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.to_log_scale = nn.Conv1d(d_model, d_model, kernel_size=1)

        self.energy = StudentTEnergy(nu=nu)

    # ── Core prediction ──────────────────────────────────────────────────────

    def _predict(
        self, E: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict (μ, log_scale) at ALL positions given masked E.

        Parameters
        ----------
        E    : [B, M, d]
        mask : [B, M]  bool — True positions are zeroed out before feeding in.

        Returns
        -------
        mu        : [B, M, d]
        log_scale : [B, M, d]   clamped to [-2, 2]
        """
        E_in = E.clone()
        E_in[mask] = 0.0
        h = self.net(E_in.transpose(1, 2))         # [B, d, M]
        mu        = self.to_mu(h).transpose(1, 2)  # [B, M, d]
        log_scale = self.to_log_scale(h).transpose(1, 2).clamp(-2.0, 2.0)
        return mu, log_scale

    # ── Training forward ─────────────────────────────────────────────────────

    def forward_train(
        self, E: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Training pass: random mask, compute loss on masked positions only.

        Parameters
        ----------
        E : [B, M, d]

        Returns
        -------
        dyn_loss : scalar Tensor  Eq (8)
        u2_masked: [B, M] Tensor  u2 values (0 at unmasked positions)
        """
        B, M, _ = E.shape
        mask = make_window_mask(B, M, self.dyn_mask_ratio, E.device, training=True)
        mu, log_scale = self._predict(E, mask)

        # Eq 7: per-position u2
        u2 = self.energy(E - mu, log_scale=log_scale)   # [B, M]

        # Eq 8: loss only at masked positions
        if mask.any():
            dyn_loss = u2[mask].mean()
        else:
            dyn_loss = u2.new_tensor(0.0)

        # Zero out unmasked positions so the output is interpretable
        u2_out = torch.zeros_like(u2)
        u2_out[mask] = u2[mask]
        return dyn_loss, u2_out

    # ── Inference: leave-group-out ───────────────────────────────────────────

    @torch.no_grad()
    def forward_eval(self, E: torch.Tensor) -> torch.Tensor:
        """Inference pass: leave-group-out over K groups.

        Every position is predicted exactly once from a context that does
        not include that position.

        Parameters
        ----------
        E : [B, M, d]

        Returns
        -------
        u2 : [B, M]
        """
        B, M, _ = E.shape
        u2_out = torch.zeros(B, M, device=E.device, dtype=E.dtype)
        K = self.eval_groups
        for g in range(K):
            mask = torch.zeros(B, M, dtype=torch.bool, device=E.device)
            mask[:, g::K] = True
            mu, log_scale = self._predict(E, mask)
            e = self.energy(E - mu, log_scale=log_scale)   # [B, M]
            u2_out[mask] = e[mask]
        return u2_out


# Backward-compatible import name for checkpoints and third-party scripts.
IntraRunModule = ContextualTrajectoryModel
