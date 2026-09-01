"""Student-t robust energy — Eq (1) in the paper.

    ρ_ν(δ) = (ν+1)/2 · log(1 + ‖δ‖²_mean / ν)

where ‖·‖²_mean = mean over the last dimension, so the energy scale is
independent of descriptor / residual dimension. Non-negative by construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StudentTEnergy(nn.Module):
    """Non-negative robust energy based on the Student-t log-density.

    Parameters
    ----------
    nu : float
        Degrees of freedom. Lower values give heavier tails (more robust).
        Default 4.0 balances robustness with sensitivity to large deviations.
    """

    def __init__(self, nu: float = 4.0):
        super().__init__()
        self.nu = float(nu)

    def forward(
        self,
        delta: torch.Tensor,
        log_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-sample energy.

        Parameters
        ----------
        delta : Tensor [..., D]
            Residual or difference vector.  The last dimension is D.
        log_scale : Tensor broadcastable to delta, optional
            Per-feature log scale predicted by the dynamics head.
            Clamped to [-2, 2] to prevent scale-driven objective collapse.

        Returns
        -------
        Tensor [...]
            Non-negative energy with the last (D) dimension reduced.
        """
        if log_scale is not None:
            log_scale = log_scale.clamp(-2.0, 2.0)
            scale = torch.exp(log_scale).clamp_min(1e-6)
            delta = delta / scale

        # Mean-reduced squared norm — dimension-invariant
        dist2 = (delta ** 2).mean(dim=-1)
        energy = 0.5 * (self.nu + 1.0) * torch.log1p(dist2 / self.nu)
        return torch.clamp_min(energy, 0.0)
