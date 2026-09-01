"""Anti-collapse regularization — Eqs (15) and (16) in the paper.

Returns var_loss and cov_loss SEPARATELY so the model can apply
independent weights (w_var and w_cov) matching Eq (17).

  L_var  = (1/d) Σ_k max(0, γ − Std_n(e_k))            [Eq 15]
  L_cov  = (1/d) Σ_{k≠l} [Cov_n(e)]²_{kl}              [Eq 16]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiCollapseRegularizer(nn.Module):
    """Per-view variance hinge + off-diagonal covariance penalty.

    Parameters
    ----------
    gamma : float   lower bound on per-feature std (γ).
    eps   : float   numerical stability in sqrt.
    use_covariance : bool   set False for M5 ablation.
    """

    def __init__(
        self,
        gamma: float = 0.2,
        eps: float = 1e-4,
        use_covariance: bool = True,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.eps   = float(eps)
        self.use_covariance = bool(use_covariance)

    def _var_hinge(self, z: torch.Tensor):
        std  = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)  # [d]
        loss = F.relu(self.gamma - std).mean()
        return loss, std

    def _cov_off_diag(self, z: torch.Tensor) -> torch.Tensor:
        n, d = z.shape
        zc   = z - z.mean(dim=0, keepdim=True)
        cov  = (zc.t() @ zc) / max(n - 1, 1)
        off  = cov - torch.diag(torch.diag(cov))
        return (off ** 2).sum() / d

    def forward(
        self, ea: torch.Tensor, eb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        var_loss  : scalar  — Eq (15), averaged over both views
        cov_loss  : scalar  — Eq (16), averaged over both views (0 if disabled)
        std_mean  : scalar  — diagnostic
        std_min   : scalar  — diagnostic
        """
        var_a, std_a = self._var_hinge(ea)
        var_b, std_b = self._var_hinge(eb)
        var_loss = 0.5 * (var_a + var_b)

        if self.use_covariance:
            cov_loss = 0.5 * (self._cov_off_diag(ea) + self._cov_off_diag(eb))
        else:
            cov_loss = var_loss.new_tensor(0.0)

        std_all = torch.cat([std_a, std_b])
        return var_loss, cov_loss, std_all.mean(), std_all.min()
