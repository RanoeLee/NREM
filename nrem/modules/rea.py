"""Residual Evidence Aggregation (REA).

Implements the reference-calibration and weighted Cauchy-fusion equations from
the paper.

Multi-stream operation
----------------------
REA aggregates one or more streams of per-window energy (e.g. u1 and u2) by:
  1. calibrating each stream independently against its own training reference;
  2. averaging clipped Cauchy transforms within each stream; and
  3. combining stream statistics with explicit normalized weights.

The reported NREM uses weights (0.20, 0.80) for (u1, u2). Independent
calibration prevents their raw numerical scales from determining the fusion.
When weights are omitted, equal stream weighting is used for compatibility.

Reference construction (called once after training on the training set):
  - For each training sample j and each stream s ∈ {rec, u1, u2, u12},
    record its sorted window energies U^{(s)}_j ∈ R^M.
  - Store the reference matrices R^{(s)} = [U^{(s)}_1; …; U^{(s)}_N] in a
    single .npz file with keys 'rec_sorted', 'u1_sorted', 'u2_sorted',
    'u12_sorted'.  All four are computed and saved by the runner;
    inference picks whichever stream(s) the config requests.

Inference (called per test batch):
  - For each stream s and each test window energy u*_s, compute
    per-training-sample survival
      S^{(s)}_j(u*_s) = (1 + |{m : U^{(s)}_{j,m} ≥ u*_s}|) / (M+1)   [Eq 10]
  - Per-stream median p-value:
      p^{(s)}(u*_s) = median_j S^{(s)}_j(u*_s)                       [Eq 11]
  - Per-stream Cauchy statistic and weighted fusion:
      τ_s = (1/M) Σ_m clip(tan(π(0.5 - p^{(s)}_m)))                 [Eq 12]
      τ = Σ_s w_s τ_s,  Σ_s w_s = 1                               [Eq 13]
      p_comb = 0.5 - (1/π) arctan(τ)                                [Eq 14]
      S = −log(p_comb)                                               [Eq 15]

The K=1 case reproduces the original single-stream behaviour exactly.
"""
from __future__ import annotations

import warnings
import numpy as np
import torch
import torch.nn as nn


class ResidualEvidenceAggregator(nn.Module):
    """Stateless multi-stream scorer; reference is passed in at inference.

    The forward method accepts EITHER a single tensor pair (single-stream,
    legacy behaviour) OR a list of K (energy, reference) pairs (multi-stream,
    new default).  Single-stream input is wrapped internally and reproduces
    the original single-stream output exactly.
    """

    def __init__(self, cauchy_clip: float = 0.0):
        super().__init__()
        self.cauchy_clip = float(cauchy_clip)

    # ── Cauchy combination (Eqs 12–14) ──────────────────────────────────────

    def _cauchy_combine(self, p: torch.Tensor) -> torch.Tensor:
        """p of shape [B, L] → [B] anomaly scores.

        L is the number of p-values to combine per sample. For single-stream
        this is M (per-window); for K-stream multi-stream this is K·M
        (per (stream, window) pair). The math is identical either way —
        Cauchy combination treats each p-value as an independent piece of
        evidence and is asymptotically valid under arbitrary dependence
        (Liu and Xie, 2020).
        """
        eps = torch.finfo(p.dtype).eps
        p = p.clamp(eps, 1.0 - eps)
        stat = torch.tan(torch.pi * (0.5 - p))
        if self.cauchy_clip > 0:
            stat = stat.clamp(-self.cauchy_clip, self.cauchy_clip)
        tau = stat.mean(dim=1)
        p_comb = (0.5 - torch.atan(tau) / torch.pi).clamp(eps, 1.0)
        return -torch.log(p_comb)

    # ── Per-window median p-value (Eqs 10–11) ───────────────────────────────

    def _per_window_pvalue(
        self,
        window_energy: torch.Tensor,
        sample_ref_sorted: torch.Tensor,
    ) -> torch.Tensor:
        """Calibrate one stream of per-window energies into median p-values.

        Parameters
        ----------
        window_energy     : [B, M]
        sample_ref_sorted : [N_train, M_ref] (each row sorted ascending)

        Returns
        -------
        p : [B, M]  per-window median-reference p-value in (eps, 1-eps).
        """
        sample_ref = sample_ref_sorted.to(
            device=window_energy.device, dtype=window_energy.dtype
        )
        n_ref, m_ref = sample_ref.shape
        u = window_energy.reshape(-1)                            # [B*M]
        u_exp = u.unsqueeze(0).expand(n_ref, -1).contiguous()    # [N, B*M]
        idx = torch.searchsorted(sample_ref.contiguous(), u_exp, right=False)
        count_ge = m_ref - idx                                   # [N, B*M]
        survival = (count_ge.float() + 1.0) / (m_ref + 1.0)      # Eq 10
        p = survival.median(dim=0).values                        # [B*M] Eq 11
        eps = 1.0 / (m_ref + 1.0)
        return p.clamp(eps, 1.0 - eps).view_as(window_energy)    # [B, M]

    # ── Forward: single- or multi-stream dispatch ───────────────────────────

    def forward(
        self,
        window_energies,
        sample_refs_sorted,
        stream_weights=None,
    ) -> torch.Tensor:
        """Score a batch of test samples.

        Parameters
        ----------
        window_energies : Tensor [B, M]  OR  list/tuple of K such tensors
            Per-window energy for each stream. When explicit weights are used,
            list order must match the weight order.
        sample_refs_sorted : Tensor [N, M_ref]  OR  list/tuple of K such
            Per-stream training-set reference matrices (each row sorted).
            Length and order must match `window_energies`.

        Returns
        -------
        Tensor [B]  non-negative anomaly scores.

        Notes
        -----
        When both arguments are single tensors, the behaviour reproduces
        the original single-stream MR-RESA bit-identically.
        """
        # Normalise tensor inputs to single-element lists for uniform handling.
        if torch.is_tensor(window_energies):
            window_energies = [window_energies]
        if torch.is_tensor(sample_refs_sorted):
            sample_refs_sorted = [sample_refs_sorted]

        if len(window_energies) != len(sample_refs_sorted):
            raise ValueError(
                f"REA: got {len(window_energies)} energy stream(s) but "
                f"{len(sample_refs_sorted)} reference matri(x|ces). Counts "
                "must match."
            )
        if not window_energies:
            raise ValueError("REA: at least one energy stream is required.")

        # Per-stream calibration: list of [B, M] p-value tensors.
        pvalues = [
            self._per_window_pvalue(we, ref)
            for we, ref in zip(window_energies, sample_refs_sorted)
        ]

        # Concatenate across the window dimension → [B, K·M].
        if stream_weights is None:
            p_all = torch.cat(pvalues, dim=1)
            return self._cauchy_combine(p_all)

        if len(stream_weights) != len(pvalues):
            raise ValueError(
                f"REA: got {len(stream_weights)} stream weight(s) for "
                f"{len(pvalues)} stream(s)."
            )
        weights = torch.as_tensor(
            stream_weights, device=pvalues[0].device, dtype=pvalues[0].dtype
        )
        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError(
                "REA stream weights must be finite, non-negative, and sum to a positive value."
            )
        weights = weights / weights.sum()

        eps = torch.finfo(pvalues[0].dtype).eps
        stream_stats = []
        for p in pvalues:
            p = p.clamp(eps, 1.0 - eps)
            stat = torch.tan(torch.pi * (0.5 - p))
            if self.cauchy_clip > 0:
                stat = stat.clamp(-self.cauchy_clip, self.cauchy_clip)
            stream_stats.append(stat.mean(dim=1))
        tau = sum(weight * stat for weight, stat in zip(weights, stream_stats))
        p_comb = (0.5 - torch.atan(tau) / torch.pi).clamp(eps, 1.0)
        return -torch.log(p_comb)


# Backward-compatible import name for existing user code.
MRResa = ResidualEvidenceAggregator


# ── Reference construction helpers ─────────────────────────────────────────

def build_reference(all_window_energies: np.ndarray) -> dict:
    """Build per-sample sorted reference from training-set window energies.

    Parameters
    ----------
    all_window_energies : np.ndarray [N, M]

    Returns
    -------
    dict with key 'sample_ref_sorted' [N, M] (np.float32).
    """
    if all_window_energies.ndim != 2 or all_window_energies.size == 0:
        raise ValueError(
            f"Expected 2-D [N, M] array, got shape {all_window_energies.shape}"
        )
    ref_sorted = np.sort(all_window_energies.astype(np.float32), axis=1)
    return {"sample_ref_sorted": ref_sorted}


def save_reference(path, ref: dict) -> None:
    """Save reference dict to a compressed .npz file."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **ref)


# ── Key migration: old → new format ────────────────────────────────────────

# Maps old single-key names to the new resa_energy type they best represent.
# Old runs stored one reference under 'ref_sorted' or 'sample_ref_sorted'
# (pooled view energy).  We expose them under all three new keys so any
# resa_energy config ('u12', 'u1', 'rec') can evaluate without retraining.
_OLD_KEY_ALIASES = {
    "ref_sorted":        "legacy",
    "sample_ref_sorted": "sample_ref_sorted",   # new format single-key (rare)
}


def _migrate_old_reference(z) -> dict | None:
    """Try to load an old-format .npz and return a new-format dict.

    Returns None if the file is already in the new multi-key format.
    """
    files = set(z.files)

    # Already new format if any of the typed keys are present
    new_keys = {"u12_sorted", "u1_sorted", "u2_sorted", "rec_sorted"}
    if files & new_keys:
        return None   # handled by caller

    # Detect old single-key formats
    old_arr = None
    for old_key in ("ref_sorted", "sample_ref_sorted"):
        if old_key in files:
            old_arr = np.asarray(z[old_key], dtype=np.float32)
            warnings.warn(
                f"[MR-RESA] Loaded legacy reference key '{old_key}'. "
                "This reference was built with an older codebase. "
                "Run  python tools/rebuild_references.py --runs_root <path>  "
                "to rebuild proper multi-key references without retraining. "
                "For now, the old reference is used for all energy types.",
                UserWarning,
                stacklevel=3,
            )
            break

    if old_arr is None:
        return None   # unknown format; let caller raise the proper error

    # Reshape if it was stored 1-D (very old pooled format)
    if old_arr.ndim == 1:
        old_arr = old_arr.reshape(1, -1)

    # Expose under all typed keys so any resa_energy setting works
    return {
        "u12_sorted": old_arr,
        "u1_sorted":  old_arr,
        "u2_sorted":  old_arr,
        "rec_sorted": old_arr,
        "n_samples":  int(old_arr.shape[0]),
        "m_windows":  int(old_arr.shape[1]),
        "_migrated":  True,
    }


def load_reference(path, device: str = "cpu") -> dict:
    """Load reference from .npz and move tensors to device.

    Handles both the new multi-key format (u12_sorted, u1_sorted, …) and
    old single-key formats (ref_sorted, sample_ref_sorted) from previous
    versions of the codebase.
    """
    z = np.load(path, allow_pickle=False)
    files = set(z.files)

    # ── Try migration first ──────────────────────────────────────────────────
    migrated = _migrate_old_reference(z)
    if migrated is not None:
        out = {}
        for k, v in migrated.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v).to(device)
            else:
                out[k] = v
        return out

    # ── New multi-key format ─────────────────────────────────────────────────
    new_keys = {"u12_sorted", "u1_sorted", "u2_sorted", "rec_sorted"}
    found    = files & new_keys
    if not found:
        raise RuntimeError(
            f"MR-RESA reference at {path} has unrecognised keys: {files}. "
            "Run  python tools/rebuild_references.py --runs_root <path>  "
            "to rebuild the reference, or re-run training."
        )

    out: dict = {}
    for k in found:
        arr = np.asarray(z[k], dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            raise RuntimeError(f"Invalid reference shape {arr.shape} for key '{k}' in {path}")
        out[k] = torch.from_numpy(arr).to(device)

    # Infer metadata from any available array
    sample_arr = next(iter(out.values()))
    out["n_samples"] = int(sample_arr.shape[0])
    out["m_windows"] = int(sample_arr.shape[1])
    return out
