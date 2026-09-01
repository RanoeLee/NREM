"""Nested Residual Energy Model (NREM) — assembled hierarchical model.

Maps directly to the paper's methodology section:
  Section 4  — Backbone + decoupled reconstruction          (rec_window)
  Section 5  — SRE                →  u1_window              (within-sample)
  Section 6  — CTM                →  u2_window              (within-condition)
  Section 7  — REA                →  sample score           (global)
  Section 8  — AntiCollapseRegularizer
  Section 9  — End-to-end joint loss                        Eq (17)
  Section 10 — Inference

Config keys under 'nrem:' in the YAML:
  window_size, stride              int
  backbone                         'conv' | 'mlp'
  conv_hidden                      list[int]  default [32,64,128]
  conv_latent, conv_kernel         int
  latent_dim, hidden_dim           int  (mlp only)
  d_model                          int  default 128
  residual_enc_hidden              list[int]  default [64,128]
  use_view                         bool
  reconstruction_mode              'full'|'decoupled'|'masked_input'
  mask_ratio                       float  κ₁  default 0.30
  view_mode                        'byol'|'symmetric'
  predictor_hidden_dim             int
  student_t_nu                     float  default 4.0
  use_dynamics                     bool
  dyn_mask_ratio                   float  κ₂  default 0.15
  dynamics_hidden, dynamics_layers int
  eval_dyn_groups                  int   default 4
  use_covariance                   bool
  var_gamma, var_eps               float
  loss_weights.rec/view/dyn/var/cov  float
  score_mode                       str
  resa_energy                      single stream  ('u12'|'u1'|'u2'|'rec')
                                   or LIST of streams (e.g. ['u1','u2'])
                                   for multi-stream MR-RESA (new default)
  resa_stream_weights              optional list[float], aligned with
                                   resa_energy and normalized internally
  resa_cauchy_clip                 float
  dropout                          float
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import ConvWindowAE, MLPWindowAE
from .sre import SampleResidualEvidence
from .ctm import ContextualTrajectoryModel
from ..modules.student_t import StudentTEnergy
from ..modules.anti_collapse import AntiCollapseRegularizer
from ..modules.windowing import windowize
from ..modules.masking import complementary_element_mask
from ..modules.rea import ResidualEvidenceAggregator


@dataclass
class NREMForward:
    loss:          Optional[torch.Tensor]
    sample_score:  torch.Tensor   # [B]  raw mean-energy score (no RESA)
    window_energy: torch.Tensor   # [B, M]  u1 + u2
    rec_window:    torch.Tensor   # [B, M]
    u1_window:     torch.Tensor   # [B, M]  within-sample
    u2_window:     torch.Tensor   # [B, M]  within-condition
    logs:          Dict[str, float]


class NREMModel(nn.Module):
    """Hierarchical residual evidence model.  Instantiate via cfg dict."""

    def __init__(self, cfg: dict):
        super().__init__()
        ncfg = cfg.get("nrem", {}) or {}
        dcfg = cfg.get("data", {}) or {}
        mcfg = cfg.get("model", {}) or {}

        # Dimensions
        self.channels    = int(dcfg.get("channels", mcfg.get("channels", 6)))
        self.window_size = int(ncfg.get("window_size", 1024))
        self.stride      = int(ncfg.get("stride", 512))
        self.d_model     = int(ncfg.get("d_model", 128))
        self.input_dim   = self.channels * self.window_size
        max_len          = int(dcfg.get("max_len", 48000))
        self.m_windows   = self._n_windows(max_len, self.window_size, self.stride)

        # Flags
        self.use_view     = bool(ncfg.get("use_view", True))
        self.use_dynamics = bool(ncfg.get("use_dynamics", True))
        self.rec_mode     = str(ncfg.get("reconstruction_mode", "decoupled")).lower()
        if self.rec_mode not in {"full", "decoupled", "masked_input"}:
            raise ValueError(f"Unknown reconstruction_mode: {self.rec_mode}")

        nu      = float(ncfg.get("student_t_nu", 4.0))
        dropout = float(ncfg.get("dropout", 0.0))

        # Loss weights — Eq (17)
        lw = ncfg.get("loss_weights", {}) or {}
        self.w_rec  = float(lw.get("rec",  1.0))
        self.w_view = float(lw.get("view", 0.1))
        self.w_dyn  = float(lw.get("dyn",  0.1))
        self.w_var  = float(lw.get("var",  0.05))
        self.w_cov  = float(lw.get("cov",  1.0))

        # Scoring
        self.default_score_mode = str(ncfg.get("score_mode", "rea")).lower()
        # resa_energy: a single stream name ('u1' | 'u2' | 'u12' | 'rec') or
        # a list of stream names for multi-stream MR-RESA.  Lists trigger the
        # independent-calibration + Cauchy-combine pipeline (the new default).
        _VALID_STREAMS = {"u1", "u2", "u12", "rec"}
        _re_raw = ncfg.get("resa_energy", "u12")
        if isinstance(_re_raw, (list, tuple)):
            self.resa_energy = [str(e).lower() for e in _re_raw]
            for _e in self.resa_energy:
                if _e not in _VALID_STREAMS:
                    raise ValueError(
                        f"Unknown resa_energy stream: {_e!r}. "
                        f"Valid streams: {sorted(_VALID_STREAMS)}."
                    )
            if len(self.resa_energy) == 0:
                raise ValueError("resa_energy list must contain at least one stream.")
        else:
            self.resa_energy = str(_re_raw).lower()
            if self.resa_energy not in _VALID_STREAMS:
                raise ValueError(
                    f"Unknown resa_energy: {self.resa_energy!r}. "
                    f"Valid streams: {sorted(_VALID_STREAMS)}."
                )

        _weights_raw = ncfg.get("resa_stream_weights")
        self.resa_stream_weights = None
        if _weights_raw is not None:
            if not isinstance(_weights_raw, (list, tuple)):
                raise ValueError("resa_stream_weights must be a list aligned with resa_energy.")
            expected = len(self.resa_energy) if isinstance(self.resa_energy, list) else 1
            if len(_weights_raw) != expected:
                raise ValueError(
                    f"resa_stream_weights has {len(_weights_raw)} value(s), expected {expected}."
                )
            weights = [float(value) for value in _weights_raw]
            if any((not math.isfinite(value)) or value < 0 for value in weights) or sum(weights) <= 0:
                raise ValueError(
                    "resa_stream_weights must be finite, non-negative, and sum to a positive value."
                )
            total = sum(weights)
            self.resa_stream_weights = [value / total for value in weights]

        # Backbone
        bb = str(ncfg.get("backbone", "conv")).lower()
        if bb == "conv":
            self.backbone = ConvWindowAE(
                channels=self.channels,
                window_size=self.window_size,
                hidden_channels=tuple(ncfg.get("conv_hidden", [32, 64, 128])),
                latent_channels=int(ncfg.get("conv_latent", 64)),
                kernel_size=int(ncfg.get("conv_kernel", 7)),
                dropout=dropout,
            )
        elif bb == "mlp":
            self.backbone = MLPWindowAE(
                input_dim=self.input_dim,
                latent_dim=int(ncfg.get("latent_dim", 256)),
                hidden_dim=int(ncfg.get("hidden_dim", 1024)),
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown backbone: {bb}")

        # Reconstruction energy
        self.rec_energy = StudentTEnergy(nu=nu)
        # rec_loss controls the per-window reconstruction objective:
        #   'student_t' — robust Student-t energy (default, contamination-aware)
        #   'mse'       — plain mean squared error (conventional AE, no robustness)
        self.rec_loss_type = str(ncfg.get("rec_loss", "student_t")).lower()
        if self.rec_loss_type not in {"student_t", "mse"}:
            raise ValueError(
                f"rec_loss must be 'student_t' or 'mse', got '{self.rec_loss_type}'"
            )

        # Within-sample
        if self.use_view:
            self.intra_sample = SampleResidualEvidence(
                channels=self.channels,
                window_size=self.window_size,
                d_model=self.d_model,
                residual_enc_hidden=tuple(ncfg.get("residual_enc_hidden", [64, 128])),
                predictor_hidden_dim=int(ncfg.get("predictor_hidden_dim", 256)),
                mask_ratio=float(ncfg.get("mask_ratio", 0.30)),
                nu=nu,
                view_mode=str(ncfg.get("view_mode", "byol")),
                kernel_size=int(ncfg.get("conv_kernel", 7)),
                dropout=dropout,
            )
            self.anti_collapse = AntiCollapseRegularizer(
                gamma=float(ncfg.get("var_gamma", 0.2)),
                eps=float(ncfg.get("var_eps", 1e-4)),
                use_covariance=bool(ncfg.get("use_covariance", True)),
            )
        else:
            self.intra_sample  = None
            self.anti_collapse = None

        # Within-condition
        if self.use_dynamics:
            self.intra_run = ContextualTrajectoryModel(
                d_model=self.d_model,
                hidden=int(ncfg.get("dynamics_hidden", 256)),
                layers=int(ncfg.get("dynamics_layers", 4)),
                dropout=dropout,
                nu=nu,
                dyn_mask_ratio=float(ncfg.get("dyn_mask_ratio", 0.15)),
                eval_groups=int(ncfg.get("eval_dyn_groups", 4)),
            )
        else:
            self.intra_run = None

        self.mr_resa = ResidualEvidenceAggregator(
            cauchy_clip=float(ncfg.get("resa_cauchy_clip", 0.0))
        )

    @property
    def sre(self):
        """Paper-named access to Sample Residual Evidence.

        The registered attribute remains ``intra_sample`` so checkpoints from
        the experiments retain identical state-dict keys.
        """
        return self.intra_sample

    @property
    def ctm(self):
        """Paper-named access to Contextual Trajectory Modeling."""
        return self.intra_run

    @property
    def rea(self):
        """Paper-named access to Residual Evidence Aggregation."""
        return self.mr_resa

    # Helpers
    @staticmethod
    def _n_windows(t: int, w: int, s: int) -> int:
        return 1 if t < w else 1 + (t - w) // s

    def _zeros_bm(self, b, m, ref):
        return torch.zeros(b, m, device=ref.device, dtype=ref.dtype)

    def _zeros_e(self, bm, ref):
        return torch.zeros(bm, self.d_model, device=ref.device, dtype=ref.dtype)

    def _get_window_energy(self, out: NREMForward, key: str) -> torch.Tensor:
        return {"u12": out.u1_window + out.u2_window,
                "u1":  out.u1_window,
                "u2":  out.u2_window,
                "rec": out.rec_window}[key]

    # Forward
    def forward(self, x: torch.Tensor, training: bool = True) -> NREMForward:
        win, _ = windowize(x, self.window_size, self.stride)  # [B,M,C,W]
        B, M, C, W = win.shape
        flat = win.reshape(B * M, C * W)                      # [B*M, C*W]

        # ── Backbone ──────────────────────────────────────────────────────────
        if self.rec_mode in {"full", "decoupled"}:
            recon, _ = self.backbone(flat)
            r         = flat - recon                          # [B*M, C*W]
            mi_masks  = None
        else:
            # M1 ablation: mask the AE input, not the residual
            xw   = flat.view(B * M, C, W)
            kr   = self.sre.mask_ratio if self.sre else 0.5
            ma_i, mb_i = complementary_element_mask(xw, mask_ratio=kr, training=training)
            recon_a, _ = self.backbone((ma_i * xw).flatten(1))
            recon_b, _ = self.backbone((mb_i * xw).flatten(1))
            r           = 0.5 * ((flat - recon_a) + (flat - recon_b))
            mi_masks    = (ma_i, mb_i, recon_a, recon_b)

        # Per-window reconstruction energy — Student-t (robust) or plain MSE
        if self.rec_loss_type == "mse":
            rec_w = (r ** 2).mean(dim=-1).view(B, M)          # [B, M] plain MSE
        else:
            rec_w = self.rec_energy(r).view(B, M)             # [B, M] Student-t

        # ── Within-sample (Section 5) ─────────────────────────────────────────
        u1_w      = self._zeros_bm(B, M, flat)
        E         = self._zeros_e(B * M, flat).view(B, M, self.d_model)
        var_loss  = flat.new_tensor(0.0)
        cov_loss  = flat.new_tensor(0.0)
        e_std_mean = flat.new_tensor(0.0)
        e_std_min  = flat.new_tensor(0.0)
        ea = eb = None

        if self.use_view and self.sre is not None:
            if self.rec_mode in {"full", "decoupled"}:
                r_sg = r.detach().view(B * M, C, W)
                u1, ea, eb, e = self.sre(r_sg, training=training)
            else:
                # M1: encode per-view residuals directly (already masked)
                r_a = (flat - mi_masks[2]).detach().view(B * M, C, W)
                r_b = (flat - mi_masks[3]).detach().view(B * M, C, W)
                ea  = self.sre.residual_encoder(mi_masks[0] * r_a)
                eb  = self.sre.residual_encoder(mi_masks[1] * r_b)
                if self.sre.view_mode == "byol":
                    pa  = self.sre.predictor(ea)
                    pb  = self.sre.predictor(eb)
                    u1  = 0.5 * (self.sre.energy(pa - eb.detach()) +
                                 self.sre.energy(pb - ea.detach()))
                else:
                    u1 = self.sre.energy(ea - eb)
                e  = 0.5 * (ea + eb)

            u1_w = u1.view(B, M)
            E    = e.view(B, M, self.d_model)
            var_loss, cov_loss, e_std_mean, e_std_min = self.anti_collapse(ea, eb)

        # ── Within-condition (Section 6) ──────────────────────────────────────
        dyn_loss = flat.new_tensor(0.0)
        u2_w     = self._zeros_bm(B, M, flat)

        if self.use_dynamics and self.ctm is not None:
            if training:
                dyn_loss, _ = self.ctm.forward_train(E)
                # u2_w stays 0 during training; loss captured in dyn_loss
            else:
                u2_w = self.ctm.forward_eval(E)

        # ── Combined energy and raw score ─────────────────────────────────────
        window_energy = u1_w + u2_w
        sample_score  = window_energy.mean(dim=1)

        # ── Loss — Eq (17) ────────────────────────────────────────────────────
        loss = None
        if training:
            view_loss = u1_w.mean() if self.use_view else flat.new_tensor(0.0)
            loss = (
                self.w_rec  * rec_w.mean()
              + self.w_view * view_loss
              + self.w_dyn  * dyn_loss
              + self.w_var  * var_loss
              + self.w_cov  * cov_loss
            )

        logs: Dict[str, float] = {
            "rec":        float(rec_w.mean().detach().cpu()),
            "u1":         float(u1_w.mean().detach().cpu()),
            "dyn_loss":   float(dyn_loss.detach().cpu()),
            "var":        float(var_loss.detach().cpu()),
            "cov":        float(cov_loss.detach().cpu()),
            "e_std_mean": float(e_std_mean.detach().cpu()),
            "e_std_min":  float(e_std_min.detach().cpu()),
            "score":      float(sample_score.mean().detach().cpu()),
        }
        return NREMForward(loss=loss, sample_score=sample_score,
                           window_energy=window_energy, rec_window=rec_w,
                           u1_window=u1_w, u2_window=u2_w, logs=logs)

    # Score
    @torch.no_grad()
    def score(
        self,
        x: torch.Tensor,
        score_mode: str | None = None,
        resa_ref: dict | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Assign anomaly scores.  See module docstring for score_mode values."""
        was_training = self.training
        self.eval()
        out = self.forward(x, training=False)
        if was_training:
            self.train()

        mode = (score_mode or self.default_score_mode).lower()
        if mode == "rec_mean":
            s = out.rec_window.mean(dim=1)
        elif mode == "u1_mean":
            s = out.u1_window.mean(dim=1)
        elif mode in {"u12_mean", "mean"}:
            s = (out.u1_window + out.u2_window).mean(dim=1)
        elif mode in {"rea", "mr_resa"}:
            if resa_ref is None:
                raise ValueError("score_mode='rea' requires a training reference.")

            # Streams to aggregate.  String → 1-element list (single-stream).
            streams = self.resa_energy
            if isinstance(streams, str):
                streams = [streams]

            _STREAM_TO_REFKEY = {"u12": "u12_sorted", "u1": "u1_sorted",
                                 "u2":  "u2_sorted",  "rec": "rec_sorted"}
            # Per-key fallback order for references built before u2 was
            # collected separately.  Use rebuild_references.py to fix properly.
            _fallback_order = {
                "u2_sorted":  ["u12_sorted", "u1_sorted"],
                "u12_sorted": ["u1_sorted",  "u2_sorted"],
                "u1_sorted":  ["u12_sorted"],
                "rec_sorted": ["u12_sorted", "u1_sorted"],
            }

            refs = []
            energies = []
            for stream in streams:
                key = _STREAM_TO_REFKEY[stream]
                if key not in resa_ref:
                    fallback = next(
                        (k for k in _fallback_order.get(key, []) if k in resa_ref),
                        None,
                    )
                    if fallback is None:
                        raise ValueError(
                            f"resa_ref missing '{key}' and no fallback available. "
                            f"Rebuild reference for stream='{stream}'."
                        )
                    import warnings
                    warnings.warn(
                        f"[REA] reference missing '{key}', using '{fallback}' "
                        f"as a temporary substitute. Run "
                        f"'python tools/rebuild_references.py --runs_root <path>' "
                        f"to rebuild proper references.",
                        UserWarning, stacklevel=2,
                    )
                    key = fallback
                refs.append(resa_ref[key])
                energies.append(self._get_window_energy(out, stream))

            # Single-tensor input when only one stream (preserves legacy
            # behaviour bit-exactly); list input for multi-stream.
            if len(energies) == 1:
                s = self.rea(energies[0], refs[0])
            else:
                s = self.rea(
                    energies, refs, stream_weights=self.resa_stream_weights
                )
        else:
            raise ValueError(f"Unknown score_mode: {mode!r}")

        aux = {k: getattr(out, k) for k in
               ("rec_window", "u1_window", "u2_window",
                "window_energy", "sample_score")}
        return s, aux
