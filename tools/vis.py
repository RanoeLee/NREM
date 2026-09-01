#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export SRE intermediate data and visualization assets during inference.

Designed for the current NREM-main project structure.

This script exports:
  1. residual windows:          r_m = x_m - x_hat_m
  2. SRE complementary masks:   M_a, M_b
  3. masked residual views:     M_a * r_m, M_b * r_m
  4. SRE descriptors:           e_a, e_b, e
  5. SRE evidence curve:        u_1^(m)
  6. CTM evidence curve:        u_2^(m), if CTM is enabled
  7. original strip heatmaps and 1:1 square maps for residuals and real masks

Recommended usage from project root:

python tools/export_sre.py ^
  --run_dir runs/your_run_dir ^
  --sample_path E:/Datasets/Midea-datasets/output_npy_new/val/ng/NG_000168.npy ^
  --ckpt_name ema.pt ^
  --out_dir vis_sre/NG_000168 ^
  --select_window top_u1 ^
  --use_abs_residual

If ema.pt is corrupted or unavailable, try:
  --ckpt_name best.pt
or:
  --ckpt_name last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Make project imports work when running from project root or tools/
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from nrem.config import load_config
from nrem.methods import build_method
from nrem.engine.checkpoint import load_checkpoint
from nrem.data.dataset import _load_npy, standardize_array_shape, pad_or_crop
from nrem.modules.windowing import windowize
from nrem.modules.masking import complementary_element_mask


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def load_normalizer(run_dir: Path):
    norm_path = run_dir / "normalizer.npz"
    if not norm_path.exists():
        return None
    z = np.load(norm_path)
    return {k: z[k] for k in z.files}


def apply_normalization(x: np.ndarray, cfg: dict, norm_stats):
    normalize = str((cfg.get("data") or {}).get("normalize", "none")).lower()

    if normalize == "none" or norm_stats is None:
        return x.astype(np.float32, copy=False)

    if normalize == "zscore":
        mean = norm_stats["mean"].reshape(-1, 1)
        std = norm_stats["std"].reshape(-1, 1)
        x = (x - mean) / np.maximum(std, 1e-6)

    elif normalize == "minmax":
        mn = norm_stats["min"].reshape(-1, 1)
        mx = norm_stats["max"].reshape(-1, 1)
        x = (x - mn) / np.maximum(mx - mn, 1e-6)

    elif normalize in {"robust_mad", "mad", "median_mad"}:
        median = norm_stats["median"].reshape(-1, 1)
        scale = norm_stats["robust_scale"].reshape(-1, 1)
        x = (x - median) / np.maximum(scale, 1e-6)

    else:
        raise ValueError(f"Unknown normalization mode: {normalize}")

    return x.astype(np.float32, copy=False)


def load_one_sample(sample_path: Path, cfg: dict, norm_stats):
    dcfg = cfg.get("data", {}) or {}

    x = _load_npy(sample_path)
    x = standardize_array_shape(
        x,
        transpose_if_tall=bool(dcfg.get("transpose_if_tall", True)),
    )

    expected_channels = int(dcfg.get("channels", (cfg.get("model") or {}).get("channels", 6)))
    if bool(dcfg.get("strict_checks", True)) and x.shape[0] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels, but got {x.shape[0]} "
            f"from {sample_path}"
        )

    x = apply_normalization(x, cfg, norm_stats)

    max_len = int(dcfg.get("max_len", 48000))
    x = pad_or_crop(x, max_len=max_len)

    return x.astype(np.float32, copy=False)


def find_checkpoint(run_dir: Path, ckpt_name: str):
    candidates = [
        run_dir / "checkpoints" / ckpt_name,
        run_dir / ckpt_name,
        run_dir / "checkpoints" / "ema.pt",
        run_dir / "checkpoints" / "best.pt",
        run_dir / "checkpoints" / "last.pt",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No checkpoint found in {run_dir}. Tried:\n" +
        "\n".join(str(p) for p in candidates)
    )


def robust_vlim_signed(arr, percentile=99):
    vmax = np.nanpercentile(np.abs(arr), percentile)
    vmax = max(float(vmax), 1e-6)
    return -vmax, vmax


def robust_vlim_positive(arr, q_low=1, q_high=99):
    vmin = np.nanpercentile(arr, q_low)
    vmax = np.nanpercentile(arr, q_high)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


# ---------------------------------------------------------------------
# Core inference extraction
# ---------------------------------------------------------------------

@torch.no_grad()
def extract_sre_intermediates(method, x: torch.Tensor, training_mask: bool = False):
    """
    Reproduce the NREM forward path and explicitly cache SRE intermediate data.

    Parameters
    ----------
    method:
        NREMMethod instance.
    x:
        Tensor [1, C, T].
    training_mask:
        If False, uses deterministic alternating mask, matching inference.
        If True, samples random complementary masks.

    Returns
    -------
    cache: dict of tensors on CPU.
    """
    model = method.model
    model.eval()

    win, _ = windowize(x, model.window_size, model.stride)  # [B, M, C, W]
    B, M, C, W = win.shape
    flat = win.reshape(B * M, C * W)

    # ---------------------------------------------------------
    # Backbone reconstruction
    # ---------------------------------------------------------
    if model.rec_mode in {"full", "decoupled"}:
        recon, latent = model.backbone(flat)
        r = flat - recon
        r_sg = r.detach().view(B * M, C, W)

    elif model.rec_mode == "masked_input":
        # For WS_mask_input ablation.
        xw = flat.view(B * M, C, W)
        kr = model.intra_sample.mask_ratio if model.intra_sample else 0.5
        ma_i, mb_i = complementary_element_mask(
            xw,
            mask_ratio=kr,
            training=False,
        )
        recon_a, _ = model.backbone((ma_i * xw).flatten(1))
        recon_b, _ = model.backbone((mb_i * xw).flatten(1))
        r = 0.5 * ((flat - recon_a) + (flat - recon_b))
        recon = 0.5 * (recon_a + recon_b)
        r_sg = r.detach().view(B * M, C, W)
        latent = None

    else:
        raise ValueError(f"Unknown reconstruction mode: {model.rec_mode}")

    # Reconstruction energy
    if model.rec_loss_type == "mse":
        rec_w = (r ** 2).mean(dim=-1).view(B, M)
    else:
        rec_w = model.rec_energy(r).view(B, M)

    # ---------------------------------------------------------
    # SRE extraction
    # ---------------------------------------------------------
    if not model.use_view or model.intra_sample is None:
        raise RuntimeError("This model has use_view=False; no SRE module is available.")

    intra = model.intra_sample

    mask_a, mask_b = complementary_element_mask(
        r_sg,
        mask_ratio=intra.mask_ratio,
        training=training_mask,
    )

    view_a = mask_a * r_sg
    view_b = mask_b * r_sg

    e_a = intra.residual_encoder(view_a)
    e_b = intra.residual_encoder(view_b)

    if intra.view_mode == "byol":
        pred_a = intra.predictor(e_a)
        pred_b = intra.predictor(e_b)
        u1 = 0.5 * (
            intra.energy(pred_a - e_b.detach()) +
            intra.energy(pred_b - e_a.detach())
        )
    else:
        pred_a = torch.zeros_like(e_a)
        pred_b = torch.zeros_like(e_b)
        u1 = intra.energy(e_a - e_b)

    e = 0.5 * (e_a + e_b)

    u1_w = u1.view(B, M)
    E = e.view(B, M, model.d_model)

    # ---------------------------------------------------------
    # CTM evidence, useful for side-by-side plot
    # ---------------------------------------------------------
    if model.use_dynamics and model.intra_run is not None:
        u2_w = model.intra_run.forward_eval(E)
    else:
        u2_w = torch.zeros_like(u1_w)

    window_energy = u1_w + u2_w

    cache = {
        "x_windows": win.detach().cpu(),                         # [1, M, C, W]
        "x_hat_windows": recon.view(B, M, C, W).detach().cpu(),   # [1, M, C, W]
        "residual_windows": r_sg.view(B, M, C, W).detach().cpu(), # [1, M, C, W]

        "mask_a": mask_a.view(B, M, C, W).detach().cpu(),
        "mask_b": mask_b.view(B, M, C, W).detach().cpu(),
        "view_a": view_a.view(B, M, C, W).detach().cpu(),
        "view_b": view_b.view(B, M, C, W).detach().cpu(),

        "e_a": e_a.view(B, M, -1).detach().cpu(),
        "e_b": e_b.view(B, M, -1).detach().cpu(),
        "pred_a": pred_a.view(B, M, -1).detach().cpu(),
        "pred_b": pred_b.view(B, M, -1).detach().cpu(),
        "e": E.detach().cpu(),

        "rec_window": rec_w.detach().cpu(),
        "u1": u1_w.detach().cpu(),
        "u2": u2_w.detach().cpu(),
        "window_energy": window_energy.detach().cpu(),
    }

    return cache


# ---------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------

def select_window(cache: dict, mode: str, explicit_idx: int | None = None):
    u1 = cache["u1"].numpy()[0]
    u2 = cache["u2"].numpy()[0]
    rec = cache["rec_window"].numpy()[0]
    residual = cache["residual_windows"].numpy()[0]  # [M, C, W]

    if explicit_idx is not None:
        idx = int(explicit_idx)
        if idx < 0 or idx >= len(u1):
            raise ValueError(f"--window_idx={idx} out of range [0, {len(u1)-1}]")
        return idx

    mode = mode.lower()

    if mode == "top_u1":
        return int(np.argmax(u1))

    if mode == "top_u2":
        return int(np.argmax(u2))

    if mode == "top_rec":
        return int(np.argmax(rec))

    if mode == "top_residual_abs":
        strength = np.abs(residual).mean(axis=(1, 2))
        return int(np.argmax(strength))

    if mode == "middle":
        return int(len(u1) // 2)

    raise ValueError(f"Unknown select_window mode: {mode}")


# ---------------------------------------------------------------------
# Residual square semantic maps
# ---------------------------------------------------------------------

def residual_to_square_map(
    residual,
    time_bins=32,
    square_size=32,
    metric="absmean",
):
    """
    Convert residual [C, W] into a 1:1 square semantic map.

    This preserves the channel-time semantics better than directly resizing or
    reshaping the flattened residual:
      1. temporal pooling: [C, W] -> [C, time_bins]
      2. channel-axis interpolation: [C, time_bins] -> [square_size, square_size]

    Parameters
    ----------
    residual : np.ndarray
        Shape [C, W], e.g. [6, 1024].
    time_bins : int
        Number of coarse temporal bins before square interpolation.
    square_size : int
        Final square size, e.g. 32 or 48.
    metric : str
        One of ["absmean", "rms", "signedmean", "absmax"].

    Returns
    -------
    square_map : np.ndarray
        Shape [square_size, square_size].
    pooled_map : np.ndarray
        Shape [C, time_bins], before channel interpolation.
    """
    residual = np.asarray(residual, dtype=np.float32)
    C, W = residual.shape

    edges = np.linspace(0, W, time_bins + 1, dtype=int)
    pooled = np.zeros((C, time_bins), dtype=np.float32)

    for j in range(time_bins):
        seg = residual[:, edges[j]:edges[j + 1]]
        if seg.shape[1] == 0:
            continue

        if metric == "absmean":
            pooled[:, j] = np.mean(np.abs(seg), axis=1)
        elif metric == "rms":
            pooled[:, j] = np.sqrt(np.mean(seg ** 2, axis=1))
        elif metric == "signedmean":
            pooled[:, j] = np.mean(seg, axis=1)
        elif metric == "absmax":
            pooled[:, j] = np.max(np.abs(seg), axis=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    x = torch.from_numpy(pooled).unsqueeze(0).unsqueeze(0)  # [1,1,C,time_bins]
    x_sq = F.interpolate(
        x,
        size=(square_size, square_size),
        mode="bilinear",
        align_corners=False,
    )
    square_map = x_sq[0, 0].cpu().numpy()
    return square_map, pooled


def mask_to_square_map(
    mask,
    time_bins=32,
    square_size=32,
):
    """
    Convert a real SRE element-wise mask [C, W] into a square occupancy map.

    This uses the same square-mapping logic as residual_to_square_map:
      1) temporal pooling: [C, W] -> [C, time_bins]
      2) channel-axis interpolation: [C, time_bins] -> [square_size, square_size]

    The pooled value in each bin is the visible/observed ratio:
      1 = mostly visible, 0 = mostly masked.

    Parameters
    ----------
    mask : np.ndarray
        Shape [C, W], values 0/1.
    time_bins : int
        Number of coarse temporal bins.
    square_size : int
        Final square size.

    Returns
    -------
    square_mask : np.ndarray
        Shape [square_size, square_size], values in [0, 1].
    pooled_mask : np.ndarray
        Shape [C, time_bins], values in [0, 1].
    """
    mask = np.asarray(mask, dtype=np.float32)
    C, W = mask.shape

    edges = np.linspace(0, W, time_bins + 1, dtype=int)
    pooled = np.zeros((C, time_bins), dtype=np.float32)

    for j in range(time_bins):
        seg = mask[:, edges[j]:edges[j + 1]]
        if seg.shape[1] == 0:
            continue
        pooled[:, j] = np.mean(seg, axis=1)

    x = torch.from_numpy(pooled).unsqueeze(0).unsqueeze(0)  # [1,1,C,time_bins]
    x_sq = F.interpolate(
        x,
        size=(square_size, square_size),
        mode="bilinear",
        align_corners=False,
    )
    square_mask = x_sq[0, 0].cpu().numpy()
    square_mask = np.clip(square_mask, 0.0, 1.0)

    return square_mask, pooled


def square_masked_view_from_maps(
    square_residual,
    square_mask,
    masked_value=np.nan,
    min_visible=0.5,
):
    """
    Build a square masked residual view from a square residual map and a real
    square mask occupancy map.

    This is for visualization only. It uses the real element-wise mask after
    the same square conversion as the residual map. Positions whose occupancy
    is below min_visible are rendered as masked.
    """
    square_residual = np.asarray(square_residual, dtype=np.float32)
    square_mask = np.asarray(square_mask, dtype=np.float32)
    if square_residual.shape != square_mask.shape:
        raise ValueError(
            f"square_residual shape {square_residual.shape} != square_mask shape {square_mask.shape}"
        )

    out = square_residual.copy()
    out[square_mask < min_visible] = masked_value
    return out


def masked_residual_to_square_map(
    residual,
    mask,
    time_bins=32,
    square_size=32,
    metric="absmean",
):
    """
    Convert masked residual [C, W] into a 1:1 square semantic map.
    Only observed entries contribute to each temporal bin.
    """
    residual = np.asarray(residual, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    C, W = residual.shape
    edges = np.linspace(0, W, time_bins + 1, dtype=int)
    pooled = np.full((C, time_bins), np.nan, dtype=np.float32)

    for j in range(time_bins):
        seg = residual[:, edges[j]:edges[j + 1]]
        seg_mask = mask[:, edges[j]:edges[j + 1]]
        if seg.shape[1] == 0:
            continue

        valid = seg_mask.sum(axis=1)

        if metric == "absmean":
            num = np.sum(np.abs(seg) * seg_mask, axis=1)
            val = num / np.maximum(valid, 1.0)
        elif metric == "rms":
            num = np.sum((seg ** 2) * seg_mask, axis=1)
            val = np.sqrt(num / np.maximum(valid, 1.0))
        elif metric == "signedmean":
            num = np.sum(seg * seg_mask, axis=1)
            val = num / np.maximum(valid, 1.0)
        elif metric == "absmax":
            val = np.full(C, np.nan, dtype=np.float32)
            for c in range(C):
                observed = np.abs(seg[c][seg_mask[c] > 0.5])
                if observed.size > 0:
                    val[c] = np.max(observed)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        val[valid < 1] = np.nan
        pooled[:, j] = val

    valid_map = np.isfinite(pooled).astype(np.float32)
    pooled_filled = np.nan_to_num(pooled, nan=0.0)

    x_val = torch.from_numpy(pooled_filled).unsqueeze(0).unsqueeze(0)
    x_msk = torch.from_numpy(valid_map).unsqueeze(0).unsqueeze(0)

    val_sq = F.interpolate(
        x_val,
        size=(square_size, square_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()

    msk_sq = F.interpolate(
        x_msk,
        size=(square_size, square_size),
        mode="nearest",
    )[0, 0].cpu().numpy()

    val_sq[msk_sq < 0.5] = np.nan
    return val_sq, pooled




def square_map_vrange(arrs, q_low=1, q_high=99):
    """Shared color range for a list of square maps."""
    vals = []
    for arr in arrs:
        arr = np.asarray(arr)
        if np.isfinite(arr).any():
            vals.append(arr[np.isfinite(arr)])

    if len(vals) == 0:
        return 0.0, 1.0

    vals = np.concatenate(vals)
    vmin = np.percentile(vals, q_low)
    vmax = np.percentile(vals, q_high)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

# The plotting functions below intentionally remove titles, axis labels,
# tick labels, legends, and colorbars by default. This makes the exported
# images easier to use as clean visual elements in a paper/PPT figure.


def _save_fig_clean(fig, save_path: Path, pad: float = 0.02):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=pad)
    plt.savefig(save_path, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)


def _turn_off_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_residual_heatmap(
    arr,
    save_path: Path,
    title: str = "",
    use_abs: bool = False,
    show_colorbar: bool = False,
):
    """Save a text-free strip residual heatmap."""
    arr = np.asarray(arr)

    if use_abs:
        plot_arr = np.abs(arr)
        cmap = "magma"
        vmin, vmax = robust_vlim_positive(plot_arr, 1, 99)
    else:
        plot_arr = arr
        cmap = "coolwarm"
        vmin, vmax = robust_vlim_signed(plot_arr, 99)

    fig, ax = plt.subplots(figsize=(4.8, 1.6), dpi=300)
    im = ax.imshow(
        plot_arr,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    _turn_off_axis(ax)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.ax.tick_params(labelsize=0, length=0)
        cbar.ax.set_yticklabels([])

    _save_fig_clean(fig, save_path)


def save_masked_heatmap(residual, mask, save_path: Path, title: str = ""):
    """Save a text-free strip masked residual heatmap."""
    residual = np.asarray(residual)
    mask = np.asarray(mask)

    arr = residual.copy()
    arr[mask < 0.5] = np.nan

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#D9D9D9")

    vmin, vmax = robust_vlim_signed(residual, 99)

    fig, ax = plt.subplots(figsize=(4.8, 1.6), dpi=300)
    im = ax.imshow(
        arr,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    _turn_off_axis(ax)
    _save_fig_clean(fig, save_path)


def save_square_residual_map(
    square_map,
    save_path: Path,
    title="",
    cmap="magma",
    vmin=None,
    vmax=None,
    show_axis: bool = False,
    show_colorbar: bool = False,
):
    """Save a clean 1:1 square residual semantic map without nearby text."""
    square_map = np.asarray(square_map)

    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#D9D9D9")

    if vmin is None or vmax is None:
        vmin, vmax = square_map_vrange([square_map], q_low=1, q_high=99)

    im = ax.imshow(
        square_map,
        cmap=cmap_obj,
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    if show_axis:
        ax.tick_params(axis="both", labelsize=0, length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        _turn_off_axis(ax)

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=0, length=0)
        cbar.ax.set_yticklabels([])

    _save_fig_clean(fig, save_path)


def save_square_mask_map(
    square_mask,
    save_path: Path,
    title="",
    cmap="Greys",
    vmin=0.0,
    vmax=1.0,
    show_axis: bool = False,
    show_colorbar: bool = False,
):
    """Save a clean 1:1 square real-mask occupancy heatmap."""
    save_square_residual_map(
        square_mask,
        save_path=save_path,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_axis=show_axis,
        show_colorbar=show_colorbar,
    )


def save_square_masked_view_map(
    square_view,
    save_path: Path,
    title="",
    cmap="magma",
    vmin=None,
    vmax=None,
    show_axis: bool = False,
    show_colorbar: bool = False,
):
    """Save a clean 1:1 square masked residual view derived from the real mask."""
    save_square_residual_map(
        square_view,
        save_path=save_path,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_axis=show_axis,
        show_colorbar=show_colorbar,
    )

def save_evidence_curve(
    values,
    save_path: Path,
    selected_idx: int,
    ylabel: str = "",
    title: str = "",
    high_quantile: float = 0.90,
):
    """Save a text-free evidence curve for direct use as a figure element."""
    values = np.asarray(values).reshape(-1)
    x = np.arange(1, len(values) + 1)
    thr = np.quantile(values, high_quantile)

    fig, ax = plt.subplots(figsize=(5.6, 2.0), dpi=300)
    ax.plot(x, values, linewidth=1.15, color="#4A4A4A", zorder=1)

    low_mask = values < thr
    high_mask = values >= thr

    ax.scatter(x[low_mask], values[low_mask], s=18, color="#2CA02C", zorder=2)
    ax.scatter(x[high_mask], values[high_mask], s=24, color="#D62728", zorder=3)
    ax.scatter(
        [selected_idx + 1],
        [values[selected_idx]],
        s=72,
        facecolor="none",
        edgecolor="#000000",
        linewidth=1.2,
        zorder=4,
    )
    ax.axvline(selected_idx + 1, linestyle="--", linewidth=0.8, color="#7F7F7F", alpha=0.7)

    # Keep the curve shape only; remove nearby text and tick labels.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    _save_fig_clean(fig, save_path, pad=0.05)


def save_sre_overview(
    cache,
    selected_idx: int,
    save_path: Path,
    use_abs_residual: bool = False,
    square_size: int = 32,
    time_bins: int = 32,
    square_metric: str = "absmean",
):
    """
    Save a clean SRE overview figure:
        real residual map | real mask A map | real mask B map | u1 curve

    No titles, axis labels, tick labels, legends, or colorbars are included.
    The mask maps are converted from the actual SRE masks using the same
    time-binning and channel-axis interpolation as the residual map.
    """
    residual = cache["residual_windows"].numpy()[0, selected_idx]
    mask_a = cache["mask_a"].numpy()[0, selected_idx]
    mask_b = cache["mask_b"].numpy()[0, selected_idx]
    u1 = cache["u1"].numpy()[0]

    metric = "absmean" if use_abs_residual else square_metric

    sq_residual, _ = residual_to_square_map(
        residual,
        time_bins=time_bins,
        square_size=square_size,
        metric=metric,
    )
    sq_mask_a, _ = mask_to_square_map(
        mask_a,
        time_bins=time_bins,
        square_size=square_size,
    )
    sq_mask_b, _ = mask_to_square_map(
        mask_b,
        time_bins=time_bins,
        square_size=square_size,
    )

    cmap_res = plt.get_cmap("magma").copy()
    cmap_res.set_bad(color="#D9D9D9")
    vmin_res, vmax_res = square_map_vrange([sq_residual], q_low=1, q_high=99)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(9.6, 2.55),
        dpi=300,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 1.65]},
    )

    # Real residual semantic map
    axes[0].imshow(
        sq_residual,
        aspect="equal",
        interpolation="nearest",
        cmap=cmap_res,
        vmin=vmin_res,
        vmax=vmax_res,
    )
    _turn_off_axis(axes[0])

    # Real Mask A/B occupancy maps. Greys_r: visible area is darker, masked area is lighter.
    for ax, arr in zip(axes[1:3], [sq_mask_a, sq_mask_b]):
        ax.imshow(
            arr,
            aspect="equal",
            interpolation="nearest",
            cmap="Greys_r",
            vmin=0.0,
            vmax=1.0,
        )
        _turn_off_axis(ax)

    values = np.asarray(u1).reshape(-1)
    x = np.arange(1, len(values) + 1)
    thr = np.quantile(values, 0.90)
    low_mask = values < thr
    high_mask = values >= thr

    axes[3].plot(x, values, linewidth=1.05, color="#4A4A4A")
    axes[3].scatter(x[low_mask], values[low_mask], s=12, color="#2CA02C")
    axes[3].scatter(x[high_mask], values[high_mask], s=16, color="#D62728")
    axes[3].scatter(
        [selected_idx + 1],
        [values[selected_idx]],
        s=52,
        facecolor="none",
        edgecolor="#000000",
        linewidth=1.1,
        zorder=5,
    )
    axes[3].axvline(selected_idx + 1, linestyle="--", linewidth=0.7, color="#7F7F7F")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    axes[3].set_xlabel("")
    axes[3].set_ylabel("")
    axes[3].set_title("")
    axes[3].spines["top"].set_visible(False)
    axes[3].spines["right"].set_visible(False)
    axes[3].spines["left"].set_linewidth(0.6)
    axes[3].spines["bottom"].set_linewidth(0.6)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.05, w_pad=0.18)
    plt.savefig(save_path, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)


def save_sre_overview_full(
    cache,
    selected_idx: int,
    save_path: Path,
    use_abs_residual: bool = False,
    square_size: int = 32,
    time_bins: int = 32,
    square_metric: str = "absmean",
):
    """
    Save a fuller clean SRE overview figure:
        residual map | mask A | masked view A | mask B | masked view B | u1 curve

    This is useful if you want to show both the real mask maps and the resulting
    masked residual semantic views.
    """
    residual = cache["residual_windows"].numpy()[0, selected_idx]
    mask_a = cache["mask_a"].numpy()[0, selected_idx]
    mask_b = cache["mask_b"].numpy()[0, selected_idx]
    u1 = cache["u1"].numpy()[0]

    metric = "absmean" if use_abs_residual else square_metric

    sq_residual, _ = residual_to_square_map(
        residual,
        time_bins=time_bins,
        square_size=square_size,
        metric=metric,
    )
    sq_mask_a, _ = mask_to_square_map(mask_a, time_bins=time_bins, square_size=square_size)
    sq_mask_b, _ = mask_to_square_map(mask_b, time_bins=time_bins, square_size=square_size)
    sq_view_a = square_masked_view_from_maps(sq_residual, sq_mask_a, masked_value=np.nan, min_visible=0.5)
    sq_view_b = square_masked_view_from_maps(sq_residual, sq_mask_b, masked_value=np.nan, min_visible=0.5)

    cmap_res = plt.get_cmap("magma").copy()
    cmap_res.set_bad(color="#D9D9D9")
    vmin_res, vmax_res = square_map_vrange([sq_residual], q_low=1, q_high=99)

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(13.2, 2.55),
        dpi=300,
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.75]},
    )

    axes[0].imshow(sq_residual, aspect="equal", interpolation="nearest", cmap=cmap_res, vmin=vmin_res, vmax=vmax_res)
    _turn_off_axis(axes[0])
    axes[1].imshow(sq_mask_a, aspect="equal", interpolation="nearest", cmap="Greys_r", vmin=0.0, vmax=1.0)
    _turn_off_axis(axes[1])
    axes[2].imshow(sq_view_a, aspect="equal", interpolation="nearest", cmap=cmap_res, vmin=vmin_res, vmax=vmax_res)
    _turn_off_axis(axes[2])
    axes[3].imshow(sq_mask_b, aspect="equal", interpolation="nearest", cmap="Greys_r", vmin=0.0, vmax=1.0)
    _turn_off_axis(axes[3])
    axes[4].imshow(sq_view_b, aspect="equal", interpolation="nearest", cmap=cmap_res, vmin=vmin_res, vmax=vmax_res)
    _turn_off_axis(axes[4])

    values = np.asarray(u1).reshape(-1)
    x = np.arange(1, len(values) + 1)
    thr = np.quantile(values, 0.90)
    low_mask = values < thr
    high_mask = values >= thr
    axes[5].plot(x, values, linewidth=1.05, color="#4A4A4A")
    axes[5].scatter(x[low_mask], values[low_mask], s=12, color="#2CA02C")
    axes[5].scatter(x[high_mask], values[high_mask], s=16, color="#D62728")
    axes[5].scatter([selected_idx + 1], [values[selected_idx]], s=52, facecolor="none", edgecolor="#000000", linewidth=1.1, zorder=5)
    axes[5].axvline(selected_idx + 1, linestyle="--", linewidth=0.7, color="#7F7F7F")
    axes[5].set_xticks([])
    axes[5].set_yticks([])
    axes[5].set_xlabel("")
    axes[5].set_ylabel("")
    axes[5].set_title("")
    axes[5].spines["top"].set_visible(False)
    axes[5].spines["right"].set_visible(False)
    axes[5].spines["left"].set_linewidth(0.6)
    axes[5].spines["bottom"].set_linewidth(0.6)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(pad=0.05, w_pad=0.18)
    plt.savefig(save_path, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)

def save_raw_recon_residual_overview(cache, selected_idx: int, save_path: Path):
    """Save a text-free raw/reconstruction/residual waveform sanity-check figure."""
    xw = cache["x_windows"].numpy()[0, selected_idx]
    xhat = cache["x_hat_windows"].numpy()[0, selected_idx]
    residual = cache["residual_windows"].numpy()[0, selected_idx]

    ch_strength = np.abs(residual).mean(axis=1)
    ch = int(np.argmax(ch_strength))
    t = np.arange(xw.shape[-1])

    fig, axes = plt.subplots(3, 1, figsize=(6.0, 3.0), dpi=300, sharex=True)
    axes[0].plot(t, xw[ch], linewidth=1.0, color="#1F77B4")
    axes[1].plot(t, xhat[ch], linewidth=1.0, color="#FF7F0E")
    axes[2].plot(t, residual[ch], linewidth=1.0, color="#D62728")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_linewidth(0.6)

    _save_fig_clean(fig, save_path, pad=0.05)


def save_npz(cache, selected_idx: int, save_path: Path, sample_path: str):
    np.savez_compressed(
        save_path,
        sample_path=np.array(sample_path),
        selected_window_idx=np.array(selected_idx, dtype=np.int64),

        x_windows=cache["x_windows"].numpy()[0],
        x_hat_windows=cache["x_hat_windows"].numpy()[0],
        residual_windows=cache["residual_windows"].numpy()[0],

        mask_a=cache["mask_a"].numpy()[0],
        mask_b=cache["mask_b"].numpy()[0],
        view_a=cache["view_a"].numpy()[0],
        view_b=cache["view_b"].numpy()[0],

        e_a=cache["e_a"].numpy()[0],
        e_b=cache["e_b"].numpy()[0],
        pred_a=cache["pred_a"].numpy()[0],
        pred_b=cache["pred_b"].numpy()[0],
        e=cache["e"].numpy()[0],

        rec_window=cache["rec_window"].numpy()[0],
        u1=cache["u1"].numpy()[0],
        u2=cache["u2"].numpy()[0],
        window_energy=cache["window_energy"].numpy()[0],
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run directory containing config.yaml, checkpoints/, normalizer.npz.",
    )
    p.add_argument(
        "--sample_path",
        type=str,
        required=True,
        help="Path to one .npy sample file to visualize.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="analysis/visualization",
        help="Output directory for npz and figures.",
    )
    p.add_argument(
        "--ckpt_name",
        type=str,
        default="ema.pt",
        help="Checkpoint name. Default: ema.pt.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu.",
    )
    p.add_argument(
        "--select_window",
        type=str,
        default="top_u1",
        choices=["top_u1", "top_u2", "top_rec", "top_residual_abs", "middle"],
    )
    p.add_argument(
        "--window_idx",
        type=int,
        default=None,
        help="Explicit 0-based window index. Overrides --select_window.",
    )
    p.add_argument(
        "--random_training_mask",
        action="store_true",
        help="Use random SRE masks instead of deterministic inference mask.",
    )
    p.add_argument(
        "--use_abs_residual",
        action="store_true",
        help="Use absmean residual maps in the overview figure.",
    )
    p.add_argument(
        "--square_size",
        type=int,
        default=32,
        help="Square residual map size. Default: 32.",
    )
    p.add_argument(
        "--time_bins",
        type=int,
        default=32,
        help="Temporal bins before square interpolation. Default: 32.",
    )
    p.add_argument(
        "--square_metric",
        type=str,
        default="absmean",
        choices=["absmean", "rms", "signedmean", "absmax"],
        help="Residual metric for square semantic maps.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    sample_path = Path(args.sample_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not sample_path.exists():
        raise FileNotFoundError(f"sample_path does not exist: {sample_path}")

    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found: {cfg_path}")

    cfg = load_config(cfg_path)
    norm_stats = load_normalizer(run_dir)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    method = build_method(cfg)
    ckpt_path = find_checkpoint(run_dir, args.ckpt_name)

    print("=" * 80)
    print("[SRE export]")
    print(f"run_dir      : {run_dir}")
    print(f"config       : {cfg_path}")
    print(f"checkpoint   : {ckpt_path}")
    print(f"ckpt size    : {ckpt_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"sample       : {sample_path}")
    print(f"device       : {device}")
    print("=" * 80)

    try:
        load_checkpoint(ckpt_path, method, map_location=device)
    except RuntimeError as e:
        msg = str(e)
        if "failed finding central directory" in msg or "PytorchStreamReader" in msg:
            raise RuntimeError(
                f"Checkpoint file appears corrupted or incomplete: {ckpt_path}\n"
                f"File size: {ckpt_path.stat().st_size / 1024 / 1024:.2f} MB\n"
                "Try --ckpt_name best.pt or --ckpt_name last.pt, or re-copy the checkpoint."
            ) from e
        raise

    method.to(device)
    method.eval()

    x_np = load_one_sample(sample_path, cfg, norm_stats)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device)
    print(f"sample shape : {tuple(x_np.shape)}")

    cache = extract_sre_intermediates(
        method,
        x,
        training_mask=bool(args.random_training_mask),
    )

    selected_idx = select_window(
        cache,
        mode=args.select_window,
        explicit_idx=args.window_idx,
    )

    residual = cache["residual_windows"].numpy()[0, selected_idx]
    mask_a = cache["mask_a"].numpy()[0, selected_idx]
    mask_b = cache["mask_b"].numpy()[0, selected_idx]
    u1 = cache["u1"].numpy()[0]
    u2 = cache["u2"].numpy()[0]
    rec = cache["rec_window"].numpy()[0]

    save_npz(
        cache,
        selected_idx=selected_idx,
        save_path=out_dir / "sre_intermediates.npz",
        sample_path=str(sample_path),
    )

    # Original strip heatmaps, kept for reference.
    save_residual_heatmap(
        residual,
        out_dir / f"residual_heatmap_win{selected_idx:03d}.png",
        title=f"Residual heatmap, window {selected_idx}",
        use_abs=False,
    )

    save_residual_heatmap(
        residual,
        out_dir / f"abs_residual_heatmap_win{selected_idx:03d}.png",
        title=f"|Residual| heatmap, window {selected_idx}",
        use_abs=True,
    )

    save_masked_heatmap(
        residual,
        mask_a,
        out_dir / f"masked_view_A_win{selected_idx:03d}.png",
        title=f"Masked residual view A, window {selected_idx}",
    )

    save_masked_heatmap(
        residual,
        mask_b,
        out_dir / f"masked_view_B_win{selected_idx:03d}.png",
        title=f"Masked residual view B, window {selected_idx}",
    )

    # New 1:1 square maps derived from the real residual and real SRE masks.
    # The mask maps use the same time-binning and channel-axis interpolation as
    # the residual map. This avoids the misleading visualization-only coarse mask.
    sq_residual, pooled_residual = residual_to_square_map(
        residual,
        time_bins=int(args.time_bins),
        square_size=int(args.square_size),
        metric=str(args.square_metric),
    )
    sq_mask_a, pooled_mask_a = mask_to_square_map(
        mask_a,
        time_bins=int(args.time_bins),
        square_size=int(args.square_size),
    )
    sq_mask_b, pooled_mask_b = mask_to_square_map(
        mask_b,
        time_bins=int(args.time_bins),
        square_size=int(args.square_size),
    )
    sq_view_a = square_masked_view_from_maps(
        sq_residual,
        sq_mask_a,
        masked_value=np.nan,
        min_visible=0.5,
    )
    sq_view_b = square_masked_view_from_maps(
        sq_residual,
        sq_mask_b,
        masked_value=np.nan,
        min_visible=0.5,
    )

    vmin_sq, vmax_sq = square_map_vrange([sq_residual], q_low=1, q_high=99)

    save_square_residual_map(
        sq_residual,
        out_dir / f"square_residual_map_win{selected_idx:03d}.png",
        title=f"Residual semantic map, window {selected_idx}",
        cmap="magma",
        vmin=vmin_sq,
        vmax=vmax_sq,
    )

    save_square_mask_map(
        sq_mask_a,
        out_dir / f"square_mask_A_map_win{selected_idx:03d}.png",
        title=f"Real mask A map, window {selected_idx}",
        cmap="Greys_r",
        vmin=0.0,
        vmax=1.0,
    )

    save_square_mask_map(
        sq_mask_b,
        out_dir / f"square_mask_B_map_win{selected_idx:03d}.png",
        title=f"Real mask B map, window {selected_idx}",
        cmap="Greys_r",
        vmin=0.0,
        vmax=1.0,
    )

    save_square_masked_view_map(
        sq_view_a,
        out_dir / f"square_masked_view_A_win{selected_idx:03d}.png",
        title=f"Real masked residual view A, window {selected_idx}",
        cmap="magma",
        vmin=vmin_sq,
        vmax=vmax_sq,
    )

    save_square_masked_view_map(
        sq_view_b,
        out_dir / f"square_masked_view_B_win{selected_idx:03d}.png",
        title=f"Real masked residual view B, window {selected_idx}",
        cmap="magma",
        vmin=vmin_sq,
        vmax=vmax_sq,
    )

    # Save pooled maps as npy for optional later use.
    np.save(out_dir / f"pooled_residual_map_win{selected_idx:03d}.npy", pooled_residual)
    np.save(out_dir / f"pooled_mask_A_map_win{selected_idx:03d}.npy", pooled_mask_a)
    np.save(out_dir / f"pooled_mask_B_map_win{selected_idx:03d}.npy", pooled_mask_b)

    # Curves.
    save_evidence_curve(
        u1,
        out_dir / "sre_u1_curve.png",
        selected_idx=selected_idx,
        ylabel=r"$u_1^{(m)}$",
        title="Window-wise SRE evidence",
    )

    save_evidence_curve(
        u2,
        out_dir / "ctm_u2_curve.png",
        selected_idx=selected_idx,
        ylabel=r"$u_2^{(m)}$",
        title="Window-wise CTM evidence",
    )

    save_evidence_curve(
        rec,
        out_dir / "rec_energy_curve.png",
        selected_idx=selected_idx,
        ylabel="Rec. energy",
        title="Window-wise reconstruction energy",
    )

    # Compact overview with 1:1 square maps.
    save_sre_overview(
        cache,
        selected_idx,
        out_dir / f"sre_overview_win{selected_idx:03d}.png",
        use_abs_residual=bool(args.use_abs_residual),
        square_size=int(args.square_size),
        time_bins=int(args.time_bins),
        square_metric=str(args.square_metric),
    )

    save_sre_overview_full(
        cache,
        selected_idx,
        out_dir / f"sre_overview_full_win{selected_idx:03d}.png",
        use_abs_residual=bool(args.use_abs_residual),
        square_size=int(args.square_size),
        time_bins=int(args.time_bins),
        square_metric=str(args.square_metric),
    )

    # Raw / reconstruction / residual waveform overview for sanity check.
    save_raw_recon_residual_overview(
        cache,
        selected_idx,
        out_dir / f"raw_recon_residual_win{selected_idx:03d}.png",
    )

    print("[DONE]")
    print(f"Output dir          : {out_dir}")
    print(f"Selected window idx : {selected_idx}")
    print(f"u1[selected]        : {float(u1[selected_idx]):.6f}")
    print(f"u2[selected]        : {float(u2[selected_idx]):.6f}")
    print(f"rec[selected]       : {float(rec[selected_idx]):.6f}")
    print("Saved:")
    print("  - sre_intermediates.npz")
    print("  - original strip residual heatmaps")
    print("  - square residual semantic map")
    print("  - square real mask A/B occupancy maps")
    print("  - square real masked residual views A/B")
    print("  - u1/u2/rec curves")
    print("  - compact SRE overview figure")


if __name__ == "__main__":
    main()
