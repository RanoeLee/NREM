"""Training runner and evaluation entry-points.

Key change vs. the original: the MR-RESA reference is built from ALL
three energy types (rec, u1, u12) in one pass, stored as sorted per-sample
arrays.  The scorer picks the appropriate key at inference time based on
the cfg.nrem.resa_energy setting.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import save_yaml, load_config
from ..data import FolderTimeSeriesDataset, pad_collate_fn, estimate_norm_stats
from ..methods import build_method
from ..engine.train_loop import train_epochs
from ..engine.eval_loop import evaluate_scores
from ..engine.checkpoint import save_checkpoint, load_checkpoint
from ..modules.rea import build_reference, save_reference, load_reference
from ..utils.io import write_json
from ..utils.metrics import compute_binary_metrics


# ── Dataset / loader helpers ──────────────────────────────────────────────────

def build_dataset(root: str, cfg: dict, norm_stats=None):
    dcfg = cfg.get("data", {}) or {}
    return FolderTimeSeriesDataset(
        root=root,
        exts=tuple(dcfg.get("exts", [".npy"])),
        recursive=bool(dcfg.get("recursive", True)),
        transpose_if_tall=bool(dcfg.get("transpose_if_tall", True)),
        max_len=int(dcfg.get("max_len", 48000)),
        strict_checks=bool(dcfg.get("strict_checks", True)),
        expected_channels=int(dcfg.get("channels",
                              (cfg.get("model") or {}).get("channels", 6))),
        npy_mmap_mode=dcfg.get("npy_mmap_mode", None),
        normalize=str(dcfg.get("normalize", "none")).lower(),
        norm_stats=norm_stats,
    )


def make_loader(ds, cfg: dict, shuffle: bool):
    tcfg = cfg.get("train", {}) or {}
    return DataLoader(
        ds,
        batch_size=int(tcfg.get("batch_size", 4)),
        shuffle=shuffle,
        num_workers=int(tcfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=pad_collate_fn,
    )


# ── MR-RESA reference construction ───────────────────────────────────────────

@torch.no_grad()
def collect_window_energies(method, loader, device: str) -> np.ndarray:
    """Run inference over training set; collect per-sample per-window energies.

    Returns a dict with keys 'rec', 'u1', 'u12', each an ndarray of
    shape [N_train, M_windows] (float32).  The runner saves all three so
    any score_mode / resa_energy combination can be used at inference
    without retraining.
    """
    method.to(device)
    method.eval()
    rec_all, u1_all, u2_all, u12_all = [], [], [], []

    for batch in tqdm(loader, desc="build_ref", leave=False):
        batch_d = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                   for k, v in batch.items()}
        out = method(batch_d, training=False)
        rec  = out.rec_window.detach().cpu().numpy().astype(np.float32)   # [B,M]
        u1   = out.u1_window.detach().cpu().numpy().astype(np.float32)
        u2   = out.u2_window.detach().cpu().numpy().astype(np.float32)
        u12  = (out.u1_window + out.u2_window).detach().cpu().numpy().astype(np.float32)
        rec_all.append(rec)
        u1_all.append(u1)
        u2_all.append(u2)
        u12_all.append(u12)

    def _cat(lst):
        return np.concatenate(lst, axis=0) if lst else np.empty((0, 0), np.float32)

    return {
        "rec": _cat(rec_all),
        "u1":  _cat(u1_all),
        "u2":  _cat(u2_all),
        "u12": _cat(u12_all),
    }


def prepare_reference(energies: dict) -> dict:
    """Build sorted per-sample reference arrays for all energy types."""
    ref = {}
    for key, arr in energies.items():
        if arr.ndim == 2 and arr.size > 0:
            ref[f"{key}_sorted"] = np.sort(arr.astype(np.float32), axis=1)
    return ref


# ── Training entry-point ──────────────────────────────────────────────────────

def train_one(
    config_path, cfg: dict, train_dir, out_dir, seed: int, device: str
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalisation stats (estimated on raw training data)
    raw_cfg = json.loads(json.dumps(cfg))
    raw_cfg.setdefault("data", {})["normalize"] = "none"
    raw_ds = build_dataset(str(train_dir), raw_cfg, norm_stats=None)
    norm = None
    if str((cfg.get("data") or {}).get("normalize", "none")).lower() != "none":
        dcfg = cfg.get("data") or {}
        norm = estimate_norm_stats(
            raw_ds,
            max_samples=dcfg.get("norm_max_samples", None),
            robust_max_points_per_channel=dcfg.get(
                "robust_norm_max_points_per_channel", 500_000),
            robust_seed=dcfg.get("robust_norm_seed", 2025),
        )
        np.savez_compressed(out_dir / "normalizer.npz", **norm)

    ds     = build_dataset(str(train_dir), cfg, norm_stats=norm)
    loader = make_loader(ds, cfg, shuffle=True)

    method = build_method(cfg)
    history, opt, ema_state = train_epochs(method, loader, cfg, device)

    ckpt_dir = out_dir / "checkpoints"
    save_checkpoint(ckpt_dir / "last.pt", method, opt,
                    epoch=len(history) - 1, cfg=cfg)
    save_checkpoint(ckpt_dir / "best.pt", method, opt,
                    epoch=len(history) - 1, cfg=cfg)
    if ema_state is not None:
        save_checkpoint(ckpt_dir / "ema.pt", method, None,
                        epoch=len(history) - 1, cfg=cfg,
                        state_dict=ema_state)

    save_yaml(out_dir / "config.yaml", cfg)
    write_json(out_dir / "history.json", history)

    # Build MR-RESA reference (always; needed for mr_resa score_mode)
    ref_loader = make_loader(ds, cfg, shuffle=False)
    # Use EMA weights for reference so it matches the inference checkpoint
    restore = None
    if ema_state is not None:
        restore = {k: v.detach().clone() for k, v in method.state_dict().items()}
        method.load_state_dict(ema_state, strict=False)
    energies = collect_window_energies(method, ref_loader, device)
    if restore is not None:
        method.load_state_dict(restore, strict=False)

    ref = prepare_reference(energies)
    save_reference(out_dir / "resa_reference.npz", ref)

    write_json(out_dir / "summary.json", {
        "config_path": str(config_path),
        "train_dir":   str(train_dir),
        "seed":        int(seed),
        "n_train":     len(ds),
        "m_windows":   int(energies["u12"].shape[1]) if energies["u12"].ndim == 2 else 0,
        "ema_available": ema_state is not None,
        "ref_keys":    list(ref.keys()),
    })
    return out_dir


# ── Evaluation entry-point ────────────────────────────────────────────────────

def _rebuild_reference(
    run_dir: Path,
    cfg: dict,
    method,
    device: str,
    norm,
    output_path: Path | None = None,
) -> Path:
    summary = json.loads((run_dir / "summary.json").read_text())
    train_dir = summary.get("train_dir")
    if not train_dir:
        raise RuntimeError(f"Cannot rebuild reference: train_dir missing in {run_dir}/summary.json")
    ds      = build_dataset(train_dir, cfg, norm_stats=norm)
    loader  = make_loader(ds, cfg, shuffle=False)
    energies = collect_window_energies(method, loader, device)
    ref      = prepare_reference(energies)
    ref_path = Path(output_path) if output_path is not None else run_dir / "resa_reference.npz"
    save_reference(ref_path, ref)
    return ref_path


def finalize_evaluation_metrics(
    metrics: dict,
    *,
    protocol_provenance: dict | None = None,
) -> dict:
    finalized = dict(metrics)
    if protocol_provenance is None:
        return finalized
    positive = int(finalized.get("positive", 0) or 0)
    negative = int(finalized.get("negative", 0) or 0)
    total = int(finalized.get("n", positive + negative) or 0)
    expected_positive = int(protocol_provenance["eval_ng"])
    expected_negative = int(protocol_provenance["eval_ok"])
    expected_total = expected_positive + expected_negative
    if (
        positive != expected_positive
        or negative != expected_negative
        or total != expected_total
    ):
        raise RuntimeError(
            "Evaluation metric label-count mismatch: "
            f"expected n={expected_total}, positive={expected_positive}, "
            f"negative={expected_negative}; got n={total}, positive={positive}, "
            f"negative={negative}"
        )
    finalized.update(protocol_provenance)
    return finalized


def evaluate_run(
    run_dir,
    eval_root,
    cfg: dict | None = None,
    ckpt_name: str = "ema.pt",
    device: str = "cuda",
    score_mode: str | None = None,
    reference_path: str | Path | None = None,
    result_tag: str | None = None,
    protocol_provenance: dict | None = None,
):
    run_dir = Path(run_dir)
    if cfg is None:
        cfg = load_config(run_dir / "config.yaml")

    norm = None
    norm_path = run_dir / "normalizer.npz"
    if norm_path.exists():
        z    = np.load(norm_path)
        norm = {k: z[k] for k in z.files}

    ds     = build_dataset(str(eval_root), cfg, norm_stats=norm)
    loader = make_loader(ds, cfg, shuffle=False)

    method = build_method(cfg)
    # Checkpoint fallback order
    ckpt_path = run_dir / "checkpoints" / ckpt_name
    if not ckpt_path.exists():
        for fb in [run_dir / ckpt_name,
                   run_dir / "checkpoints" / "ema.pt",
                   run_dir / "checkpoints" / "best.pt",
                   run_dir / "checkpoints" / "last.pt"]:
            if fb.exists():
                ckpt_path = fb
                break
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    load_checkpoint(ckpt_path, method, map_location=device)
    if hasattr(method, "load_artifacts"):
        method.load_artifacts(run_dir)

    smode    = str(score_mode or (cfg.get("nrem") or {}).get("score_mode", "rea")).lower()
    resa_ref = None
    resolved_ref_path = None
    if smode in {"rea", "mr_resa"}:
        if reference_path is None:
            ref_path = run_dir / "resa_reference.npz"
        else:
            ref_path = Path(reference_path).expanduser()
            if not ref_path.is_absolute():
                ref_path = run_dir / ref_path
            ref_path = ref_path.resolve()
        if not ref_path.exists():
            ref_path = _rebuild_reference(
                run_dir, cfg, method, device, norm, output_path=ref_path
            )
        resa_ref = load_reference(ref_path, device=device)
        resolved_ref_path = str(ref_path)

    out     = evaluate_scores(method, loader, device,
                              score_mode=smode, resa_ref=resa_ref)
    metrics = compute_binary_metrics(out["labels"], out["scores"])
    metrics["score_mode"] = smode
    if resa_ref is not None:
        metrics["resa_ref_samples"] = int(resa_ref.get("n_samples", 0))
        metrics["resa_ref_windows"] = int(resa_ref.get("m_windows", 0))
        metrics["resa_reference_path"] = resolved_ref_path
    metrics = finalize_evaluation_metrics(
        metrics, protocol_provenance=protocol_provenance
    )

    eval_name = Path(eval_root).name
    tag_suffix = f"_{result_tag}" if result_tag else ""
    save_dir  = run_dir / f"eval_{eval_name}_{Path(ckpt_name).stem}_{smode}{tag_suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_dir / "scores.npz",
        scores=out["scores"], labels=out["labels"],
        paths=np.array(out["paths"], dtype=object),
    )
    write_json(save_dir / "metrics.json", metrics)
    return metrics
