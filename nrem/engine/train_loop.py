"""End-to-end joint training loop.

All losses (rec, view, dyn, var, cov) are active from epoch 0.
No two-stage warmup, no epoch-dependent loss gating.
"""
from __future__ import annotations

from typing import Dict, List
import copy
import numpy as np
import torch
from tqdm import tqdm


def _clone_state(model) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _update_ema(ema_state: dict, model, decay: float) -> dict:
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k not in ema_state:
                ema_state[k] = v.detach().clone()
                continue
            if torch.is_floating_point(v):
                ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
            else:
                ema_state[k].copy_(v.detach())
    return ema_state


def train_epochs(method, loader, cfg: dict, device: str):
    tcfg      = cfg.get("train", {}) or {}
    epochs    = int(tcfg.get("epochs", 30))
    lr        = float(tcfg.get("lr", 5e-4))
    wd        = float(tcfg.get("weight_decay", 1e-5))
    grad_clip = float(tcfg.get("grad_clip", 5.0))
    use_ema   = bool(tcfg.get("use_ema", True))
    ema_decay = float(tcfg.get("ema_decay", 0.995))

    method.to(device)
    if hasattr(method, "before_training"):
        method.before_training(loader, device)
    if hasattr(method, "training_epoch"):
        custom_optimizers = (
            method.configure_optimizers(cfg) if hasattr(method, "configure_optimizers") else None
        )
        history: List[Dict] = []
        for ep in range(epochs):
            row = {"epoch": ep}
            row.update(method.training_epoch(loader, cfg, device, ep, custom_optimizers))
            row["ema_decay"] = 0.0
            print(row)
            history.append(row)
        return history, None, None
    optimizer = torch.optim.AdamW(method.parameters(), lr=lr, weight_decay=wd)
    ema_state = _clone_state(method) if use_ema else None
    history: List[Dict] = []

    for ep in range(epochs):
        if hasattr(method, "set_epoch"):
            method.set_epoch(ep)
        method.train()
        meters: Dict[str, List[float]] = {"loss": []}
        pbar = tqdm(loader, desc=f"epoch {ep+1}/{epochs}", leave=False)

        for batch in pbar:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            loss, logs = method.training_step(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(method.parameters(), grad_clip)
            optimizer.step()
            if use_ema:
                _update_ema(ema_state, method, ema_decay)
            meters["loss"].append(float(loss.detach().cpu()))
            for k, v in logs.items():
                meters.setdefault(k, []).append(float(v))
            pbar.set_postfix({k: f"{np.mean(v):.4f}"
                              for k, v in meters.items() if v})

        row = {"epoch": ep}
        row.update({k: float(np.mean(v)) for k, v in meters.items() if v})
        row["ema_decay"] = ema_decay if use_ema else 0.0
        print(row)
        history.append(row)

    return history, optimizer, ema_state
