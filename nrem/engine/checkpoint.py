from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path: str | Path, model, optimizer=None, epoch: int = 0, metrics: dict | None = None, cfg: dict | None = None, state_dict=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        'epoch': int(epoch),
        'model_state_dict': state_dict if state_dict is not None else model.state_dict(),
        'metrics': metrics or {},
        'config': cfg or {},
    }
    if optimizer is not None:
        obj['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(obj, path)


def load_checkpoint(path: str | Path, model, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    return ckpt
