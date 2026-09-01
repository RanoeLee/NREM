from __future__ import annotations

from typing import Dict, List
import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_scores(method, loader, device: str, score_mode: str | None = None, resa_ref: torch.Tensor | None = None):
    method.to(device)
    method.eval()
    scores: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    paths: List[str] = []
    for batch in tqdm(loader, desc='infer', leave=False):
        batch_dev = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        s, aux = method.score_batch(batch_dev, score_mode=score_mode, resa_ref=resa_ref)
        scores.append(s.detach().cpu().numpy())
        labels.append(batch['label'].detach().cpu().numpy())
        paths.extend(batch['path'])
    return {
        'scores': np.concatenate(scores, axis=0) if scores else np.array([]),
        'labels': np.concatenate(labels, axis=0) if labels else np.array([]),
        'paths': paths,
    }
