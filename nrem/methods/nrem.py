"""NREMMethod: thin wrapper that adapts NREMModel to the engine interface."""
from __future__ import annotations

from typing import Dict, Tuple
import torch
import torch.nn as nn

from ..models.nrem import NREMModel


class NREMMethod(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg   = cfg
        self.model = NREMModel(cfg)

    # The engine calls set_epoch() at the start of every epoch.
    # With end-to-end joint training there is no epoch-dependent behaviour,
    # but the hook is kept for forward compatibility.
    def set_epoch(self, epoch: int) -> None:
        pass

    def forward(self, batch: Dict, training: bool = True):
        return self.model(batch["x"], training=training)

    def training_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(batch, training=True)
        assert out.loss is not None, "Loss is None during training_step"
        return out.loss, out.logs

    @torch.no_grad()
    def score_batch(
        self,
        batch: Dict,
        score_mode: str | None = None,
        resa_ref: dict | None = None,
    ) -> Tuple[torch.Tensor, Dict]:
        return self.model.score(batch["x"], score_mode=score_mode, resa_ref=resa_ref)
