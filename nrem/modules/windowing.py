from __future__ import annotations

import torch
import torch.nn.functional as F


def windowize(x: torch.Tensor, window_size: int, stride: int) -> tuple[torch.Tensor, int]:
    """x [B,C,T] -> windows [B,M,C,W]. Pads at the end if needed."""
    b, c, t = x.shape
    w = int(window_size)
    s = int(stride)
    if t < w:
        x = F.pad(x, (0, w - t))
        t = w
    # Ignore the incomplete tail rather than padding a mostly artificial window.
    # This matches the common EoL baseline protocol and keeps evaluation deterministic.
    # unfold returns [B,C,M,W]
    win = x.unfold(dimension=-1, size=w, step=s).permute(0, 2, 1, 3).contiguous()
    return win, x.shape[-1]
