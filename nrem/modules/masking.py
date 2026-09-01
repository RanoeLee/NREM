from __future__ import annotations

import torch


def complementary_element_mask(x: torch.Tensor, mask_ratio: float = 0.5, training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Create complementary masks for window tensor [N, C, W]."""
    if training:
        ma = (torch.rand_like(x) > float(mask_ratio)).float()
    else:
        # deterministic alternating time mask, same for every batch element/channel
        n, c, w = x.shape
        base = torch.arange(w, device=x.device).view(1, 1, w)
        ma = ((base % 2) == 0).float().expand(n, c, w)
    mb = 1.0 - ma
    return ma, mb


def make_window_mask(batch: int, length: int, ratio: float, device: torch.device, training: bool = True) -> torch.Tensor:
    if training:
        mask = torch.rand(batch, length, device=device) < float(ratio)
        # ensure at least one masked position per sample
        if length > 0:
            none = ~mask.any(dim=1)
            if none.any():
                idx = torch.randint(0, length, (int(none.sum()),), device=device)
                mask[none, idx] = True
        return mask
    # deterministic 4-group mask in eval should be handled elsewhere
    mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
    mask[:, ::4] = True
    return mask
