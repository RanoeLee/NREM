"""Backbone autoencoder models.

ConvWindowAE  — optional 1-D CNN encoder-decoder retained for comparison.
               for hidden=(32,64,128), latent=64, kernel=7.  Accepts flat
               [N, C*W] or 3-D [N, C, W]; returns output in the same form.

MLPWindowAE   — final paper backbone. Drop-in
               compatible; ~13 M parameters (intentionally overparameterised
               so the ablation fairly tests the effect of temporal structure).
"""
from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ─────────────────────────────────────────────────────────────────

def _gn(num_channels: int, max_groups: int = 8) -> nn.Module:
    """GroupNorm with a safe group count for small channel widths."""
    g = max(1, min(max_groups, num_channels))
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


# ── Conv backbone ────────────────────────────────────────────────────────────

class ConvWindowAE(nn.Module):
    """1-D CNN encoder-decoder for multichannel windows.

    The latent space has shape [N, latent_channels, W'] where
    W' = W / 2**len(hidden_channels), preserving a coarsened time axis.

    Both flat [N, C*W] and 3-D [N, C, W] inputs are accepted; output is
    returned in the same form.

    Parameters
    ----------
    channels : int
        Number of signal channels C.
    window_size : int
        Window length W.  Must satisfy W % 2**depth == 0.
    hidden_channels : tuple of int
        Channel widths at each encoder stage; each stage downsamples by 2.
    latent_channels : int
        Bottleneck channel width.
    kernel_size : int
        Convolution kernel size (odd).
    dropout : float
        Dropout probability after each activation (0 = disabled).
    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        hidden_channels: Tuple[int, ...] = (32, 64, 128),
        latent_channels: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.window_size = int(window_size)
        depth = len(hidden_channels)
        if self.window_size % (2 ** depth) != 0:
            raise ValueError(
                f"window_size={self.window_size} must be divisible by "
                f"2^depth={2**depth}. Adjust window_size or hidden_channels."
            )
        pad = kernel_size // 2

        # Encoder: stride-2 convolution at each stage
        enc: list[nn.Module] = []
        in_c = channels
        for h in hidden_channels:
            enc += [nn.Conv1d(in_c, h, kernel_size, stride=2, padding=pad),
                    _gn(h), nn.GELU()]
            if dropout > 0:
                enc.append(nn.Dropout(dropout))
            in_c = h
        enc.append(nn.Conv1d(in_c, latent_channels, kernel_size=1))
        self.encoder = nn.Sequential(*enc)

        # Decoder: transposed convolution mirrors the encoder
        dec: list[nn.Module] = [
            nn.Conv1d(latent_channels, hidden_channels[-1], 1), nn.GELU()
        ]
        in_c = hidden_channels[-1]
        for h in reversed(hidden_channels[:-1]):
            dec += [nn.ConvTranspose1d(in_c, h, kernel_size, stride=2,
                                       padding=pad, output_padding=1),
                    _gn(h), nn.GELU()]
            if dropout > 0:
                dec.append(nn.Dropout(dropout))
            in_c = h
        dec.append(
            nn.ConvTranspose1d(in_c, channels, kernel_size, stride=2,
                               padding=pad, output_padding=1)
        )
        self.decoder = nn.Sequential(*dec)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _to_3d(self, x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if x.ndim == 2:
            return x.view(-1, self.channels, self.window_size), True
        return x, False

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [N, C*W] or [N, C, W]  →  (recon same shape, latent [N, L, W'])"""
        x3, was_flat = self._to_3d(x)
        z = self.encoder(x3)
        recon = self.decoder(z)
        if recon.shape[-1] != x3.shape[-1]:          # minor padding mismatch
            recon = F.interpolate(
                recon, size=x3.shape[-1], mode="linear", align_corners=False
            )
        if was_flat:
            recon = recon.flatten(1)
        return recon, z


# ── MLP backbone (final paper model) ────────────────────────────────────────

class MLPWindowAE(nn.Module):
    """Flat MLP encoder-decoder.

    Kept for the M6 backbone-ablation config to demonstrate that
    the method's gains are not attributable to the conv backbone.
    Input / output: [N, C*W].
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [N, C*W]  →  (recon [N, C*W], latent [N, latent_dim])"""
        z = self.encoder(x)
        return self.decoder(z), z
