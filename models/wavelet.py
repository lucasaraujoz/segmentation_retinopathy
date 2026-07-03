"""
GPU-native 2D DWT skip connection enhancement.

The DWT is implemented as fixed (non-learnable) grouped convolutions with
stride=2, avoiding CPU↔GPU round-trips. Supports Haar, Daubechies (db2/db4),
and Symlets (sym4) — all standard discrete wavelets compatible with DWT-2D.

Note on Mexican Hat (Ricker): that is a Continuous Wavelet and cannot be
used with DWT-2D. Use db2/db4 as discrete alternatives.

Architecture per skip connection:
  Input skip [B, C, H, W]
    ↓ DWT × `level` (only detail subbands extracted: LH, HL, HH per level)
    ↓ Upsample all detail maps to original H×W
    ↓ Concat: [skip, LH_1, HL_1, HH_1, ..., LH_L, HL_L, HH_L]
                  C + 3×C×level channels
    ↓ Conv1×1 → C channels + BN + ReLU
  Output [B, C, H, W]  (same shape as input — decoder sees unchanged size)
"""

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_2d_filters(wavelet_name: str) -> torch.Tensor:
    """
    Build 2D analysis filter bank from pywt wavelet coefficients.
    Returns tensor of shape [4, 1, L, L]: [LL, LH, HL, HH].
    """
    w = pywt.Wavelet(wavelet_name)
    lo = torch.tensor(w.dec_lo, dtype=torch.float32)
    hi = torch.tensor(w.dec_hi, dtype=torch.float32)

    # 2D filter = outer product of 1D filters
    # kernel[i, j] operates on (height_offset=i, width_offset=j) in F.conv2d
    LL = torch.outer(lo, lo)   # approx: lo in height, lo in width
    LH = torch.outer(lo, hi)   # horizontal details: lo in height, hi in width
    HL = torch.outer(hi, lo)   # vertical details: hi in height, lo in width
    HH = torch.outer(hi, hi)   # diagonal details: hi in both

    # Stack to [4, 1, L, L]
    return torch.stack([LL, LH, HL, HH]).unsqueeze(1)


class WaveletSkipConnection(nn.Module):
    """
    Applies multi-level DWT-2D to a single encoder skip connection.
    All convolutions run on GPU; no numpy ops during forward pass.

    Args:
        in_channels:   Number of channels of the skip tensor.
        wavelet:       pywt wavelet name ('haar', 'db2', 'db4', 'sym4', ...).
        level:         Number of DWT decomposition levels (1, 2, or 3).
    """

    def __init__(self, in_channels: int, wavelet: str = 'haar', level: int = 1):
        super().__init__()
        self.level = level
        self.in_channels = in_channels

        filters = _build_2d_filters(wavelet)  # [4, 1, L, L]
        self.register_buffer('filters', filters)

        L = filters.shape[-1]
        # Padding so that output size = input_size // 2 (valid for even inputs)
        self.pad = (L - 2) // 2

        # Channel reduction after concatenation
        # Input: C (skip) + 3*C*level (detail subbands) = C*(1 + 3*level)
        total_in = in_channels * (1 + 3 * level)
        self.reduce = nn.Sequential(
            nn.Conv2d(total_in, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def _dwt_one_level(self, x: torch.Tensor):
        """Apply one level of DWT. Returns (LL, LH, HL, HH)."""
        B, C, H, W = x.shape
        p = self.pad

        xp = F.pad(x, (p, p, p, p), mode='reflect')
        # Process all channels simultaneously as groups
        xr = xp.reshape(B * C, 1, xp.shape[-2], xp.shape[-1])

        # filters: [4, 1, L, L] — applied to each channel independently
        out = F.conv2d(xr, self.filters, stride=2)   # [B*C, 4, H//2, W//2]
        out = out.reshape(B, C, 4, H // 2, W // 2)

        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        parts = [x]          # original skip always included
        approx = x

        for _ in range(self.level):
            approx, LH, HL, HH = self._dwt_one_level(approx)
            # Upsample detail subbands back to original skip resolution
            for detail in (LH, HL, HH):
                parts.append(
                    F.interpolate(detail, size=(H, W), mode='bilinear', align_corners=False)
                )

        return self.reduce(torch.cat(parts, dim=1))
