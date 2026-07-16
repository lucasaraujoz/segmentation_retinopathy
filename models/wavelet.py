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
    ↓ (optional, include_ll) also concat the final approximation LL_L upsampled → +C channels
    ↓ Conv1×1 → C channels + BN + ReLU
  Output [B, C, H, W]  (same shape as input — decoder sees unchanged size)

The LL (approximation) is normally discarded (only feeds the next level). For lesions whose
signal is low-frequency — e.g. hemorrhages, which are dark diffuse blobs separable in the LL band,
not the edge/detail bands — set include_ll=True to also feed the approximation to the decoder.
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

    def __init__(self, in_channels: int, wavelet: str = 'haar', level: int = 1,
                 include_ll: bool = False):
        super().__init__()
        self.level = level
        self.in_channels = in_channels
        self.include_ll = include_ll

        filters = _build_2d_filters(wavelet)  # [4, 1, L, L]
        self.register_buffer('filters', filters)

        L = filters.shape[-1]
        # Padding so that output size = input_size // 2 (valid for even inputs)
        self.pad = (L - 2) // 2

        # Channel reduction after concatenation
        # Input: C (skip) + 3*C*level (detail subbands) [+ C (final LL) if include_ll]
        total_in = in_channels * (1 + 3 * level) + (in_channels if include_ll else 0)
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

        if self.include_ll:
            # Also feed the final approximation (low-frequency) band to the decoder.
            parts.append(
                F.interpolate(approx, size=(H, W), mode='bilinear', align_corners=False)
            )

        return self.reduce(torch.cat(parts, dim=1))


class _CBAMLite(nn.Module):
    """Lightweight channel + spatial attention (CBAM, Woo et al. 2018), no MLP bottleneck bells.

    Used to denoise the high-frequency detail bands before IDWT reconstruction: the channel
    gate rescales per-channel responses, the spatial gate suppresses noisy background locations.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention: shared MLP over avg- and max-pooled descriptors.
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        x = x * torch.sigmoid(avg + mx)
        # Spatial attention: conv over channel-wise avg/max maps.
        s = torch.cat([x.mean(dim=1, keepdim=True),
                       x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(s))


class ActiveWaveletFusion(nn.Module):
    """Active wavelet skip: DWT → per-band enhancement → IDWT reconstruction → residual add.

    Unlike the passive WaveletSkipConnection (upsample + concat + 1x1), this reconstructs the
    enhanced sub-bands with a true inverse DWT (conv_transpose with the orthonormal analysis
    filters, which are their own synthesis transpose) so the decoder receives a coherent
    frequency-domain refinement of the skip rather than a bag of upsampled detail maps.

    Modes (``enhance``):
      * False ('idwt')      — a light per-band 1x1 conv reweights LL and the detail bands.
      * True  ('idwt_enh')  — LL gets a 3x3 conv booster (semantics / false-positive suppression),
                              the detail bands share a CBAM-lite denoising gate.

    The output is ``skip + reduce(recon)``; ``reduce``'s BN is zero-initialised so the block starts
    as an identity and learns its contribution, avoiding the "net ignores the branch" failure of the
    passive variant.
    """

    def __init__(self, in_channels: int, wavelet: str = 'haar', level: int = 1,
                 include_ll: bool = False, enhance: bool = False):
        super().__init__()
        self.level = level
        self.in_channels = in_channels
        self.enhance = enhance
        # include_ll is accepted for config symmetry; IDWT always reconstructs from the LL band,
        # so the approximation always participates regardless of the flag.

        filters = _build_2d_filters(wavelet)  # [4, 1, L, L]; orthonormal → synthesis = transpose
        self.register_buffer('filters', filters)
        L = filters.shape[-1]
        self.pad = (L - 2) // 2

        C = in_channels
        if enhance:
            self.ll_proc = nn.Sequential(
                nn.Conv2d(C, C, 3, padding=1, bias=False), nn.BatchNorm2d(C), nn.ReLU(inplace=True),
            )
            self.hf_proc = _CBAMLite(C)
        else:
            self.ll_proc = nn.Conv2d(C, C, 1, bias=True)
            self.hf_proc = nn.Conv2d(C, C, 1, bias=True)

        # Residual projection; zero-init the BN so the block is identity at start.
        self.reduce = nn.Sequential(nn.Conv2d(C, C, 1, bias=False), nn.BatchNorm2d(C))
        nn.init.zeros_(self.reduce[1].weight)

    def _dwt_one_level(self, x: torch.Tensor):
        """One DWT level. Returns (LL, LH, HL, HH) at half resolution."""
        B, C, H, W = x.shape
        p = self.pad
        xp = F.pad(x, (p, p, p, p), mode='reflect') if p > 0 else x
        xr = xp.reshape(B * C, 1, xp.shape[-2], xp.shape[-1])
        out = F.conv2d(xr, self.filters, stride=2)          # [B*C, 4, H//2, W//2]
        out = out.reshape(B, C, 4, out.shape[-2], out.shape[-1])
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

    def _idwt_one_level(self, ll, lh, hl, hh, out_size):
        """Inverse of _dwt_one_level via transposed conv, centre-cropped to out_size."""
        B, C = ll.shape[:2]
        coeffs = torch.stack([ll, lh, hl, hh], dim=2).reshape(B * C, 4, ll.shape[-2], ll.shape[-1])
        rec = F.conv_transpose2d(coeffs, self.filters, stride=2)   # weight [4,1,L,L]: 4→1 channels
        rec = rec.reshape(B, C, rec.shape[-2], rec.shape[-1])
        Ht, Wt = out_size
        top = (rec.shape[-2] - Ht) // 2
        left = (rec.shape[-1] - Wt) // 2
        return rec[..., top:top + Ht, left:left + Wt]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        approx = x
        sizes, details = [], []                 # per-level input size and detail bands
        for _ in range(self.level):
            sizes.append(approx.shape[-2:])
            approx, LH, HL, HH = self._dwt_one_level(approx)
            details.append((LH, HL, HH))

        # Enhance bands, then reconstruct level by level in reverse.
        ll = self.ll_proc(approx)
        for lvl in reversed(range(self.level)):
            LH, HL, HH = (self.hf_proc(b) for b in details[lvl])
            ll = self._idwt_one_level(ll, LH, HL, HH, sizes[lvl])

        return x + self.reduce(ll)


class AsymmetricWaveletSkip(nn.Module):
    """Asymmetric wavelet skip for LOW-frequency lesions (hemorrhage) — our contribution.

    The usual wavelet-enhancement paradigm (WFDENet, GobletNet) boosts the high-frequency
    detail bands. For hemorrhage that is counter-productive: the lesion signal lives in the
    LL (low-freq) band, while the dominant false-positive source — vessels — lives in the
    ORIENTED high-freq bands (strong, coherent LH/HL). So we treat the bands by ROLE:

      * LL  → enhanced (semantics / FP suppression), reconstructed via IDWT.
      * HF  → NOT reconstructed as lesion detail; instead it builds a spatial vessel-
              suppression gate g∈[0,1] that attenuates the reconstruction where oriented
              high-freq energy indicates a vessel.

      out = x + α · proj_out( IDWT(LLe, 0, 0, 0) ⊙ g )

    Channel-unify to `work_channels` first, so the DWT does not run on the raw 160-ch deep
    skip (the noise source we diagnosed in H2L2A/H5). Ablation flags:
      use_gate=False → LL-enhance + IDWT only (isolates the gate's effect).
      symmetric=True → enhance & reconstruct ALL bands (the "fixed H5" control); the gate is
                       disabled and HF is reconstructed as detail — tests the core hypothesis
                       that enhancing HF hurts vs. suppressing it.
    """

    def __init__(self, in_channels: int, wavelet: str = 'haar', level: int = 1,
                 work_channels: int = 64, use_gate: bool = True, symmetric: bool = False):
        super().__init__()
        self.level = level
        self.symmetric = symmetric
        self.use_gate = bool(use_gate) and not symmetric
        Cw = work_channels

        filters = _build_2d_filters(wavelet)              # [4,1,L,L]; orthonormal
        self.register_buffer('filters', filters)
        self.pad = (filters.shape[-1] - 2) // 2

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, Cw, 1, bias=False), nn.BatchNorm2d(Cw), nn.ReLU(inplace=True),
        )
        self.ll_conv = nn.Sequential(
            nn.Conv2d(Cw, Cw, 3, padding=1, bias=False), nn.BatchNorm2d(Cw), nn.ReLU(inplace=True),
        )
        if self.use_gate:
            # Vessel gate from the oriented HF magnitudes (|LH|,|HL|,|HH|) → 1 spatial map.
            self.gate_conv = nn.Sequential(
                nn.Conv2d(3 * Cw, Cw, 3, padding=1, bias=False), nn.BatchNorm2d(Cw), nn.ReLU(inplace=True),
                nn.Conv2d(Cw, 1, 1, bias=True),
            )
        if symmetric:
            self.hf_conv = nn.ModuleList([nn.Conv2d(Cw, Cw, 1, bias=True) for _ in range(3)])
        self.proj_out = nn.Conv2d(Cw, in_channels, 1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(0.1))     # gentle residual (not zero-init like H5)

    def _dwt_one_level(self, x: torch.Tensor):
        B, C, H, W = x.shape
        p = self.pad
        xp = F.pad(x, (p, p, p, p), mode='reflect') if p > 0 else x
        xr = xp.reshape(B * C, 1, xp.shape[-2], xp.shape[-1])
        out = F.conv2d(xr, self.filters, stride=2)
        out = out.reshape(B, C, 4, out.shape[-2], out.shape[-1])
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

    def _idwt_one_level(self, ll, lh, hl, hh, out_size):
        B, C = ll.shape[:2]
        coeffs = torch.stack([ll, lh, hl, hh], dim=2).reshape(B * C, 4, ll.shape[-2], ll.shape[-1])
        rec = F.conv_transpose2d(coeffs, self.filters, stride=2)
        rec = rec.reshape(B, C, rec.shape[-2], rec.shape[-1])
        Ht, Wt = out_size
        top = (rec.shape[-2] - Ht) // 2
        left = (rec.shape[-1] - Wt) // 2
        return rec[..., top:top + Ht, left:left + Wt]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.proj_in(x)
        approx = u
        sizes, details = [], []
        for _ in range(self.level):
            sizes.append(approx.shape[-2:])
            approx, lh, hl, hh = self._dwt_one_level(approx)
            details.append((lh, hl, hh))

        ll = self.ll_conv(approx)                         # enhanced coarsest LL
        for lvl in reversed(range(self.level)):
            if self.symmetric:
                lh, hl, hh = (self.hf_conv[i](b) for i, b in enumerate(details[lvl]))
            else:
                z = torch.zeros_like(ll)                  # HF dropped from reconstruction
                lh = hl = hh = z
            ll = self._idwt_one_level(ll, lh, hl, hh, sizes[lvl])

        rec = ll                                          # low-freq reconstruction at x resolution
        if self.use_gate:
            lh, hl, hh = details[0]                        # finest oriented HF
            mag = torch.cat([lh.abs(), hl.abs(), hh.abs()], dim=1)
            g = torch.sigmoid(self.gate_conv(mag))        # [B,1,h/2,w/2]
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
            rec = rec * g

        return x + self.alpha * self.proj_out(rec)


class MultiScaleAsymWaveletSkip(nn.Module):
    """Multi-scale asymmetric wavelet skip — sees ALL skips at once and fuses ACROSS levels.

    Why this exists: A0 (`AsymmetricWaveletSkip`) has the full per-skip machinery (channel unify,
    LL enhancement, IDWT, vessel gate) yet lands at baseline, while the full WFDENet (W0) clearly
    wins. The difference is that W0 aggregates the low-frequency bands ACROSS levels (its FPN-like
    LFB). So the active ingredient is multi-scale low-frequency aggregation, NOT per-skip band
    manipulation. This module isolates exactly that hypothesis: keep the asymmetry (reconstruct
    low-frequency only), add the cross-level fusion, and drop everything else W0 has — no complex
    Fourier attention (CCFAM), no custom decoder, no deep supervision, and no vessel gate (it
    demonstrably never fired: A0's FP went UP).

    Question it answers: is multi-scale low-frequency fusion alone enough to match the heavy SOTA?

      per level i:  u_i  = proj_in_i(x_i)                     (1×1 → Cw, unify)
                    LL_i = DWT(u_i).LL                        (half res; details discarded)
      top-down:     E_n  = conv3x3(LL_n)                      (coarsest)
                    E_i  = conv3x3(Up2(E_{i+1}) + LL_i)       (the cross-level fusion)
      per level i:  out_i = x_i + α_i · proj_out_i( IDWT(E_i, 0, 0, 0) )

    Resolution alignment: skip_i has 2× the resolution of skip_{i+1}, so LL_i has 2× the resolution
    of LL_{i+1} and Up2 lines them up exactly.

    `skips` must be ordered finest → coarsest (i.e. ascending `wavelet_skip_indices`, as configured).
    The FPN loop is inlined rather than importing `FPNFuse` from models/wfdenet.py on purpose:
    wfdenet.py already imports from this module, so that would be a circular import.
    """

    def __init__(self, in_channels_list, wavelet: str = 'haar', work_channels: int = 64):
        super().__init__()
        self.n = len(in_channels_list)
        Cw = work_channels

        filters = _build_2d_filters(wavelet)
        self.register_buffer('filters', filters)
        self.pad = (filters.shape[-1] - 2) // 2

        self.proj_in = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, Cw, 1, bias=False), nn.BatchNorm2d(Cw), nn.ReLU(inplace=True))
            for c in in_channels_list
        ])
        self.fpn_conv = nn.ModuleList([
            nn.Sequential(nn.Conv2d(Cw, Cw, 3, padding=1, bias=False), nn.BatchNorm2d(Cw), nn.ReLU(inplace=True))
            for _ in range(self.n)
        ])
        self.proj_out = nn.ModuleList([nn.Conv2d(Cw, c, 1, bias=True) for c in in_channels_list])
        self.alpha = nn.Parameter(torch.full((self.n,), 0.1))

    def _dwt_one_level(self, x: torch.Tensor):
        B, C, H, W = x.shape
        p = self.pad
        xp = F.pad(x, (p, p, p, p), mode='reflect') if p > 0 else x
        xr = xp.reshape(B * C, 1, xp.shape[-2], xp.shape[-1])
        out = F.conv2d(xr, self.filters, stride=2)
        out = out.reshape(B, C, 4, out.shape[-2], out.shape[-1])
        return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]

    def _idwt_one_level(self, ll, lh, hl, hh, out_size):
        B, C = ll.shape[:2]
        coeffs = torch.stack([ll, lh, hl, hh], dim=2).reshape(B * C, 4, ll.shape[-2], ll.shape[-1])
        rec = F.conv_transpose2d(coeffs, self.filters, stride=2)
        rec = rec.reshape(B, C, rec.shape[-2], rec.shape[-1])
        Ht, Wt = out_size
        top = (rec.shape[-2] - Ht) // 2
        left = (rec.shape[-1] - Wt) // 2
        return rec[..., top:top + Ht, left:left + Wt]

    def forward(self, skips):
        sizes, LLs = [], []
        for i in range(self.n):
            u = self.proj_in[i](skips[i])
            sizes.append(u.shape[-2:])
            ll, _, _, _ = self._dwt_one_level(u)          # details discarded (asymmetric)
            LLs.append(ll)

        # Cross-level FPN fusion of the low-frequency bands: coarsest → finest.
        E = [None] * self.n
        E[self.n - 1] = self.fpn_conv[self.n - 1](LLs[self.n - 1])
        for i in range(self.n - 2, -1, -1):
            up = F.interpolate(E[i + 1], size=LLs[i].shape[-2:], mode='bilinear', align_corners=False)
            E[i] = self.fpn_conv[i](up + LLs[i])

        out = []
        for i in range(self.n):
            z = torch.zeros_like(E[i])                    # low-frequency-only reconstruction
            rec = self._idwt_one_level(E[i], z, z, z, sizes[i])
            out.append(skips[i] + self.alpha[i] * self.proj_out[i](rec))
        return out
