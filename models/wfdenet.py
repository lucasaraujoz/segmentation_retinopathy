"""
Faithful port of WFDENet (Li, Ma, Wu — Pattern Recognition 172, 2026):
"Wavelet-based frequency decomposition and enhancement network for diabetic
retinopathy lesion segmentation."

This is NOT the passive/active wavelet-skip of models/wavelet.py. It replaces the
whole UNet decoder with WFDENet's own pipeline:

    encoder (any smp backbone) → unify channels to C
      → WHLFD:  DWT each level i → low-freq L_i (LL, C) and high-freq H_i (LH|HL|HH, 3C)
      → LFB:    FPN top-down fusion of {L_i}                       (semantics, Eq. 2)
      → HFB:    per-level CCFAM(H_i) then the same FPN fusion       (details, §3.4)
      → SD:     per level  G_i = IDWT(Ê_L_i, Ê_H_i); fuse adjacent   (decoder, Eq. 9-11)
                encoder levels residually (low neighbour via its LL); top-down decode
      → head:   1×1 conv on D_1 + ×2 upsample to input resolution

Deep supervision (faithful): auxiliary 1×1 heads on the LEVEL-1 (finest) outputs of
LFB and HFB, λ=0.5 Dice — matching the paper (one level, on the booster outputs), and
plugging into train.py's existing `(logits, aux_list)` contract.

Ablation toggles (mirror the paper's Tables 6/8):
    use_lfb   — False: Ê_L_i = L_i (raw LL, no FPN boost)
    use_hfb   — False: Ê_H_i = H_i (raw details, straight to decoder)
    use_ccfam — False: HFB does FPN only (no Fourier complex attention)
    use_sd    — False: replace adjacent-encoder aggregation with a residual block

Faithfulness notes / documented simplifications:
  * Complex BN is applied per-part (separate BN on real/imag) rather than Trabelsi's
    full 2D-covariance whitening — the common practical approximation.
  * Complex ReLU / Sigmoid are CReLU / per-part sigmoid (as in the cited complex-net work).
  * Backbone is efficientnet-b4 for comparability with our H0 baseline; the paper uses B1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet import _build_2d_filters


# ── DWT / IDWT primitives (fixed grouped conv, GPU-native) ────────────────────

def _dwt2(x: torch.Tensor, filters: torch.Tensor, pad: int):
    """One Haar/orthonormal DWT level. x:[B,C,H,W] → (LL, LH, HL, HH) at half res."""
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect') if pad > 0 else x
    xr = xp.reshape(B * C, 1, xp.shape[-2], xp.shape[-1])
    out = F.conv2d(xr, filters, stride=2)                 # [B*C, 4, H//2, W//2]
    out = out.reshape(B, C, 4, out.shape[-2], out.shape[-1])
    return out[:, :, 0], out[:, :, 1], out[:, :, 2], out[:, :, 3]


def _idwt2(ll, lh, hl, hh, filters: torch.Tensor, out_size):
    """Inverse of _dwt2 via transposed conv, centre-cropped to out_size."""
    B, C = ll.shape[:2]
    coeffs = torch.stack([ll, lh, hl, hh], dim=2).reshape(B * C, 4, ll.shape[-2], ll.shape[-1])
    rec = F.conv_transpose2d(coeffs, filters, stride=2)   # weight [4,1,L,L]: 4→1
    rec = rec.reshape(B, C, rec.shape[-2], rec.shape[-1])
    Ht, Wt = out_size
    top = (rec.shape[-2] - Ht) // 2
    left = (rec.shape[-1] - Wt) // 2
    return rec[..., top:top + Ht, left:left + Wt]


def _conv3x3(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


# ── Complex-valued operations (CCFAM building blocks, §3.4.1-3.4.3) ────────────

class ComplexConv2d(nn.Module):
    """Complex 2D conv (Eq. 4): two real convs with weights A (real) and B (imag).
    out_r = A*Xr − B*Xi ;  out_i = A*Xi + B*Xr."""

    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.conv_a = nn.Conv2d(cin, cout, kernel_size, padding=padding, bias=True)
        self.conv_b = nn.Conv2d(cin, cout, kernel_size, padding=padding, bias=True)

    def forward(self, xr, xi):
        return self.conv_a(xr) - self.conv_b(xi), self.conv_a(xi) + self.conv_b(xr)


class ComplexConv1d(nn.Module):
    """Complex 1D conv over the channel axis (ECA-style local channel attention)."""

    def __init__(self, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.conv_a = nn.Conv1d(1, 1, kernel_size, padding=p, bias=True)
        self.conv_b = nn.Conv1d(1, 1, kernel_size, padding=p, bias=True)

    def forward(self, xr, xi):
        return self.conv_a(xr) - self.conv_b(xi), self.conv_a(xi) + self.conv_b(xr)


class ComplexBN(nn.Module):
    """Per-part batch norm (documented approximation of Trabelsi complex BN)."""

    def __init__(self, num_features):
        super().__init__()
        self.bn_r = nn.BatchNorm2d(num_features)
        self.bn_i = nn.BatchNorm2d(num_features)

    def forward(self, xr, xi):
        return self.bn_r(xr), self.bn_i(xi)


def _crelu(xr, xi):
    return F.relu(xr), F.relu(xi)


def _csigmoid(xr, xi):
    return torch.sigmoid(xr), torch.sigmoid(xi)


def _cmul(ar, ai, br, bi):
    """Complex Hadamard product (broadcasts): (ar+i·ai)·(br+i·bi)."""
    return ar * br - ai * bi, ar * bi + ai * br


class CCAB(nn.Module):
    """Complex channel attention (Eq. 5-6): (GAP+GMP) → complex conv1d(k=3) → csigmoid → ⊙."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = ComplexConv1d(kernel_size)

    def forward(self, xr, xi):
        B, C = xr.shape[:2]
        # Complex global average + max pooling over spatial dims → [B, C, 1, 1]
        ar = F.adaptive_avg_pool2d(xr, 1) + F.adaptive_max_pool2d(xr, 1)
        ai = F.adaptive_avg_pool2d(xi, 1) + F.adaptive_max_pool2d(xi, 1)
        # [B,C,1,1] → [B,1,C] for conv1d over channels
        ar, ai = ar.view(B, 1, C), ai.view(B, 1, C)
        ar, ai = self.conv(ar, ai)
        ar, ai = _csigmoid(ar, ai)
        ar, ai = ar.view(B, C, 1, 1), ai.view(B, C, 1, 1)
        return _cmul(xr, xi, ar, ai)


class CSAB(nn.Module):
    """Complex spatial attention (Eq. 7-8): two 1×1 complex convs (ReLU→sigmoid) → 1ch → ⊙."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ComplexConv2d(channels, channels, kernel_size=1, padding=0)
        self.conv2 = ComplexConv2d(channels, 1, kernel_size=1, padding=0)

    def forward(self, xr, xi):
        ar, ai = self.conv1(xr, xi)
        ar, ai = _crelu(ar, ai)
        ar, ai = self.conv2(ar, ai)
        ar, ai = _csigmoid(ar, ai)               # [B,1,H,W]
        return _cmul(xr, xi, ar, ai)


class CCFAM(nn.Module):
    """Complex Convolutional Frequency Attention Module (§3.4): FFT → complex conv
    → CCAB → CSAB → IFFT. Operates on the 3C high-frequency features of one level."""

    def __init__(self, channels):
        super().__init__()
        self.cconv = ComplexConv2d(channels, channels, kernel_size=3, padding=1)
        self.cbn = ComplexBN(channels)
        self.ccab = CCAB(channels)
        self.csab = CSAB(channels)

    def forward(self, x):
        spec = torch.fft.fft2(x, norm='ortho')
        xr, xi = spec.real, spec.imag
        xr, xi = self.cconv(xr, xi)
        xr, xi = self.cbn(xr, xi)
        xr, xi = _crelu(xr, xi)
        xr, xi = self.ccab(xr, xi)
        xr, xi = self.csab(xr, xi)
        rec = torch.fft.ifft2(torch.complex(xr, xi), norm='ortho')
        return rec.real


# ── FPN multi-scale fusion (LFB core / HFB step 2, Eq. 2) ──────────────────────

class FPNFuse(nn.Module):
    """Top-down FPN over 5 levels (finest→coarsest = index 0→4 in resolution).
    Ê_top = conv(x_top);  Ê_i = conv(Up2(Ê_{i+1}) + x_i).  Levels given coarse→fine? No —
    we receive them fine(0)→coarse(4) and fuse from coarse down to fine."""

    def __init__(self, channels, n_levels=5):
        super().__init__()
        self.convs = nn.ModuleList([_conv3x3(channels, channels) for _ in range(n_levels)])

    def forward(self, feats):
        # feats: list length 5, index 0 = finest (highest res), 4 = coarsest (lowest res)
        n = len(feats)
        out = [None] * n
        out[n - 1] = self.convs[n - 1](feats[n - 1])                 # coarsest, no lower input
        for i in range(n - 2, -1, -1):
            up = F.interpolate(out[i + 1], size=feats[i].shape[-2:],
                               mode='bilinear', align_corners=False)
            out[i] = self.convs[i](up + feats[i])
        return out


# ── The network ───────────────────────────────────────────────────────────────

class WFDENet(nn.Module):
    """WFDENet with an smp encoder backbone. Returns logits (eval) or
    (logits, [aux_lfb, aux_hfb]) (train, if deep supervision)."""

    def __init__(self, encoder, encoder_channels, out_channels: int,
                 wavelet: str = 'haar', unify_channels: int = 64,
                 use_lfb: bool = True, use_hfb: bool = True,
                 use_ccfam: bool = True, use_sd: bool = True,
                 deep_supervision: bool = False):
        super().__init__()
        self.encoder = encoder
        self.n_levels = 5
        self.use_lfb = use_lfb
        self.use_hfb = use_hfb
        self.use_ccfam = use_ccfam
        self.use_sd = use_sd
        self.deep_supervision = deep_supervision
        C = unify_channels

        # DWT/IDWT filter bank (orthonormal → synthesis = analysis transpose)
        filters = _build_2d_filters(wavelet)                        # [4,1,L,L]
        self.register_buffer('filters', filters)
        self.pad = (filters.shape[-1] - 2) // 2

        # Channel unification: each of the 5 encoder levels → C (paper: 2 convs; 1×1 here)
        self.unify = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ec, C, 1, bias=False), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
            for ec in encoder_channels
        ])

        # LFB: FPN over LL (C channels). HFB: per-level CCFAM(3C) + FPN over 3C.
        if use_lfb:
            self.lfb = FPNFuse(C, self.n_levels)
        if use_hfb:
            if use_ccfam:
                self.ccfam = nn.ModuleList([CCFAM(3 * C) for _ in range(self.n_levels)])
            self.hfb = FPNFuse(3 * C, self.n_levels)

        # Segmentation decoder (SD)
        if use_sd:
            self.neigh_low = nn.ModuleList([nn.Conv2d(C, C, 1, bias=True) for _ in range(self.n_levels)])
            self.neigh_high = nn.ModuleList([nn.Conv2d(C, C, 1, bias=True) for _ in range(self.n_levels)])
        self.fuse_conv = nn.ModuleList([_conv3x3(C, C) for _ in range(self.n_levels)])
        self.dec_conv = nn.ModuleList([_conv3x3(C, C) for _ in range(self.n_levels)])

        self.head = nn.Conv2d(C, out_channels, 1)
        if deep_supervision:
            if use_lfb:
                self.aux_lfb = nn.Conv2d(C, out_channels, 1)
            if use_hfb:
                self.aux_hfb = nn.Conv2d(3 * C, out_channels, 1)

    def _whlfd(self, feats):
        """DWT each unified level → (L_i [C], H_i [3C]) at half resolution."""
        Ls, Hs = [], []
        for f in feats:
            ll, lh, hl, hh = _dwt2(f, self.filters, self.pad)
            Ls.append(ll)
            Hs.append(torch.cat([lh, hl, hh], dim=1))
        return Ls, Hs

    def forward(self, x):
        input_hw = x.shape[-2:]
        feats = self.encoder(x)
        feats = list(feats)[-self.n_levels:]                        # E_1(finest)…E_5(coarsest)
        feats = [self.unify[i](feats[i]) for i in range(self.n_levels)]

        Ls, Hs = self._whlfd(feats)                                 # half-res bands

        # Low-frequency booster
        EL = self.lfb(Ls) if self.use_lfb else Ls

        # High-frequency booster
        if self.use_hfb:
            H_in = [self.ccfam[i](Hs[i]) for i in range(self.n_levels)] if self.use_ccfam else Hs
            EH = self.hfb(H_in)
        else:
            EH = Hs

        # Segmentation decoder: reconstruct + fuse adjacent encoder levels, top-down
        D = [None] * self.n_levels
        for i in range(self.n_levels - 1, -1, -1):
            lh, hl, hh = torch.chunk(EH[i], 3, dim=1)
            G = _idwt2(EL[i], lh, hl, hh, self.filters, feats[i].shape[-2:])   # C, E_i res

            if self.use_sd:
                agg = torch.zeros_like(feats[i])
                if i > 0:  # low-level neighbour via its LL (already at E_i resolution)
                    agg = agg + self.neigh_low[i](Ls[i - 1])
                if i < self.n_levels - 1:  # high-level neighbour, upsampled
                    up = F.interpolate(feats[i + 1], size=feats[i].shape[-2:],
                                       mode='bilinear', align_corners=False)
                    agg = agg + self.neigh_high[i](up)
                A = feats[i] + self.fuse_conv[i](agg)
            else:
                A = feats[i] + self.fuse_conv[i](feats[i])          # residual block (no SD)

            M = G + A
            if i < self.n_levels - 1:
                up = F.interpolate(D[i + 1], size=M.shape[-2:], mode='bilinear', align_corners=False)
                M = M + up
            D[i] = self.dec_conv[i](M)

        logits = self.head(D[0])
        logits = F.interpolate(logits, size=input_hw, mode='bilinear', align_corners=False)

        if self.deep_supervision and self.training:
            aux = []
            if self.use_lfb:
                aux.append(self.aux_lfb(EL[0]))
            if self.use_hfb:
                aux.append(self.aux_hfb(EH[0]))
            return logits, aux
        return logits
