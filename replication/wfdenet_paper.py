"""Faithful port of WFDENet (Li, Ma & Wu, Pattern Recognition 172 (2026) 112492).

Ported 1:1 from the authors' mmsegmentation release:
    github.com/xuanli01/WFDENet
      mmseg/models/decode_heads/wfdenet_head.py
      configs/_base_/models/wfdenet_b1.py

This is deliberately NOT models/wfdenet.py. That module is the loose FGADR port
behind the W0/W_* experiments and must keep producing the same numbers; this one
exists to reproduce the published IDRiD results, and differs from it in seven
places (lateral convs, DWT scaling, rfft vs fft, complex BN, CSAB bottleneck,
the NeighborFuse residual, and the backbone).

Reference config (configs/_base_/models/wfdenet_b1.py):
    backbone   EfficientNet arch=b1, out_indices=(1,2,3,4,6) -> [16,24,40,112,1280]
    channels   64          num_classes 4 (sigmoid, multilabel)
    norm       SyncBN      align_corners False      dropout_ratio 0
Target: 9.51M parameters (paper Table 4).
"""

from typing import List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(x: torch.Tensor, size, align_corners: bool = False) -> torch.Tensor:
    return F.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)


def conv_bn_relu(in_ch: int, out_ch: int, kernel: int, padding: int = 0,
                 bias: bool = True) -> nn.Sequential:
    """mmcv ConvModule with norm_cfg=BN, act_cfg=ReLU.

    The authors pass bias=True explicitly, so the conv keeps its bias even
    though it is followed by BN. Replicated for exact parameter parity.
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=bias),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ── Backbone ──────────────────────────────────────────────────────────────────

class EfficientNetB1Features(nn.Module):
    """EfficientNet-B1 emitting the five levels the paper uses.

    mmpretrain's out_indices=(1,2,3,4,6) yields [16, 24, 40, 112, 1280] at
    strides [2, 4, 8, 16, 32]. Neither smp ([32,24,40,112,320]) nor timm's
    features_only ([16,24,40,112,320]) matches: level 5 must come from the
    1280-channel head conv, not from the last block. So we drive the full timm
    model by hand. Block->channel mapping verified against timm 1.0.27.
    """

    out_channels = (16, 24, 40, 112, 1280)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = timm.create_model(
            'efficientnet_b1', pretrained=pretrained, drop_path_rate=0.0
        )
        self.conv_stem = net.conv_stem
        self.bn1 = net.bn1                      # timm BatchNormAct2d: BN + act
        self.blocks = net.blocks
        self.conv_head = net.conv_head
        self.bn2 = net.bn2

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.bn1(self.conv_stem(x))
        f1 = self.blocks[0](x)                  #   16, 1/2
        f2 = self.blocks[1](f1)                 #   24, 1/4
        f3 = self.blocks[2](f2)                 #   40, 1/8
        x = self.blocks[3](f3)                  #   80, 1/16
        f4 = self.blocks[4](x)                  #  112, 1/16
        x = self.blocks[5](f4)                  #  192, 1/32
        x = self.blocks[6](x)                   #  320, 1/32
        f5 = self.bn2(self.conv_head(x))        # 1280, 1/32
        return [f1, f2, f3, f4, f5]


# ── Wavelet transform (Haar, authors' non-orthonormal 1/2 scaling) ────────────

class DWT(nn.Module):
    """Haar DWT by slicing. Scaling is /2 on analysis and /2 on synthesis,
    which is not orthonormal (1/sqrt(2)) but reconstructs exactly."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, torch.cat([lh, hl, hh], dim=1)


class IDWT(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll, lh, hl, hh = torch.chunk(x, chunks=4, dim=1)
        x1 = (ll - lh - hl + hh) / 2
        x2 = (ll + lh - hl - hh) / 2
        x3 = (ll - lh + hl - hh) / 2
        x4 = (ll + lh + hl + hh) / 2

        out = torch.zeros(
            (ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2),
            device=ll.device, dtype=ll.dtype,
        )
        out[:, :, 0::2, 0::2] = x1
        out[:, :, 1::2, 0::2] = x2
        out[:, :, 0::2, 1::2] = x3
        out[:, :, 1::2, 1::2] = x4
        return out


def pad_to_even(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    _, _, h, w = x.size()
    pad_h = 1 if h % 2 else 0
    pad_w = 1 if w % 2 else 0
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, pad_h, pad_w


def unpad(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h:
        x = x[:, :, :-pad_h, :]
    if pad_w:
        x = x[:, :, :, :-pad_w]
    return x


# ── Complex-valued building blocks (§3.4) ────────────────────────────────────
# Complex tensors are carried as real tensors of shape [..., 2] (real, imag).

class ComplexConv2d(nn.Module):
    """Eq. (4): (A + iB) * (Xr + iXi) = (A·Xr - B·Xi) + i(A·Xi + B·Xr)."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, bias=bias)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.real_conv(x[..., 0]) - self.imag_conv(x[..., 1])
        imag = self.real_conv(x[..., 1]) + self.imag_conv(x[..., 0])
        return torch.stack((real, imag), dim=-1)


class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.real_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   stride, padding, bias=bias)
        self.imag_conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                                   stride, padding, bias=bias)

    def forward(self, x_real, x_imag):
        real = self.real_conv(x_real) - self.imag_conv(x_imag)
        imag = self.real_conv(x_imag) + self.imag_conv(x_real)
        return real, imag


class ComplexBatchNorm(nn.Module):
    """Trabelsi complex BN: whitens using the full 2x2 real/imag covariance.

    Note this is the real thing, not the per-part BN approximation used in
    models/wfdenet.py.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.num_features = num_features // 2
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.Wrr = nn.Parameter(torch.Tensor(self.num_features))
            self.Wri = nn.Parameter(torch.Tensor(self.num_features))
            self.Wii = nn.Parameter(torch.Tensor(self.num_features))
            self.Br = nn.Parameter(torch.Tensor(self.num_features))
            self.Bi = nn.Parameter(torch.Tensor(self.num_features))
        else:
            for name in ('Wrr', 'Wri', 'Wii', 'Br', 'Bi'):
                self.register_parameter(name, None)

        if self.track_running_stats:
            self.register_buffer('RMr', torch.zeros(self.num_features))
            self.register_buffer('RMi', torch.zeros(self.num_features))
            self.register_buffer('RVrr', torch.ones(self.num_features))
            self.register_buffer('RVri', torch.zeros(self.num_features))
            self.register_buffer('RVii', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked',
                                 torch.tensor(0, dtype=torch.long))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.RMr.zero_()
            self.RMi.zero_()
            self.RVrr.fill_(1)
            self.RVri.zero_()
            self.RVii.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.Br.data.zero_()
            self.Bi.data.zero_()
            self.Wrr.data.fill_(1)
            self.Wri.data.uniform_(-.9, +.9)
            self.Wii.data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        xr, xi = inputs[..., 0], inputs[..., 1]
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum

        training = self.training or not self.track_running_stats
        redux = [i for i in reversed(range(xr.dim())) if i != 1]
        vdim = [1] * xr.dim()
        vdim[1] = xr.size(1)

        if training:
            Mr, Mi = xr, xi
            for d in redux:
                Mr = Mr.mean(d, keepdim=True)
                Mi = Mi.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RMr.lerp_(Mr.squeeze().detach(), exponential_average_factor)
                self.RMi.lerp_(Mi.squeeze().detach(), exponential_average_factor)
        else:
            Mr = self.RMr.view(vdim)
            Mi = self.RMi.view(vdim)

        xr, xi = xr - Mr, xi - Mi

        if training:
            Vrr, Vri, Vii = xr * xr, xr * xi, xi * xi
            for d in redux:
                Vrr = Vrr.mean(d, keepdim=True)
                Vri = Vri.mean(d, keepdim=True)
                Vii = Vii.mean(d, keepdim=True)
            if self.track_running_stats:
                self.RVrr.lerp_(Vrr.squeeze().detach(), exponential_average_factor)
                self.RVri.lerp_(Vri.squeeze().detach(), exponential_average_factor)
                self.RVii.lerp_(Vii.squeeze().detach(), exponential_average_factor)
        else:
            Vrr = self.RVrr.view(vdim)
            Vri = self.RVri.view(vdim)
            Vii = self.RVii.view(vdim)

        Vrr = Vrr + self.eps
        Vii = Vii + self.eps
        tau = Vrr + Vii
        delta = torch.addcmul(Vrr * Vii, Vri, Vri, value=-1)
        s = delta.sqrt()
        t = (tau + 2 * s).sqrt()
        rst = (s * t).reciprocal()
        Urr = (s + Vii) * rst
        Uii = (s + Vrr) * rst
        Uri = (-Vri) * rst

        if self.affine:
            Wrr, Wri, Wii = (self.Wrr.view(vdim), self.Wri.view(vdim),
                             self.Wii.view(vdim))
            Zrr = (Wrr * Urr) + (Wri * Uri)
            Zri = (Wrr * Uri) + (Wri * Uii)
            Zir = (Wri * Urr) + (Wii * Uri)
            Zii = (Wri * Uri) + (Wii * Uii)
        else:
            Zrr, Zri, Zir, Zii = Urr, Uri, Uri, Uii

        yr = (Zrr * xr) + (Zri * xi)
        yi = (Zir * xr) + (Zii * xi)

        if self.affine:
            yr = yr + self.Br.view(vdim)
            yi = yi + self.Bi.view(vdim)

        return torch.stack((yr, yi), dim=-1)


class ComplexConvLayer(nn.Module):
    """CCL: 1x1 complex conv -> complex BN -> CReLU."""

    def __init__(self, channels: int):
        super().__init__()
        self.ccl = nn.Sequential(
            ComplexConv2d(channels, channels, kernel_size=1, padding=0, stride=1),
            ComplexBatchNorm(channels * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ccl(x)


class ComplexCA(nn.Module):
    """CCAB (Eq. 5-6): ECA-style channel attention in the Fourier domain."""

    def __init__(self):
        super().__init__()
        self.avg_pool_real = nn.AdaptiveAvgPool2d(1)
        self.max_pool_real = nn.AdaptiveMaxPool2d(1)
        self.avg_pool_imag = nn.AdaptiveAvgPool2d(1)
        self.max_pool_imag = nn.AdaptiveMaxPool2d(1)
        self.conv = ComplexConv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_real, x_imag = x[..., 0], x[..., 1]
        real = self.avg_pool_real(x_real) + self.max_pool_real(x_real)
        imag = self.avg_pool_imag(x_imag) + self.max_pool_imag(x_imag)
        real, imag = self.conv(real.squeeze(-1).transpose(-1, -2),
                               imag.squeeze(-1).transpose(-1, -2))
        real = real.transpose(-1, -2).unsqueeze(-1)
        imag = imag.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(torch.stack((real, imag), dim=-1))
        return x * y.expand_as(x)


class ComplexSA(nn.Module):
    """CSAB (Eq. 7-8): 1x1 complex convs squeeze C -> C/4 -> 1."""

    def __init__(self, channels: int):
        super().__init__()
        self.sa = nn.Sequential(
            ComplexConv2d(channels, channels // 4, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            ComplexConv2d(channels // 4, 1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.sa(x)


class CCFAM(nn.Module):
    """Complex Convolutional Frequency Attention Module (§3.4).

    Runs in fp32 regardless of autocast: complex FFT ops are numerically
    fragile in half precision, and the authors force this too.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.cconv = ComplexConvLayer(channels)
        self.ccam = ComplexCA()
        self.csam = ComplexSA(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.float32)
            x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
            x = self.cconv(torch.stack((x.real, x.imag), dim=-1))
            x = self.ccam(x)
            x = self.csam(x)
            x = torch.view_as_complex(x.contiguous())
            return torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')


# ── Fusion blocks ────────────────────────────────────────────────────────────

class FPNHead(nn.Module):
    """MFF, Eq. (2): out_i = conv_i(x_i + Up(out_{i+1})), top-down over 5 levels."""

    def __init__(self, channels: int, align_corners: bool = False, bias: bool = True):
        super().__init__()
        self.align_corners = align_corners
        self.in_nums = 5
        self.scale_heads = nn.ModuleList([
            conv_bn_relu(channels, channels, 3, padding=1, bias=bias)
            for _ in range(self.in_nums)
        ])

    def forward(self, laterals: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = list(laterals)          # do not mutate the caller's list
        out = [None] * self.in_nums
        for i in range(self.in_nums - 1, 0, -1):
            out[i] = self.scale_heads[i](laterals[i])
            laterals[i - 1] = laterals[i - 1] + resize(
                out[i], size=laterals[i - 1].shape[2:],
                align_corners=self.align_corners,
            )
        out[0] = self.scale_heads[0](laterals[0])
        return out


class NeighborFuse(nn.Module):
    """Eq. (10): A_l = x_l + conv3x3(x_l + conv1x1(L_{l-1}) + conv1x1(Up(x_{l+1}))).

    down_f is the low-frequency component of the finer level below (already at
    x's resolution after its DWT); up_f is the raw encoder feature of the
    coarser level above, which needs upsampling.
    """

    def __init__(self, channels: int, align_corners: bool = False,
                 bias: bool = True, bottom: bool = False, top: bool = False):
        super().__init__()
        self.align_corners = align_corners
        if not bottom:
            self.upconv = nn.Conv2d(channels, channels, kernel_size=1)
        if not top:
            self.downconv = nn.Conv2d(channels, channels, kernel_size=1)
        self.fuse = conv_bn_relu(channels, channels, 3, padding=1, bias=bias)

    def forward(self, x, down_f, up_f):
        agg = x
        if down_f is not None:
            agg = agg + self.downconv(down_f)
        if up_f is not None:
            agg = agg + self.upconv(
                resize(up_f, size=x.shape[2:], align_corners=self.align_corners)
            )
        return x + self.fuse(agg)


# ── Full network ─────────────────────────────────────────────────────────────

class WFDENetPaper(nn.Module):
    """WFDENet exactly as released by the authors.

    Returns raw logits at input resolution. In training mode returns
    (main, aux_hfb, aux_lfb); in eval mode returns only the main map.
    """

    def __init__(self, num_classes: int = 4, channels: int = 64,
                 in_channels: Tuple[int, ...] = (16, 24, 40, 112, 1280),
                 align_corners: bool = False, bias: bool = True,
                 pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.align_corners = align_corners

        self.backbone = EfficientNetB1Features(pretrained=pretrained)

        # Two convs per level to unify channels to C=64 (§3.1).
        # First is a bare 1x1 (no norm, no act), second is 3x3 + BN + ReLU.
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, channels, 1, bias=True),
                conv_bn_relu(channels, channels, 3, padding=1, bias=bias),
            )
            for c in in_channels
        ])

        self.dwt = nn.ModuleList([DWT() for _ in range(5)])
        self.idwt = nn.ModuleList([IDWT() for _ in range(5)])
        self.ccfam = nn.ModuleList([CCFAM(channels * 3) for _ in range(5)])

        self.nf = nn.ModuleList([
            NeighborFuse(channels, align_corners, bias, top=True),
            NeighborFuse(channels, align_corners, bias),
            NeighborFuse(channels, align_corners, bias),
            NeighborFuse(channels, align_corners, bias),
            NeighborFuse(channels, align_corners, bias, bottom=True),
        ])

        self.fpn_l = FPNHead(channels, align_corners, bias)
        self.fpn_h = FPNHead(channels * 3, align_corners, bias)
        self.fpn_main = FPNHead(channels, align_corners, bias)

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.spv_l = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.spv_h = nn.Conv2d(channels * 3, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        out_size = x.shape[2:]
        h, w = out_size
        if h % 32 or w % 32:
            # Levels 0-3 are DWT'd without padding (only level 4 is padded, as
            # in the authors' code), so their spatial dims must stay even down
            # to stride 16. mmseg guarantees this via data_preprocessor size.
            raise ValueError(
                f'input must be divisible by 32, got {h}x{w}. '
                f'The paper uses 960x1440 for IDRiD and 1024x1024 for DDR.'
            )
        feats = self.backbone(x)

        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats)]
        lat4_pad, pad_h, pad_w = pad_to_even(laterals[4])

        # Wavelet-based high-low frequency decomposition (§3.2)
        lf, hf = [], []
        for i in range(5):
            src = lat4_pad if i == 4 else laterals[i]
            low, high = self.dwt[i](src)
            lf.append(low)
            hf.append(high)

        # Low-frequency booster (§3.3)
        out_lfb = self.fpn_l(lf)

        # High-frequency booster (§3.4)
        out_hfb = self.fpn_h([self.ccfam[i](hf[i]) for i in range(5)])

        # Segmentation decoder (§3.5)
        out_f = [self.idwt[i](torch.cat([out_lfb[i], out_hfb[i]], dim=1))
                 for i in range(5)]
        out_f[4] = unpad(out_f[4], pad_h, pad_w)

        fused = [
            self.nf[0](laterals[0], None,   laterals[1]) + out_f[0],
            self.nf[1](laterals[1], lf[0],  laterals[2]) + out_f[1],
            self.nf[2](laterals[2], lf[1],  laterals[3]) + out_f[2],
            self.nf[3](laterals[3], lf[2],  laterals[4]) + out_f[3],
            self.nf[4](laterals[4], lf[3],  None)        + out_f[4],
        ]
        out_sd = self.fpn_main(fused)

        main = resize(self.conv_seg(out_sd[0]), out_size, self.align_corners)
        if not self.training:
            return main

        aux_h = resize(self.spv_h(out_hfb[0]), out_size, self.align_corners)
        aux_l = resize(self.spv_l(out_lfb[0]), out_size, self.align_corners)
        return main, aux_h, aux_l


def build_wfdenet_paper(num_classes: int = 4, pretrained: bool = True) -> WFDENetPaper:
    return WFDENetPaper(num_classes=num_classes, pretrained=pretrained)
