"""
Model factory.

For baseline (no wavelet): wraps smp.Unet directly.
For wavelet experiments: subclasses the encoder forward to intercept features
at specified skip indices and apply WaveletSkipConnection modules.

SMP UNet encoder returns a list of feature maps:
  features[0]  : original input (stride 1) — NOT used as skip
  features[1]  : first encoder block (stride 2) ← skip index 0
  features[2]  : second encoder block (stride 4) ← skip index 1
  ...
  features[-1] : bottleneck

wavelet_skip_indices uses 0-based indexing into [features[1], features[2], ...]
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from .wavelet import WaveletSkipConnection, ActiveWaveletFusion
from config import Config


class WaveletUnet(nn.Module):
    """UNet with optional wavelet-enhanced skip connections."""

    def __init__(
        self,
        base: smp.Unet,
        wavelet_skip_indices: tuple,
        wavelet_family: str,
        wavelet_level: int,
        wavelet_include_ll: bool = False,
        wavelet_fusion: str = 'passive',
        deep_supervision: bool = False,
        out_channels: int = 1,
        in_channels: int = 3,
    ):
        super().__init__()
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.seg_head = base.segmentation_head

        # Probe feature channel sizes with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 512, 512)
            features = self.encoder(dummy)

        self.wavelet_modules = nn.ModuleDict()
        self.aux_heads = nn.ModuleDict()
        for skip_idx in wavelet_skip_indices:
            feat_idx = skip_idx + 1    # features[0] is raw input, skip[0] = features[1]
            in_ch = features[feat_idx].shape[1]
            key = str(skip_idx)
            if wavelet_fusion == 'passive':
                self.wavelet_modules[key] = WaveletSkipConnection(
                    in_ch, wavelet_family, wavelet_level, include_ll=wavelet_include_ll
                )
            else:
                self.wavelet_modules[key] = ActiveWaveletFusion(
                    in_ch, wavelet_family, wavelet_level,
                    include_ll=wavelet_include_ll, enhance=(wavelet_fusion == 'idwt_enh'),
                )
            if deep_supervision:
                self.aux_heads[key] = nn.Conv2d(in_ch, out_channels, kernel_size=1)

        self.wavelet_skip_indices = wavelet_skip_indices
        self.deep_supervision = deep_supervision

    def forward(self, x: torch.Tensor):
        features = list(self.encoder(x))

        aux_logits = []
        for skip_idx in self.wavelet_skip_indices:
            feat_idx = skip_idx + 1
            enhanced = self.wavelet_modules[str(skip_idx)](features[feat_idx])
            features[feat_idx] = enhanced
            if self.deep_supervision and self.training:
                aux_logits.append(self.aux_heads[str(skip_idx)](enhanced))

        decoder_out = self.decoder(features)
        logits = self.seg_head(decoder_out)

        # Auxiliary heads only participate during training; inference/val/TTA get a plain tensor.
        if self.deep_supervision and self.training:
            return logits, aux_logits
        return logits


def build_model(config: Config) -> nn.Module:
    """
    Instantiate and return the appropriate model for the experiment config.
    All models produce raw logits (no sigmoid) for use with BCEWithLogitsLoss.
    """
    base = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,   # 6 when input_preproc='wavelet_channels', else 3
        classes=config.out_channels,
        activation=None,       # raw logits
    )

    if not config.has_wavelet:
        return base

    return WaveletUnet(
        base=base,
        wavelet_skip_indices=config.wavelet_skip_indices,
        wavelet_family=config.wavelet_family,
        wavelet_level=config.wavelet_level,
        wavelet_include_ll=config.wavelet_include_ll,
        wavelet_fusion=config.wavelet_fusion,
        deep_supervision=config.deep_supervision,
        out_channels=config.out_channels,
        in_channels=config.in_channels,
    )
