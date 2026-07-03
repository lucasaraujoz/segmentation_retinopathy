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

from .wavelet import WaveletSkipConnection
from config import Config


class WaveletUnet(nn.Module):
    """UNet with optional wavelet-enhanced skip connections."""

    def __init__(
        self,
        base: smp.Unet,
        wavelet_skip_indices: tuple,
        wavelet_family: str,
        wavelet_level: int,
    ):
        super().__init__()
        self.encoder = base.encoder
        self.decoder = base.decoder
        self.seg_head = base.segmentation_head

        # Probe feature channel sizes with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 512, 512)
            features = self.encoder(dummy)

        self.wavelet_modules = nn.ModuleDict()
        for skip_idx in wavelet_skip_indices:
            feat_idx = skip_idx + 1    # features[0] is raw input, skip[0] = features[1]
            in_ch = features[feat_idx].shape[1]
            key = str(skip_idx)
            self.wavelet_modules[key] = WaveletSkipConnection(in_ch, wavelet_family, wavelet_level)

        self.wavelet_skip_indices = wavelet_skip_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = list(self.encoder(x))

        for skip_idx in self.wavelet_skip_indices:
            feat_idx = skip_idx + 1
            features[feat_idx] = self.wavelet_modules[str(skip_idx)](features[feat_idx])

        decoder_out = self.decoder(features)
        return self.seg_head(decoder_out)


def build_model(config: Config) -> nn.Module:
    """
    Instantiate and return the appropriate model for the experiment config.
    All models produce raw logits (no sigmoid) for use with BCEWithLogitsLoss.
    """
    base = smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=3,
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
    )
