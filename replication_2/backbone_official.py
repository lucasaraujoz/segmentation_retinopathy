"""The authors' own EfficientNet backbone, with the real ImageNet weights.

`replication/wfdenet_paper.py` uses `timm tf_efficientnet_b1.in1k` as the
closest available stand-in. This module removes that approximation: it builds
the backbone class the authors ship (mmseg/models/backbones/efficientnet.py,
a vendored mmpretrain EfficientNet) and loads the mmpretrain ImageNet
checkpoint into it.

Two things this fixes, both verified against the sources:

  1. BN eps. Their class defaults to `norm_cfg=dict(type='BN', eps=1e-3)`, but
     configs/_base_/models/wfdenet_b1.py OVERRIDES it with
     `dict(type='SyncBN', requires_grad=True)` -- no eps -- so they actually run
     at PyTorch's 1e-5. timm's `tf_` variants hardcode 1e-3.
  2. The weights themselves. The released config carries no `init_cfg`, but the
     backbone is mmpretrain's, so its 3rdparty ImageNet checkpoint is the
     matching one. Verified to load with 0 missing and 0 unexpected keys.

Conv padding was already right in both: `Conv2dAdaptivePadding` (theirs) and
`Conv2dSame` (timm) are the same TensorFlow SAME semantics.

Requires the pinned clone in replication_2/upstream/ -- see this folder's
README for the SHA and the install line.
"""

from pathlib import Path
from typing import List
import sys

import torch
import torch.nn as nn

_UPSTREAM = Path(__file__).resolve().parent / 'upstream'

# mmpretrain's efficientnet-b1_3rdparty_8xb32_in1k, the checkpoint that goes
# with the backbone class the authors vendored.
MMPRETRAIN_B1 = ('https://download.openmmlab.com/mmclassification/v0/efficientnet/'
                 'efficientnet-b1_3rdparty_8xb32_in1k_20220119-002556d9.pth')


def _import_upstream():
    if not (_UPSTREAM / 'mmseg').is_dir():
        raise RuntimeError(
            f'{_UPSTREAM} not found. The upstream clone is gitignored; recreate it:\n'
            f'  git clone https://github.com/xuanli01/WFDENet.git {_UPSTREAM}\n'
            f'  git -C {_UPSTREAM} checkout 38b0b16'
        )
    if str(_UPSTREAM) not in sys.path:
        sys.path.insert(0, str(_UPSTREAM))
    from mmseg.models.backbones.efficientnet import EfficientNet
    return EfficientNet


class OfficialEfficientNetB1(nn.Module):
    """Drop-in replacement for EfficientNetB1Features.

    Same contract: forward(x) -> [16, 24, 40, 112, 1280] at strides 2/4/8/16/32.
    """

    out_channels = (16, 24, 40, 112, 1280)

    def __init__(self, pretrained: bool = True,
                 variant: str = 'official'):
        super().__init__()
        self.variant = variant
        EfficientNet = _import_upstream()

        # Exactly configs/_base_/models/wfdenet_b1.py, with SyncBN -> BN
        # (identical on one process, and the config pins no eps either way).
        self.net = EfficientNet(
            arch='b1',
            drop_path_rate=0,
            out_indices=(1, 2, 3, 4, 6),
            frozen_stages=0,
            norm_cfg=dict(type='BN', requires_grad=True),
        )

        if pretrained:
            self._load_imagenet()

    def _load_imagenet(self) -> None:
        from mmengine.runner import CheckpointLoader

        ckpt = CheckpointLoader.load_checkpoint(MMPRETRAIN_B1, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        # The classification checkpoint prefixes everything with 'backbone.'
        backbone_state = {k[len('backbone.'):]: v for k, v in state.items()
                          if k.startswith('backbone.')}
        if not backbone_state:
            backbone_state = state

        missing, unexpected = self.net.load_state_dict(backbone_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                'mmpretrain checkpoint does not fit the authors\' backbone: '
                f'{len(missing)} missing, {len(unexpected)} unexpected. '
                'Refusing to train on a silently partial initialisation.'
            )

    def train(self, mode: bool = True):
        """Their EfficientNet.train() does not return self, which makes the
        usual `model.eval()` chain evaluate to None. Restore the nn.Module
        contract so callers can rely on it."""
        super().train(mode)
        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return list(self.net(x))
