import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss

from config import Config


class DiceFocalLoss(nn.Module):
    """
    Weighted combination of Dice and Focal loss for multi-label segmentation.

    Focal loss with pos_weight addresses extreme class imbalance (foreground
    pixels are ~100-200× rarer than background in FGADR).
    Dice loss optimises the overlap metric directly.

    Args:
        dice_weight:  Weight for Dice loss term.
        focal_weight: Weight for Focal loss term.
        pos_weight:   Per-class weight for positive pixels (list[float]).
                      Passed to BCEWithLogitsLoss. For FGADR: [136.8, 89.4].
        gamma:        Focal loss focusing parameter.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        pos_weight: list | None = None,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma

        # SMP's DiceLoss in multilabel mode handles sigmoid internally
        self.dice_loss = DiceLoss(mode='multilabel', smooth=1.0, from_logits=True)

        if pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float32)
            # Reshape to [1, C, 1, 1] so it broadcasts over [B, C, H, W]
            pw = pw.view(1, -1, 1, 1)
            self.register_buffer('pos_weight', pw)
        else:
            self.pos_weight = None

    def _focal(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = self.pos_weight
        if pw is not None:
            pw = pw.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=pw, reduction='none'
        )
        p_t = torch.exp(-bce)
        return ((1 - p_t) ** self.gamma * bce).mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        d = self.dice_loss(logits, targets.long())
        f = self._focal(logits, targets)
        return self.dice_weight * d + self.focal_weight * f


def build_loss(config: Config) -> nn.Module:
    """Return the appropriate loss for config.task. To add a new task, implement here."""
    if config.task == 'multilabel':
        return DiceFocalLoss(
            dice_weight=config.dice_weight,
            focal_weight=config.focal_weight,
            pos_weight=list(config.pos_weight),
            gamma=config.focal_gamma,
        )
    if config.task == 'multiclass':
        raise NotImplementedError(
            "task='multiclass': implement DiceCE loss and register here. "
            "Use smp DiceLoss(mode='multiclass', from_logits=True) + nn.CrossEntropyLoss."
        )
    raise ValueError(f"Unknown task: {config.task!r}. Valid: 'multilabel', 'multiclass'")
