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
        alpha: float | None = None,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        # When alpha is set we use the *corrected* focal (p_t from raw probs +
        # alpha class-balancing, no pos_weight). When alpha is None we keep the
        # legacy pos_weight path so exp00 reproduces bit-for-bit.
        self.alpha = alpha

        # SMP's DiceLoss in multilabel mode handles sigmoid internally
        self.dice_loss = DiceLoss(mode='multilabel', smooth=1.0, from_logits=True)

        if alpha is None and pos_weight is not None:
            pw = torch.tensor(pos_weight, dtype=torch.float32)
            # Reshape to [1, C, 1, 1] so it broadcasts over [B, C, H, W]
            pw = pw.view(1, -1, 1, 1)
            self.register_buffer('pos_weight', pw)
        else:
            self.pos_weight = None

    def _focal(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.alpha is not None:
            # Correct focal: p_t is the model's probability of the TRUE class, so
            # the (1 - p_t)^gamma modulation actually focuses on hard examples
            # (including the rare positives). Class balance comes from alpha.
            targets = targets.float()
            p  = torch.sigmoid(logits)
            ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # = -log(p_t)
            p_t     = p * targets + (1 - p) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            return (alpha_t * (1 - p_t) ** self.gamma * ce).mean()

        # Legacy pos_weight path (exp00 control) — unchanged.
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


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for multi-label segmentation (Abraham & Khan, 2019).

    The Tversky index generalises Dice with separate weights for false positives
    (alpha) and false negatives (beta). Setting beta > alpha penalises missed
    lesion pixels more heavily, which favours recall — well suited to the tiny,
    highly imbalanced lesions in FGADR. The focal exponent gamma > 1 concentrates
    training on the hard (low-overlap) classes.

    Args:
        alpha: Weight for false positives.
        beta:  Weight for false negatives (beta > alpha ⇒ recall-oriented).
        gamma: Focal exponent applied to (1 - Tversky).
        smooth: Laplace smoothing.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 1.333, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).flatten(2)   # [B, C, H*W]
        t = targets.float().flatten(2)

        tp = (p * t).sum(-1)                    # [B, C]
        fp = (p * (1 - t)).sum(-1)
        fn = ((1 - p) * t).sum(-1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return ((1 - tversky) ** self.gamma).mean()


def build_loss(config: Config) -> nn.Module:
    """Return the appropriate loss for config.task. To add a new task, implement here."""
    if config.task == 'multilabel':
        if config.loss_type == 'dice_focal':
            return DiceFocalLoss(
                dice_weight=config.dice_weight,
                focal_weight=config.focal_weight,
                pos_weight=list(config.pos_weight),
                gamma=config.focal_gamma,
            )
        if config.loss_type == 'dice_focal_alpha':
            return DiceFocalLoss(
                dice_weight=config.dice_weight,
                focal_weight=config.focal_weight,
                alpha=config.focal_alpha,
                gamma=config.focal_gamma,
            )
        if config.loss_type == 'focal_tversky':
            return FocalTverskyLoss(
                alpha=config.tversky_alpha,
                beta=config.tversky_beta,
                gamma=config.tversky_gamma,
            )
        raise ValueError(
            f"Unknown loss_type: {config.loss_type!r}. "
            "Valid: 'dice_focal', 'dice_focal_alpha', 'focal_tversky'"
        )
    if config.task == 'multiclass':
        raise NotImplementedError(
            "task='multiclass': implement DiceCE loss and register here. "
            "Use smp DiceLoss(mode='multiclass', from_logits=True) + nn.CrossEntropyLoss."
        )
    raise ValueError(f"Unknown task: {config.task!r}. Valid: 'multilabel', 'multiclass'")
