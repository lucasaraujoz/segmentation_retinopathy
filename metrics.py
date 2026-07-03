"""
Segmentation metrics: Dice, IoU, Hausdorff95 — computed per class.

Dice and IoU operate on batched tensors (GPU).
Hausdorff95 operates per-image on CPU numpy arrays (evaluated at test time only,
not during training, due to the KD-tree cost).
"""

from __future__ import annotations
import numpy as np
import torch
from scipy.spatial import cKDTree


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Per-class Dice score for a batch.

    Args:
        pred:   Predicted logits [B, C, H, W] or probabilities (will sigmoid if > 1).
        target: Binary ground truth [B, C, H, W].
        smooth: Laplace smoothing to avoid 0/0.

    Returns:
        Tensor of shape [C] with mean Dice per class over the batch.
    """
    if pred.requires_grad or pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    pred_bin = (pred > 0.5).float()

    # Flatten spatial dims: [B, C, H*W]
    p = pred_bin.flatten(2)
    t = target.float().flatten(2)

    intersection = (p * t).sum(dim=-1)       # [B, C]
    cardinality   = p.sum(dim=-1) + t.sum(dim=-1)  # [B, C]

    dice = (2 * intersection + smooth) / (cardinality + smooth)  # [B, C]
    return dice.mean(dim=0)   # [C]


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Per-class Jaccard (IoU) for a batch. Returns [C]."""
    if pred.requires_grad or pred.max() > 1.0 or pred.min() < 0.0:
        pred = torch.sigmoid(pred)
    pred_bin = (pred > 0.5).float()

    p = pred_bin.flatten(2)
    t = target.float().flatten(2)

    intersection = (p * t).sum(dim=-1)
    union = p.sum(dim=-1) + t.sum(dim=-1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean(dim=0)


def hausdorff95(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    """
    Symmetric Hausdorff distance at the 95th percentile between two binary masks.

    Args:
        pred_bin:   2D numpy array {0, 1}.
        target_bin: 2D numpy array {0, 1}.

    Returns:
        95th-percentile Hausdorff distance in pixels.
        Returns 0.0 if both masks are empty.
        Returns nan if one mask is empty (lesion not detected / spurious prediction).
    """
    pred_pts   = np.argwhere(pred_bin)
    target_pts = np.argwhere(target_bin)

    if len(pred_pts) == 0 and len(target_pts) == 0:
        return 0.0
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float('nan')

    tree_t = cKDTree(target_pts)
    tree_p = cKDTree(pred_pts)

    d_p2t = tree_t.query(pred_pts)[0]    # each pred pt → nearest target pt
    d_t2p = tree_p.query(target_pts)[0]  # each target pt → nearest pred pt

    return float(np.percentile(np.concatenate([d_p2t, d_t2p]), 95))


def _to_binary(logits: torch.Tensor, task: str) -> torch.Tensor:
    """Convert raw logits to binary predictions, task-aware. Returns float tensor."""
    if task == 'multilabel':
        return (torch.sigmoid(logits) > 0.5).float()
    if task == 'multiclass':
        raise NotImplementedError(
            "task='multiclass': convert via argmax → one-hot and implement here."
        )
    raise ValueError(f"Unknown task: {task!r}")


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config,               # Config — replaces class_names for task-awareness
) -> dict[str, float]:
    """
    Compute Dice and IoU for one batch. Returns flat dict for logging.
    Hausdorff95 is NOT computed here (too slow per-batch); use compute_hd95_epoch.
    """
    pred_bin = _to_binary(logits.detach(), config.task)
    dice = dice_score(pred_bin, targets)  # [C]
    iou  = iou_score(pred_bin, targets)   # [C]

    metrics = {}
    for i, name in enumerate(config.classes):
        metrics[f'dice_{name}'] = dice[i].item()
        metrics[f'iou_{name}']  = iou[i].item()
    metrics['dice_mean'] = dice.mean().item()
    metrics['iou_mean']  = iou.mean().item()
    return metrics


def compute_hd95_epoch(
    all_logits: list[torch.Tensor],
    all_targets: list[torch.Tensor],
    class_names: tuple[str, ...],
) -> dict[str, float]:
    """
    Compute mean Hausdorff95 over a full epoch (per class).
    Call at validation/test time, NOT inside the training loop.

    Args:
        all_logits:  List of [B, C, H, W] tensors (raw logits).
        all_targets: List of [B, C, H, W] tensors (binary targets).
        class_names: Class names in channel order.

    Returns:
        Dict with keys like 'hd95_HardExudate', 'hd95_Hemorrhage'.
        Values are means across images (nan-ignored).
    """
    per_class: dict[str, list[float]] = {n: [] for n in class_names}

    for logits, targets in zip(all_logits, all_targets):
        probs = torch.sigmoid(logits)
        pred_bin = (probs > 0.5).cpu().numpy().astype(np.uint8)  # [B, C, H, W]
        tgt_bin  = targets.cpu().numpy().astype(np.uint8)

        B, C, H, W = pred_bin.shape
        for b in range(B):
            for c, name in enumerate(class_names):
                hd = hausdorff95(pred_bin[b, c], tgt_bin[b, c])
                if not np.isnan(hd):
                    per_class[name].append(hd)

    return {
        f'hd95_{name}': float(np.mean(v)) if v else float('nan')
        for name, v in per_class.items()
    }
