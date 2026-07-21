"""Evaluation metrics in the M2MRF/mmseg convention used by the WFDENet paper.

Ported from the authors' release:
    mmseg/evaluation/metrics/aupr.py
    mmseg/evaluation/metrics/iou_and_dice.py

The critical property is that every statistic is **accumulated over the whole
test set and only then reduced** -- not averaged per image. Our repo's
metrics.py averages per batch, which yields different (and non-comparable)
numbers, so the paper's Table 1 can only be checked against this module.

    IoU_c   = sum_intersect_c / sum_union_c                       @ thr 0.5
    Dice_c  = 2 * sum_intersect_c / (sum_pred_c + sum_label_c)    @ thr 0.5
    AUPR_c  = auc(recall_c, precision_c) over 11 thresholds linspace(0, 1, 11)

    mAUPR / mDice / mIoU = plain mean over the four lesion classes.
    (Verified: (80.42+65.73+76.02+50.31)/4 = 68.12 = the paper's mDice.)
"""

from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score

THRESHOLD_NUM = 11
DICE_THRESHOLD = 0.5


class SegEvaluator:
    """Streaming accumulator: feed logits batch by batch, then compute()."""

    def __init__(self, class_names: Sequence[str], threshold_num: int = THRESHOLD_NUM,
                 keep_probs: bool = True):
        self.class_names = tuple(class_names)
        self.num_classes = len(self.class_names)
        self.threshs = np.linspace(0, 1, threshold_num)
        self.keep_probs = keep_probs
        self.reset()

    def reset(self) -> None:
        C, T = self.num_classes, len(self.threshs)
        self.area_intersect = torch.zeros(C, dtype=torch.float64)
        self.area_pred = torch.zeros(C, dtype=torch.float64)
        self.area_label = torch.zeros(C, dtype=torch.float64)
        self.tp = torch.zeros(T, C, dtype=torch.float64)
        self.p = torch.zeros(T, C, dtype=torch.float64)
        self.fn = torch.zeros(T, C, dtype=torch.float64)
        self._probs, self._gts = [], []

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """logits, target: [B, C, H, W]. target is binary {0, 1}."""
        probs = torch.sigmoid(logits.detach().float())
        gt = target.detach().float()

        # Dice / IoU at a fixed threshold
        pred_bin = (probs > DICE_THRESHOLD).float()
        inter = (pred_bin * gt).sum(dim=(0, 2, 3))
        self.area_intersect += inter.double().cpu()
        self.area_pred += pred_bin.sum(dim=(0, 2, 3)).double().cpu()
        self.area_label += gt.sum(dim=(0, 2, 3)).double().cpu()

        # AUPR sweep
        gt_sum = gt.sum(dim=(0, 2, 3)).double().cpu()
        for i, thr in enumerate(self.threshs):
            pb = (probs > float(thr)).float()
            tp = (pb * gt).sum(dim=(0, 2, 3)).double().cpu()
            self.tp[i] += tp
            self.p[i] += pb.sum(dim=(0, 2, 3)).double().cpu()
            self.fn[i] += gt_sum - tp

        if self.keep_probs:
            self._probs.append(probs.cpu().half())
            self._gts.append(gt.cpu().to(torch.uint8))

    def compute(self) -> Dict[str, float]:
        # Divisions are on torch tensors, which yield nan/inf silently rather
        # than warning. A class absent from the whole test set gives nan here
        # and is dropped by the nanmean below.
        union = self.area_pred + self.area_label - self.area_intersect
        iou = (self.area_intersect / union).numpy()
        dice = (2 * self.area_intersect
                / (self.area_pred + self.area_label)).numpy()

        # nan fill values follow the authors' aupr.py: precision defaults to 1
        # when nothing is predicted, recall to 0 when there is no ground truth.
        ppv = np.nan_to_num((self.tp / self.p).numpy(), nan=1.0)
        recall = np.nan_to_num((self.tp / (self.tp + self.fn)).numpy(), nan=0.0)

        out: Dict[str, float] = {}
        aupr = np.zeros(self.num_classes)
        for c, name in enumerate(self.class_names):
            aupr[c] = auc(recall[:, c], ppv[:, c])
            out[f'AUPR_{name}'] = float(aupr[c] * 100)
            out[f'Dice_{name}'] = float(dice[c] * 100)
            out[f'IoU_{name}'] = float(iou[c] * 100)

        out['mAUPR'] = float(np.nanmean(aupr) * 100)
        out['mDice'] = float(np.nanmean(dice) * 100)
        out['mIoU'] = float(np.nanmean(iou) * 100)

        # Threshold-free cross-check. Not comparable to Table 1 (the paper uses
        # the 11-threshold AUPR above), but comparable to our FGADR series,
        # which uses sklearn AP throughout (metrics_detection.py).
        if self.keep_probs and self._probs:
            probs = torch.cat(self._probs).float().numpy()
            gts = torch.cat(self._gts).numpy()
            ap = []
            for c, name in enumerate(self.class_names):
                y_true = gts[:, c].ravel()
                if y_true.max() == 0:
                    ap.append(np.nan)
                    continue
                ap.append(average_precision_score(y_true, probs[:, c].ravel()))
                out[f'AP_sklearn_{name}'] = float(ap[-1] * 100)
            out['mAP_sklearn'] = float(np.nanmean(ap) * 100)

        return out


# Paper Table 1, IDRiD test set -- the replication targets.
PAPER_IDRID = {
    'AUPR_EX': 86.61, 'AUPR_HE': 69.05, 'AUPR_SE': 81.28, 'AUPR_MA': 49.91,
    'mAUPR': 71.71,
    'Dice_EX': 80.42, 'Dice_HE': 65.73, 'Dice_SE': 76.02, 'Dice_MA': 50.31,
    'mDice': 68.12,
    'IoU_EX': 67.25, 'IoU_HE': 48.95, 'IoU_SE': 61.31, 'IoU_MA': 33.61,
    'mIoU': 52.78,
}


def format_comparison(results: Dict[str, float],
                      reference: Dict[str, float] = PAPER_IDRID) -> str:
    """Side-by-side table of our run vs. the published numbers."""
    lines = [
        f'{"metric":<16}{"ours":>9}{"paper":>9}{"delta":>9}',
        '-' * 43,
    ]
    for key in reference:
        if key not in results:
            continue
        ours, ref = results[key], reference[key]
        lines.append(f'{key:<16}{ours:>9.2f}{ref:>9.2f}{ours - ref:>+9.2f}')
        if key.startswith('m'):
            lines.append('-' * 43)
    return '\n'.join(lines)
