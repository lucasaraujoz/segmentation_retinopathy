"""
Lesion-wise detection metrics (BraTS 2023 style, adapted 3D→2D) + FROC.

Rationale: pixel Dice is dominated by large lesions and punishes small/precise
annotations (FGADR _3). For clinical detection ("did we find the lesion, how many,
where") the lesion-wise view — every lesion weighted equally — is more meaningful.
FROC (sensitivity per lesion vs false positives per image) is the mammography-CAD
standard for the same question.

Reference: BraTS-2023-Metrics (rachitsaluja/BraTS-2023-Metrics). Lesion-wise Dice =
Σ Dice(TP) / (#GT_lesions + #FP); GT is dilated before connected-component labelling
so fragments of one lesion aren't counted as several; FP and FN score 0.

Pure numpy/scipy/skimage — no torch — so it runs in any environment on saved masks.
"""

from __future__ import annotations
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score


def _structure(connectivity: int) -> np.ndarray:
    """2D structuring element. connectivity 8 → full 3x3; 4 → plus-shape."""
    return ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)


def label_lesions(mask: np.ndarray, connectivity: int = 8, min_area: int = 10):
    """Label connected components, dropping ones smaller than *min_area*.

    Returns (labels [H,W] int, n_lesions). Labels are renumbered 1..n contiguously.
    """
    struct = _structure(connectivity)
    labels, n = ndimage.label(mask > 0, structure=struct)
    if n == 0:
        return labels, 0
    # drop tiny components — vectorised area count in one pass (bincount over all
    # labels) instead of an O(n_labels × H × W) per-label scan, which explodes when
    # a low FROC threshold (e.g. 0.1) fragments a noisy prob map into 1000s of comps.
    counts = np.bincount(labels.ravel(), minlength=n + 1)
    keep = counts >= min_area
    keep[0] = False                          # background
    # renumber kept labels to 1..k
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1)
    return remap[labels], int(keep.sum())


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Binary Dice between two boolean masks."""
    a = a.astype(bool); b = b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return 1.0
    return 2.0 * (a & b).sum() / s


def _hd95(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance (pixels) between two masks.
    Same logic as metrics.hausdorff95, reimplemented here to keep this module torch-free.
    Both empty → 0; one empty → nan (caller handles the FN/FP penalty)."""
    p = np.argwhere(pred_bin); g = np.argwhere(gt_bin)
    if len(p) == 0 and len(g) == 0:
        return 0.0
    if len(p) == 0 or len(g) == 0:
        return float('nan')
    d_p2g = cKDTree(g).query(p)[0]
    d_g2p = cKDTree(p).query(g)[0]
    return float(np.percentile(np.concatenate([d_p2g, d_g2p]), 95))


def lesion_wise_stats(pred_bin: np.ndarray, gt_bin: np.ndarray,
                      dil_factor: int = 2, connectivity: int = 8,
                      min_area: int = 10, hd95_penalty: float | None = None,
                      compute_hd95: bool = True) -> dict:
    """BraTS-style lesion-wise detection for one image.

    A GT lesion is a TP if ≥1 predicted component overlaps its *dilated* region;
    a predicted component overlapping no GT lesion is a FP. Per-TP Dice/HD95 are
    computed between the (undilated) GT lesion and the union of its matched preds.

    Lesion-wise Dice = Σ Dice(TP) / (#GT + #FP)  — FN/FP contribute 0.
    Lesion-wise HD95 = (Σ HD95(TP) + (#FN + #FP)·penalty) / (#GT + #FP)  — FN/FP get a
    LARGE penalty (0 would be perfect). Default penalty = image diagonal.

    Returns TP/FP/FN counts, sensitivity, lesion Dice/HD95, FP-per-image, and per-lesion
    hit/dice/hd95 lists (aligned with GT lesions) for paired statistics.
    """
    if hd95_penalty is None:
        H, W = gt_bin.shape
        hd95_penalty = float(np.hypot(H, W))     # image diagonal (~724 for 512x512)

    gt_lab, n_gt = label_lesions(gt_bin, connectivity, min_area)
    pred_lab, n_pred = label_lesions(pred_bin, connectivity, min_area)
    struct = _structure(connectivity)

    matched_pred = set()          # predicted labels claimed by some GT lesion
    per_lesion_hit, per_lesion_dice, per_lesion_hd95 = [], [], []

    for g in range(1, n_gt + 1):
        gt_lesion = gt_lab == g
        gt_dil = ndimage.binary_dilation(gt_lesion, structure=struct, iterations=dil_factor)
        overlapping = np.unique(pred_lab[gt_dil])
        overlapping = overlapping[overlapping > 0]
        if overlapping.size > 0:
            matched_pred.update(int(p) for p in overlapping)
            pred_union = np.isin(pred_lab, overlapping)
            per_lesion_hit.append(1)
            per_lesion_dice.append(_dice(gt_lesion, pred_union))
            # HD95 (cKDTree) is skipped in the FROC sweep, which only needs tp/fn/fp;
            # nan keeps the list aligned with GT lesions without paying the cost.
            per_lesion_hd95.append(_hd95(pred_union, gt_lesion) if compute_hd95 else np.nan)
        else:
            per_lesion_hit.append(0)              # FN
            per_lesion_dice.append(0.0)
            per_lesion_hd95.append(hd95_penalty)  # FN penalised (NOT 0)

    tp = int(sum(per_lesion_hit))
    fn = n_gt - tp
    fp = n_pred - len(matched_pred)          # predicted comps not claimed by any GT

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    denom = n_gt + fp                        # = TP + FN + FP
    lesion_dice = (sum(per_lesion_dice) / denom) if denom > 0 else np.nan
    # FN penalties are already in per_lesion_hd95; add FP penalties explicitly.
    lesion_hd95 = ((sum(per_lesion_hd95) + fp * hd95_penalty) / denom) if denom > 0 else np.nan

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'n_gt': n_gt, 'n_pred': n_pred,
        'sensitivity': sensitivity,
        'lesion_dice': lesion_dice,
        'lesion_hd95': lesion_hd95,
        'fp_per_image': fp,
        'hd95_penalty': hd95_penalty,
        'per_lesion_hit': per_lesion_hit,     # aligned with GT lesions (for McNemar)
        'per_lesion_dice': per_lesion_dice,
        'per_lesion_hd95': per_lesion_hd95,
    }


def froc_points(prob_map: np.ndarray, gt_bin: np.ndarray,
                thresholds=(0.9, 0.7, 0.5, 0.3, 0.1),
                dil_factor: int = 2, connectivity: int = 8,
                min_area: int = 10) -> list[dict]:
    """One image's FROC contributions: per threshold, (sensitivity, FP count).

    Aggregate across images (mean sensitivity, mean FP/image) in the caller to
    build the FROC curve.
    """
    out = []
    for t in thresholds:
        s = lesion_wise_stats(prob_map >= t, gt_bin, dil_factor, connectivity, min_area,
                              compute_hd95=False)
        out.append({'threshold': float(t),
                    'tp': s['tp'], 'fn': s['fn'], 'fp': s['fp'],
                    'sensitivity': s['sensitivity']})
    return out


def evaluate_class(probs: np.ndarray, gts: np.ndarray, thr: float = 0.5,
                   froc_thresholds=(0.9, 0.7, 0.5, 0.3, 0.1), **kw) -> dict:
    """Aggregate pixel + lesion-wise detection metrics + FROC for ONE class over a
    whole test set. `probs` and `gts` are [N, H, W] (probability map and binary GT).

    Returns aggregate scalars, the FROC curve, and per-image / per-lesion arrays
    (for later paired statistics). Pure numpy/scipy — safe to call from train.py.
    """
    per_pix, per_lesdice, per_leshd95, hits = [], [], [], []
    tp = fp = fn = 0
    froc_acc = {t: {'tp': 0, 'fn': 0, 'fp': 0} for t in froc_thresholds}

    for prob, gt in zip(probs, gts):
        pred = prob >= thr
        per_pix.append(_dice(pred, gt))
        s = lesion_wise_stats(pred, gt, **kw)
        per_lesdice.append(s['lesion_dice'])
        per_leshd95.append(s['lesion_hd95'])
        tp += s['tp']; fp += s['fp']; fn += s['fn']
        hits.extend(s['per_lesion_hit'])
        for p in froc_points(prob, gt, froc_thresholds, **kw):
            a = froc_acc[p['threshold']]
            a['tp'] += p['tp']; a['fn'] += p['fn']; a['fp'] += p['fp']

    n = len(probs)
    froc = []
    for t in froc_thresholds:
        a = froc_acc[t]
        sens = a['tp'] / (a['tp'] + a['fn']) if (a['tp'] + a['fn']) else float('nan')
        froc.append({'threshold': float(t), 'sensitivity': sens, 'fp_per_image': a['fp'] / n})

    # Pixel-wise Area Under the Precision-Recall curve (threshold-free). This is the
    # primary metric in the DR-lesion literature (M2MRF/WSRFNet/HACDR-Net/WFDENet) and,
    # unlike Dice@0.5, is sensitive to sub-1% differences. Undefined without positives.
    gts_flat = (gts.reshape(-1) > 0).astype(np.uint8)
    aupr = (float(average_precision_score(gts_flat, probs.reshape(-1)))
            if gts_flat.any() else float('nan'))

    return {
        'aupr': aupr,
        'pixel_dice': float(np.mean(per_pix)),
        'lesion_dice': float(np.nanmean(per_lesdice)),
        'lesion_hd95': float(np.nanmean(per_leshd95)),
        'sensitivity': tp / (tp + fn) if (tp + fn) else float('nan'),
        'fp_per_image': fp / n if n else float('nan'),
        'tp': tp, 'fp': fp, 'fn': fn,
        'froc': froc,
        'per_image_pixel_dice': np.array(per_pix),
        'per_image_lesion_dice': np.array(per_lesdice),
        'per_lesion_hit': np.array(hits, dtype=np.int8),
    }
