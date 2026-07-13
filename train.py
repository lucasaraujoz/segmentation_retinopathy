"""
Training script for FGADR multi-lesion segmentation (HardExudate + Hemorrhage).

Usage
-----
    python train.py [options]

Arguments
---------
  --exp EXP           Experiment ID from config.EXPERIMENTS (default: 00)
                        00  baseline UNet + EfficientNet-B4
                        01  Haar wavelet, level 1, skip[0]
                        02-04  DB2/DB4/Sym4, level 1, skip[0]
                        05-06  Haar, level 2/3, skip[0]
                        07-09  Haar, level 1, increasing skip coverage

  --epochs N          Override num_epochs (e.g. 2 for a quick smoke-test)

  --folds N           Run only the first N folds sequentially starting from
                      fold-0 (e.g. --folds 1 = fold-0 only; default = all 5)

  --suffixes S        Comma-separated filename suffixes to include.
                      E.g. "3" (precise annotations only), "1,3", or omit for all.
                      Each combination gets its own cv_splits_<tag>.json cache.

  --device DEV        Force device: "cuda", "cpu", "cuda:1", etc.
                      (default: cuda if available, else cpu)

  --workers N         DataLoader num_workers (0 = single-process, safer for debug)

  --no-wandb          Disable Weights & Biases logging entirely; metrics go to
                      CSV only under outputs/exp_<id>_<name>/metrics_fold<N>.csv

  --no-tta            Disable test-time augmentation (faster, for smoke-tests)

  --no-hd95           Skip Hausdorff95 in test evaluation (faster, for smoke tests).
                      HD95 is never computed during training validation.

Examples
--------
    # Quick smoke-test, no W&B, only _3 images, fold-0 only
    python train.py --exp 00 --epochs 2 --folds 1 --suffixes 3 --no-wandb --no-hd95

    # Full run with _2 and _3 images only
    python train.py --exp 01 --suffixes 2,3

    # Full baseline, all images, with W&B
    python train.py --exp 00

Notes
-----
- Experiment configs live in config.py → EXPERIMENTS dict.
- To add a new dataset: see the registry in dataset.py (_DATASET_REGISTRY, _LOADER_REGISTRY).
- Splits are deterministic and cached: outputs/cv_splits_<suffix_tag>.json.
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import DiceLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, EXPERIMENTS
from dataset import build_dataset, load_dataframe, create_splits, filter_by_suffix
from losses import build_loss
from metrics import compute_batch_metrics, compute_hd95_epoch
from metrics_detection import evaluate_class
from models import build_model
from reporter import Reporter


# ── TTA ───────────────────────────────────────────────────────────────────────

def _tta_predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Average predictions over original + H-flip + V-flip."""
    logits = model(x)
    logits_hflip = model(torch.flip(x, dims=[3]))
    logits_hflip = torch.flip(logits_hflip, dims=[3])
    logits_vflip = model(torch.flip(x, dims=[2]))
    logits_vflip = torch.flip(logits_vflip, dims=[2])
    return (logits + logits_hflip + logits_vflip) / 3.0


def _score_predictions(preds_list, targets_list, config, use_hd95, want_scores=False):
    """Full test-metric suite for one set of per-batch logits (a single model or the
    ensemble). Returns (metrics: dict, froc_out: dict, scores: dict). `scores` (per-image
    sidecar arrays for paired stats) is only populated when want_scores=True."""
    m_list = [compute_batch_metrics(logits, tgt, config)
              for logits, tgt in zip(preds_list, targets_list)]
    metrics = {k: float(np.mean([m[k] for m in m_list])) for k in m_list[0]}
    if use_hd95:
        metrics.update(compute_hd95_epoch(preds_list, targets_list, config.classes))

    probs_np = torch.sigmoid(torch.cat(preds_list, dim=0)).numpy()      # [N, C, H, W]
    gts_np   = torch.cat(targets_list, dim=0).numpy().astype(np.uint8)
    froc_out, scores = {}, {}
    aupr_per_class = []
    for ci, cname in enumerate(config.classes):
        ev = evaluate_class(probs_np[:, ci], gts_np[:, ci])
        metrics[f'aupr_{cname}']         = ev['aupr']
        metrics[f'lesion_dice_{cname}']  = ev['lesion_dice']
        metrics[f'lesion_hd95_{cname}']  = ev['lesion_hd95']
        metrics[f'sensitivity_{cname}']  = ev['sensitivity']
        metrics[f'fp_per_image_{cname}'] = ev['fp_per_image']
        aupr_per_class.append(ev['aupr'])
        froc_out[cname] = ev['froc']
        if want_scores:
            scores[f'pixdice_{cname}'] = ev['per_image_pixel_dice']
            scores[f'lesdice_{cname}'] = ev['per_image_lesion_dice']
            scores[f'hits_{cname}']    = ev['per_lesion_hit']
    metrics['mAUPR'] = float(np.nanmean(aupr_per_class))
    return metrics, froc_out, scores


# ── BatchNorm freezing ────────────────────────────────────────────────────────

def _freeze_encoder_bn(model: torch.nn.Module):
    """
    Keep the encoder's BatchNorm layers in eval() during training so they use the
    stable pretrained ImageNet running statistics instead of recomputing noisy
    mean/var from a tiny batch (batch_size=4). Weights still train normally via
    backprop — only running_mean/running_var are frozen. Gradient accumulation
    does NOT fix this (BN still sees the physical mini-batch), so freezing is the
    practical remedy for small-batch fine-tuning.
    """
    enc = getattr(model, 'encoder', None)
    if enc is None:
        return
    for m in enc.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


# ── Epoch loops ───────────────────────────────────────────────────────────────

# Auxiliary Dice for deep supervision: same DiceLoss the main criterion uses (losses.py:43).
_AUX_DICE = DiceLoss(mode='multilabel', smooth=1.0, from_logits=True)


def _deepsup_loss(aux_logits, masks):
    """Mean Dice loss of the auxiliary heads vs the target downsampled to each head's resolution."""
    total = 0.0
    for aux in aux_logits:
        target = F.interpolate(masks.float(), size=aux.shape[-2:], mode='nearest')
        total = total + _AUX_DICE(aux, target.long())
    return total / len(aux_logits)


def train_epoch(model, loader, criterion, optimizer, scheduler, device, accumulation_steps, config):
    model.train()
    # Freezing encoder BN only helps when it holds pretrained running stats. Training
    # from scratch (encoder_weights=None) needs BN to LEARN — freezing random stats breaks it.
    if config.encoder_weights is not None:
        _freeze_encoder_bn(model)
    total_loss = 0.0
    all_metrics = []
    optimizer.zero_grad()

    pbar = tqdm(loader, desc='  train', ncols=95, leave=False)
    for step, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks  = batch['mask'].to(device)

        out = model(images)
        logits, aux_logits = out if isinstance(out, tuple) else (out, [])
        loss = criterion(logits, masks)
        if aux_logits:
            loss = loss + config.deepsup_weight * _deepsup_loss(aux_logits, masks)
        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step_loss = loss.item() * accumulation_steps
        total_loss += step_loss
        pbar.set_postfix(loss=f'{step_loss:.4f}')

        with torch.no_grad():
            all_metrics.append(compute_batch_metrics(logits, masks, config))

    avg_loss = total_loss / len(loader)
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return avg_loss, avg_metrics


@torch.no_grad()
def val_epoch(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_metrics = []

    for batch in tqdm(loader, desc='    val', ncols=95, leave=False):
        images = batch['image'].to(device)
        masks  = batch['mask'].to(device)

        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item()

        all_metrics.append(compute_batch_metrics(logits, masks, config))

    avg_loss = total_loss / len(loader)
    avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    return avg_loss, avg_metrics


@torch.no_grad()
def evaluate_with_tta(model, loader, config, device):
    """Full evaluation with TTA + Hausdorff95. Used only on the test set."""
    model.eval()
    all_metrics = []
    all_logits, all_targets = [], []

    for batch in loader:
        images = batch['image'].to(device)
        masks  = batch['mask'].to(device)

        logits = _tta_predict(model, images)
        all_metrics.append(compute_batch_metrics(logits, masks, config))
        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    hd  = compute_hd95_epoch(all_logits, all_targets, config.classes)
    avg.update(hd)
    return avg


# ── Single-fold training ──────────────────────────────────────────────────────

def train_fold(config: Config, fold_idx: int, train_df: pd.DataFrame, val_df: pd.DataFrame, device: torch.device):
    print(f'\n── Fold {fold_idx + 1}/{config.n_folds} ─────────────────────────')
    print(f'   Train: {len(train_df)} | Val: {len(val_df)}')

    reporter = Reporter(config, fold=fold_idx)

    train_ds = build_dataset(config, train_df, is_train=True)
    val_ds   = build_dataset(config, val_df,   is_train=False)

    nw = config.num_workers
    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=nw,
        pin_memory=(nw > 0),
        persistent_workers=(nw > 0),
        multiprocessing_context='spawn' if nw > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    model     = build_model(config).to(device)
    criterion = build_loss(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        steps_per_epoch=math.ceil(len(train_loader) / config.accumulation_steps),
        epochs=config.num_epochs,
        pct_start=config.scheduler_pct_start,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4,
    )

    best_dice = 0.0
    best_epoch = 0
    print(f'   Model ready ({sum(p.numel() for p in model.parameters()):,} params) — starting training...')

    for epoch in range(1, config.num_epochs + 1):
        lr_now = optimizer.param_groups[0]['lr']
        print(f'Ep {epoch:03d}/{config.num_epochs}  lr={lr_now:.2e}', flush=True)

        train_loss, train_m = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config.accumulation_steps, config
        )
        val_loss, val_m = val_epoch(model, val_loader, criterion, device, config)

        reporter.log('train', epoch, {**train_m, 'loss': train_loss}, lr=lr_now)
        reporter.log('val',   epoch, {**val_m,   'loss': val_loss})

        if val_m['dice_mean'] > best_dice:
            best_dice  = val_m['dice_mean']
            best_epoch = epoch
            ckpt_path  = reporter.save_checkpoint(model, epoch, val_m)

        dice_he  = val_m.get('dice_HardExudate', float('nan'))
        dice_hem = val_m.get('dice_Hemorrhage',  float('nan'))
        print(
            f'         val loss={val_loss:.4f} | '
            f'Dice={val_m["dice_mean"]:.4f} (HE={dice_he:.4f} HEM={dice_hem:.4f}) | '
            f'best={best_dice:.4f} ep{best_epoch}'
        )

        if epoch % 10 == 0 and hasattr(reporter, 'log_images'):
            # Log a few overlay examples to W&B
            sample_batch = next(iter(val_loader))
            with torch.no_grad():
                sample_logits = model(sample_batch['image'].to(device))
            reporter.log_images(
                sample_batch['image'], sample_batch['mask'], sample_logits.cpu(),
                epoch, config.classes
            )

    reporter.save_training_plot()
    reporter.finish()
    print(f'Fold {fold_idx + 1} done. Best val Dice={best_dice:.4f} at epoch {best_epoch}')
    return best_dice, best_epoch, str(ckpt_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='00',
                        help='Experiment ID from config.EXPERIMENTS')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override num_epochs (e.g. 2 for smoke-test)')
    parser.add_argument('--folds', type=int, default=None,
                        help='Run only first N folds from fold-0 (e.g. --folds 1 = fold-0 only, default = all 5)')
    parser.add_argument('--suffixes', type=str, default=None,
                        help='Comma-separated filename suffixes to include, e.g. "3" or "1,3" (default: all)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--workers', type=int, default=None,
                        help='DataLoader num_workers (0 = single-process, safer for debug)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing experiment outputs without asking')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable W&B; log to CSV only (outputs/exp_.../metrics_fold<N>.csv)')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation (faster, for smoke tests)')
    parser.add_argument('--no-hd95', action='store_true',
                        help='Skip Hausdorff95 in test evaluation (faster, for smoke tests)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training; load existing model_best_fold*.pth and only run test eval '
                             '(computes pixel + lesion-wise detection metrics into test_results.json).')
    args = parser.parse_args()

    if args.exp not in EXPERIMENTS:
        print(f'Unknown experiment "{args.exp}". Available: {list(EXPERIMENTS.keys())}')
        sys.exit(1)

    config = EXPERIMENTS[args.exp]
    if args.epochs:
        config.num_epochs = args.epochs
    if args.workers is not None:
        config.num_workers = args.workers
    if args.suffixes:
        config.allowed_suffixes = tuple(s.strip() for s in args.suffixes.split(','))
    if args.no_wandb:
        config.use_wandb = False

    use_tta  = not args.no_tta
    use_hd95 = not args.no_hd95

    # Overwrite guard — abort if experiment already has outputs
    exp_dir = config.exp_dir
    existing = list(exp_dir.glob('*.pth')) + list(exp_dir.glob('metrics*.csv'))
    if existing and not args.overwrite and not args.eval_only:
        print(f'\n[ERRO] Experimento {config.exp_id} já tem resultados em {exp_dir}/')
        print(f'       {len(existing)} arquivo(s): {[f.name for f in existing[:5]]}')
        print(f'       Use --overwrite para sobrescrever.')
        sys.exit(1)

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'\n{"="*60}')
    print(f'Experiment {config.exp_id}: {config.exp_name}')
    print(f'Device: {device}')
    print(f'Wavelet: family={config.wavelet_family}, level={config.wavelet_level}, '
          f'skips={config.wavelet_skip_indices}')
    suffix_label = ','.join(f'_{s}' for s in sorted(config.allowed_suffixes)) if config.allowed_suffixes else 'all'
    print(f'TTA: {use_tta} | HD95: {use_hd95} | Suffixes: {suffix_label}')
    print(f'{"="*60}')

    # Load data
    df = load_dataframe(config)
    df = filter_by_suffix(df, config.allowed_suffixes)
    suffix_tag = ''.join(sorted(config.allowed_suffixes)) if config.allowed_suffixes else 'all'
    splits_path = Path(config.output_dir) / f'cv_splits_{suffix_tag}.json'
    splits = create_splits(df, config, str(splits_path))

    test_df     = df.iloc[splits['test']]
    trainval_df = df.iloc[splits['trainval_indices']]

    n_folds = args.folds or config.n_folds
    fold_results = []

    if args.eval_only:
        # Score already-trained checkpoints without retraining (replaces dump_preds.py)
        ckpts = sorted(config.exp_dir.glob('model_best_fold*.pth'))
        if not ckpts:
            print(f'[ERRO] --eval-only mas não há model_best_fold*.pth em {config.exp_dir}/')
            sys.exit(1)
        fold_results = [{'fold': i, 'best_val_dice': float('nan'),
                         'best_epoch': -1, 'checkpoint': str(c)} for i, c in enumerate(ckpts)]
        n_folds = len(ckpts)
        print(f'\n[eval-only] pontuando {n_folds} checkpoint(s) existente(s), sem treinar.')
    else:
        for fold_idx, fold in enumerate(splits['folds'][:n_folds]):
            tv_reset = trainval_df.reset_index(drop=True)
            train_df = tv_reset.iloc[fold['train']]
            val_df   = tv_reset.iloc[fold['val']]

            best_dice, best_epoch, ckpt_path = train_fold(
                config, fold_idx, train_df, val_df, device
            )
            fold_results.append({
                'fold': fold_idx, 'best_val_dice': best_dice,
                'best_epoch': best_epoch, 'checkpoint': ckpt_path,
            })

    # ── Test evaluation ────────────────────────────────────────────────────
    tta_label = '+TTA' if use_tta else 'no-TTA'
    print(f'\n{"="*60}')
    print(f'Test evaluation (ensemble of {n_folds} checkpoints, {tta_label})')
    print(f'{"="*60}')

    nw = config.num_workers
    test_ds     = build_dataset(config, test_df.reset_index(drop=True), is_train=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size,
                             shuffle=False, num_workers=nw, pin_memory=(nw > 0),
                             persistent_workers=(nw > 0),
                             multiprocessing_context='spawn' if nw > 0 else None)

    # Ensemble: average logits from each fold's best checkpoint. Each fold is also
    # scored individually (right after its inference) so we can report per-model
    # mean±std alongside the ensemble — keeping only the running sum + current fold
    # in memory instead of all folds' predictions at once.
    all_preds, all_targets_list, all_files = None, [], []
    per_fold_metrics, per_fold_froc = [], []
    for res in fold_results:
        model = build_model(config).to(device)
        ckpt  = torch.load(res['checkpoint'], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for batch in test_loader:
                imgs  = batch['image'].to(device)
                masks = batch['mask']
                preds = _tta_predict(model, imgs).cpu() if use_tta else model(imgs).cpu()
                fold_preds.append(preds)
                if len(all_targets_list) < len(test_loader):
                    all_targets_list.append(masks)
                    all_files.extend(batch['filename'])

        fm, ff, _ = _score_predictions(fold_preds, all_targets_list, config, use_hd95)
        per_fold_metrics.append(fm)
        per_fold_froc.append(ff)

        if all_preds is None:
            all_preds = fold_preds
        else:
            all_preds = [a + b for a, b in zip(all_preds, fold_preds)]

    all_preds = [p / len(fold_results) for p in all_preds]

    # Ensemble metrics + FROC + per-image sidecar (for McNemar/Wilcoxon downstream).
    test_metrics, froc_out, scores = _score_predictions(
        all_preds, all_targets_list, config, use_hd95, want_scores=True)
    np.savez_compressed(config.exp_dir / 'test_scores.npz',
                        files=np.array(all_files, dtype=object), **scores)

    # ── Per-model aggregation: mean ± std of each metric across the fold models ──
    per_model_test = {}
    for k in per_fold_metrics[0]:
        vals = [pm[k] for pm in per_fold_metrics]
        per_model_test[k] = {'mean': float(np.nanmean(vals)),
                             'std':  float(np.nanstd(vals)),
                             'values': [float(v) for v in vals]}

    per_model_froc = {}
    for cname in config.classes:
        thrs = [p['threshold'] for p in per_fold_froc[0][cname]]
        agg = []
        for ti, t in enumerate(thrs):
            sens = [per_fold_froc[i][cname][ti]['sensitivity']  for i in range(len(per_fold_froc))]
            fps  = [per_fold_froc[i][cname][ti]['fp_per_image'] for i in range(len(per_fold_froc))]
            agg.append({'threshold': t,
                        'sensitivity_mean': float(np.nanmean(sens)), 'sensitivity_std': float(np.nanstd(sens)),
                        'fp_per_image_mean': float(np.nanmean(fps)),  'fp_per_image_std':  float(np.nanstd(fps))})
        per_model_froc[cname] = agg

    # CV summary (best_val_dice is nan in --eval-only, where we didn't train)
    valid_dice = [r['best_val_dice'] for r in fold_results if not np.isnan(r['best_val_dice'])]
    cv_dice = float(np.mean(valid_dice)) if valid_dice else float('nan')
    cv_std  = float(np.std(valid_dice)) if valid_dice else 0.0

    print(f'\nCV Dice (mean ± std): {cv_dice:.4f} ± {cv_std:.4f}')
    print('\nTest metrics:')
    for k, v in test_metrics.items():
        print(f'  {k}: {v:.4f}')

    # Save
    final = {
        'exp_id': config.exp_id,
        'exp_name': config.exp_name,
        'cv_dice_mean': float(cv_dice),
        'cv_dice_std': float(cv_std),
        'fold_results': fold_results,
        **{f'test_{k}': v for k, v in test_metrics.items()},
        'froc': froc_out,
        'per_model_test': per_model_test,
        'per_model_froc': per_model_froc,
    }
    reporter_final = Reporter(config, fold=None)
    reporter_final.save_test_results(final)
    reporter_final.log_per_model(per_model_test)
    reporter_final.log_froc(froc_out)
    reporter_final.finish()


if __name__ == '__main__':
    main()
