"""Train WFDENet on IDRiD under the paper's protocol.

Training settings (paper §4.2.2, verbatim):
    SGD, lr 0.01, momentum 0.9, weight decay 0.0005
    poly LR schedule, power 0.9
    40k iterations, mini-batch size 4
    input 1440x960, EfficientNet-B1 backbone, Haar wavelet

Deliberately NOT built on train.py. That loop is epoch-based, AdamW +
OneCycleLR, 5-fold, and selects the best checkpoint by validation Dice. The
paper's protocol has *no validation set at all*: it trains for a fixed 40k
iterations on the 54 training images and evaluates the final model on the 27
test images. Periodic test evaluations here are logged for the curve only and
never used for model selection.

Usage:
    python replication/train_idrid.py --iters 40000 --out-dir outputs/repro_wfdenet_idrid
    python replication/train_idrid.py --eval-only --ckpt <path>
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2  # noqa: F401  -- must precede torch (CXXABI clash in this env)
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, get_worker_info

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replication.ddr import DDR_ROOT, DDRDataset
from replication.idrid import CLASSES, IDRID_ROOT, IDRiDDataset
from replication.loss import WFDENetLoss
from replication.metrics_mmseg import (PAPER_DDR_ABLATED, PAPER_TARGETS,
                                        SegEvaluator, format_comparison)
from replication.wfdenet_paper import build_wfdenet_paper

# Paper §4.2.2 -- identical for both datasets except the iteration budget.
BASE_LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
POLY_POWER = 0.9
BATCH_SIZE = 4

# "trained ... for 40k iterations on the IDRiD dataset and 100k iterations on
# the DDR dataset with a mini-batch size of 4" (§4.2.2).
DATASETS = {
    'idrid': dict(cls=IDRiDDataset, iters=40000, size='960x1440'),
    'ddr':   dict(cls=DDRDataset,   iters=100000, size='1024x1024'),
}


def poly_lr(base_lr: float, it: int, max_iters: int, power: float = POLY_POWER,
            min_lr: float = 0.0) -> float:
    """mmseg's poly policy.

    The paper only states power 0.9 and lr 0.01, but M2MRF -- which WFDENet
    follows for the rest of the recipe -- pins `min_lr=1e-4` in both its
    schedule files (schedule_40k_idrid.py, schedule_60k_ddr.py). With min_lr=0
    the tail of training runs at essentially zero lr instead of a small floor.
    """
    return (base_lr - min_lr) * (1 - it / max_iters) ** power + min_lr


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def _assert_masks_sane(dataset, max_fraction: float = 0.30) -> None:
    """Fail loudly if any ground-truth channel covers an implausible area.

    IDRiD_81_EX.tif ships as RGBA while every other file is palette mode; if its
    alpha channel leaks into the mask, that one image is labelled 100% lesion
    and, because metrics are dataset-aggregated, it silently destroys the EX
    scores. No real lesion covers a third of the retina, so anything above
    max_fraction means the masks are being decoded wrong.
    """
    for i in range(len(dataset)):
        sample = dataset[i]
        frac = sample['mask'].flatten(1).mean(1)
        for j, cls in enumerate(dataset.classes):
            if frac[j] > max_fraction:
                raise RuntimeError(
                    f'{sample["filename"]}: {cls} mask covers '
                    f'{frac[j] * 100:.1f}% of the image (limit '
                    f'{max_fraction * 100:.0f}%). The masks are being decoded '
                    f'incorrectly -- check the RGBA handling in '
                    f'IDRiDDataset._load_mask.'
                )


def _seed_everything(seed: int) -> None:
    """Seed python, numpy and torch. Before this only torch was seeded, so the
    augmentation stream (albumentations) was different on every run."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_transforms(dataset, seed: int) -> None:
    """albumentations 2.x keeps its OWN generators inside each Compose, seeded
    from OS entropy unless told otherwise -- seeding numpy/random globally does
    not reach them. Walk the (possibly concatenated) dataset and seed each."""
    parts = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    for i, ds in enumerate(parts):
        tf = getattr(ds, 'transform', None)
        if tf is not None and hasattr(tf, 'set_random_seed'):
            tf.set_random_seed((seed + 1000 * i) % 2**32)


def _seed_worker(worker_id: int) -> None:
    """worker_init_fn. Top-level so it pickles under the spawn context. torch
    already derives a distinct, reproducible seed per worker from the loader's
    generator; propagate it to python, numpy and the transforms."""
    info = get_worker_info()
    seed = info.seed % 2**32
    random.seed(seed)
    np.random.seed(seed)
    _seed_transforms(info.dataset, seed)


def _atomic_save(obj, path: Path) -> None:
    """A crash mid-write must not destroy the previous resume point."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def size_breakdown(scores: dict, classes=CLASSES) -> dict:
    """Per-image Dice split by GT lesion size (terciles of lesion area over the
    images that contain the class). Terciles come from the ground truth only,
    so every run scored on the same test set uses the same cut points.

    Exists because aggregate Dice (pixel-weighted) and per-image Dice
    (image-weighted) disagreed on DDR HE: HiLo gained on large haemorrhages
    (88.8% of the pixels) and lost on small ones (a third of the images)."""
    out = {}
    for c in classes:
        dice = np.asarray(scores[c], dtype=float)
        area = np.asarray(scores[f'area_gt_{c}'], dtype=float)
        pred = np.asarray(scores[f'area_pred_{c}'], dtype=float)
        pos, neg = area > 0, area == 0
        rep = {'n_pos': int(pos.sum()), 'n_neg': int(neg.sum()),
               'neg_with_fp': int((neg & (pred > 0)).sum())}
        if pos.sum() >= 3:
            q1, q2 = np.quantile(area[pos], [1 / 3, 2 / 3])
            total = area[pos].sum()
            for name, m in (('small', pos & (area <= q1)),
                            ('medium', pos & (area > q1) & (area <= q2)),
                            ('large', pos & (area > q2))):
                rep[name] = {'n': int(m.sum()),
                             'pixel_share': float(area[m].sum() / total),
                             'dice_mean': float(np.nanmean(dice[m]) * 100),
                             'dice_zero': int((dice[m] == 0).sum())}
            rep['cuts_px'] = [float(q1), float(q2)]
        out[c] = rep
    return out


def format_size_breakdown(rep: dict) -> str:
    lines = ['per-image Dice by GT lesion size (terciles of area, images with the class)',
             f'{"class":5s} {"size":7s} {"n":>4s} {"%pixels":>8s} {"Dice":>6s} {"zeros":>5s}']
    for c, r in rep.items():
        for name in ('small', 'medium', 'large'):
            if name in r:
                b = r[name]
                lines.append(f'{c:5s} {name:7s} {b["n"]:4d} {100 * b["pixel_share"]:7.1f}% '
                             f'{b["dice_mean"]:6.2f} {b["dice_zero"]:5d}')
        lines.append(f'{c:5s} {"no GT":7s} {r["n_neg"]:4d}  predicted anyway in {r["neg_with_fp"]}')
    return '\n'.join(lines)


@torch.no_grad()
def evaluate(model, loader, device, keep_probs: bool = True, amp: bool = False,
             per_image_out: Path = None) -> dict:
    model.eval()
    evaluator = SegEvaluator(CLASSES, keep_probs=keep_probs)
    filenames, area_gt, area_pred = [], [], []
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(images)      # eval mode -> single tensor
        logits = logits.float()
        if logits.shape[-2:] != masks.shape[-2:]:
            # Full-resolution scoring: the dataset kept the ground truth at its
            # native size, so bring the logits back to it -- exactly what
            # mmseg's BaseSegmentor.postprocess_result does before evaluating
            # (bilinear, align_corners=False).
            #
            # Done on CPU on purpose. A DDR image is up to 3888x2592, so the
            # evaluator's 11-threshold sweep would allocate ~1.7 GB of transient
            # GPU tensors per image -- enough to OOM a training run sharing the
            # card. The forward stays on GPU; only the scoring moves.
            logits = F.interpolate(logits.cpu(), size=masks.shape[-2:],
                                   mode='bilinear', align_corners=False)
            masks = masks.cpu()
        evaluator.update(logits, masks)
        filenames.extend(batch['filename'])
        if per_image_out is not None and logits.shape[0] == 1:
            area_gt.append(masks.sum(dim=(2, 3))[0].double().cpu().numpy())
            area_pred.append((logits > 0).sum(dim=(2, 3))[0].double().cpu().numpy())  # sigmoid>0.5
    model.train()

    if per_image_out is not None:
        # Per-image Dice for paired comparisons between runs (compare_runs.py).
        # Same test_scores.npz convention the FGADR series uses.
        per_img = evaluator.per_image_dice()
        if per_img:
            extra = {}
            if len(area_gt) == len(filenames):
                ag, ap = np.stack(area_gt), np.stack(area_pred)
                for c, name in enumerate(CLASSES):
                    extra[f'area_gt_{name}'] = ag[:, c]
                    extra[f'area_pred_{name}'] = ap[:, c]
            # Accumulated PR sweep behind AUPR, [thresholds, classes]. Lets an
            # AUPR swing be inspected threshold by threshold (e.g. DDR SE -20).
            extra['pr_thresholds'] = np.asarray(evaluator.threshs)
            extra['pr_tp'] = evaluator.tp.numpy()
            extra['pr_p'] = evaluator.p.numpy()
            extra['pr_fn'] = evaluator.fn.numpy()
            np.savez(per_image_out, filenames=np.array(filenames), **per_img, **extra)

    return evaluator.compute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='idrid', choices=list(DATASETS),
                        help='Which paper dataset to run. Sets the input size, '
                             'normalisation and default iteration budget.')
    parser.add_argument('--iters', type=int, default=None,
                        help='Default: 40000 for idrid, 100000 for ddr (§4.2.2).')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--accum', type=int, default=1,
                        help='Gradient accumulation steps. Effective batch = '
                             'batch_size * accum. Use --batch-size 2 --accum 2 '
                             'to reproduce the paper batch of 4 on a 16GB GPU.')
    parser.add_argument('--lr', type=float, default=BASE_LR)
    parser.add_argument('--min-lr', type=float, default=0.0,
                        help='Floor of the poly schedule. The paper does not '
                             'state one; M2MRF, whose recipe WFDENet follows, '
                             'pins 1e-4. Default 0.0 keeps earlier runs '
                             'reproducible; pass 1e-4 for the faithful recipe.')
    parser.add_argument('--out-dir', type=str, default='outputs/repro_wfdenet_idrid')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval-interval', type=int, default=2000,
                        help='Test-set eval for logging only; never used for selection.')
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--ckpt-interval', type=int, default=10000)
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Skip ImageNet weights for the backbone.')
    parser.add_argument('--amp', action='store_true',
                        help='bf16 autocast so batch 4 fits a 16GB GPU. CCFAM '
                             'stays fp32 (its own autocast(False)); bf16 needs '
                             'no GradScaler. Small precision deviation vs fp32.')
    parser.add_argument('--backbone', type=str, default='tf_efficientnet_b1.in1k',
                        help="timm variant, or 'official' for the authors' own "
                             'backbone class + the matching mmpretrain ImageNet '
                             'weights (BN eps 1e-5, no timm stand-in). The timm '
                             'default has TF SAME padding but eps 1e-3; '
                             "'efficientnet_b1' reproduces the earliest runs.")
    parser.add_argument('--photometric', action='store_true',
                        help='DEVIATION from the paper (which lists three '
                             'augmentations, none photometric): add colour jitter '
                             "approximating M2MRF's PhotoMetricDistortion.")
    # Ablations of the paper's proposed mechanism (Tables 6-8). With both
    # boosters off, G_l = F_l exactly -- the baseline of the Table 12 caption.
    parser.add_argument('--no-lfb', action='store_true',
                        help='Disable the low-frequency booster.')
    parser.add_argument('--no-hfb', action='store_true',
                        help='Disable the high-frequency booster (implies no CCFAM).')
    parser.add_argument('--no-ccfam', action='store_true',
                        help='Keep HFB multi-scale fusion but drop the complex '
                             'Fourier attention.')
    parser.add_argument('--full-res-eval', action='store_true',
                        help='Score at the native image resolution, as mmseg does: '
                             'the logits are resized back to ori_shape and compared '
                             'against the untouched GT. The default instead shrinks '
                             'the GT to the inference size, which on DDR also squashes '
                             'the aspect ratio. ddr only, and it disables the sklearn '
                             'AP cross-check (native-resolution probs do not fit in RAM).')
    # ── Our own architecture, trained under the paper's recipe ───────────────
    # This is NOT part of the replication. It reuses the paper's dataset, loss,
    # optimiser, schedule and evaluator as a fixed harness so that the only
    # thing differing between two runs is our HiLo skip. Cross-dataset test of
    # our mechanism, not a claim about WFDENet.
    parser.add_argument('--model', type=str, default='wfdenet',
                        choices=('wfdenet', 'unet', 'unet-hilo'),
                        help="'wfdenet' = the paper's architecture (default, the "
                             "replication). 'unet'/'unet-hilo' = our smp UNet "
                             'without/with the HiLo IDWT skip on all four skips.')
    parser.add_argument('--encoder', type=str, default='efficientnet-b4',
                        help='smp encoder for --model unet*; ignored for wfdenet.')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0,
                        help='Seeds python, numpy, torch, the DataLoader order and '
                             'the albumentations generators (per worker). cuDNN '
                             'kernels stay non-deterministic.')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Dataset directory, overriding the DDR_ROOT / IDRID_ROOT '
                             'environment variables and the built-in defaults. Set this '
                             '(or the env var) when running on another machine.')
    parser.add_argument('--train-splits', type=str, default='train',
                        help="Comma list of splits to train on, e.g. 'train,valid'. "
                             "DDR only. Anything but 'train' departs from the paper "
                             'protocol, so numbers stop being comparable to Table 2.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a last.pth written by a previous run of the SAME '
                             'configuration; continues from its iteration.')
    parser.add_argument('--resume-interval', type=int, default=2000,
                        help='Iterations between overwrites of out-dir/last.pth '
                             '(model + optimizer + iter). 0 disables.')
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    if args.iters is None:
        args.iters = spec['iters']

    # An ablated run must not be scored against the full-model table. With both
    # boosters off the model IS the paper's Table 6 row 4 baseline (G_l = F_l),
    # whose published numbers are means only -- per-class is not in the paper.
    fully_ablated = args.no_lfb and args.no_hfb
    if args.dataset == 'ddr' and fully_ablated:
        paper_ref = PAPER_DDR_ABLATED
        ref_label = 'DDR Table 6 row 4 (SD only, G_l = F_l) -- means only'
    else:
        paper_ref = PAPER_TARGETS[args.dataset]
        ref_label = f'{args.dataset} full model (Table {"2" if args.dataset == "ddr" else "1"})'
        if fully_ablated:
            ref_label += '  [!] ablated run vs FULL-model targets'

    _seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location='cpu', weights_only=False)
        # Refuse to continue a run under a different recipe: the result would
        # belong to neither configuration.
        must_match = ('dataset', 'model', 'encoder', 'iters', 'batch_size', 'accum', 'lr',
                      'min_lr', 'amp', 'train_splits', 'seed', 'no_pretrained')
        saved = resume_state.get('args', {})
        diff = {k: (saved.get(k), getattr(args, k)) for k in must_match
                if k in saved and saved.get(k) != getattr(args, k)}
        if diff:
            parser.error(f'--resume config mismatch (saved, now): {diff}')

    # Run manifest. Every earlier run in outputs/_replicacao had to have its
    # protocol reconstructed from train_log.csv and directory mtimes, because
    # nothing recorded what produced it. Ten lines, written once, end that.
    import subprocess
    from datetime import datetime
    def _git(*a, default=''):
        try:
            return subprocess.check_output(['git', *a], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return default
    manifest = 'run_config.json' if resume_state is None else \
        f'run_config_resume_{resume_state["iter"]}.json'
    (out_dir / manifest).write_text(json.dumps({
        'argv': sys.argv,
        'args': vars(args),
        'git_commit': _git('rev-parse', 'HEAD'),
        'git_branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'git_dirty': bool(_git('status', '--porcelain')),
        'started_at': datetime.now().isoformat(timespec='seconds'),
    }, indent=2, default=str))

    data_root = Path(args.data_root) if args.data_root else (
        IDRID_ROOT if args.dataset == 'idrid' else DDR_ROOT)
    if not data_root.is_dir():
        parser.error(f'dataset root not found: {data_root}\n'
                     f'    pass --data-root, or export '
                     f'{"IDRID_ROOT" if args.dataset == "idrid" else "DDR_ROOT"}=<path>')
    print(f'data root: {data_root}')

    if args.dataset == 'idrid':
        train_ds = IDRiDDataset('train', root=data_root, photometric=args.photometric)
        test_ds = IDRiDDataset('test', root=data_root, full_res_eval=args.full_res_eval)
    else:
        if args.photometric:
            parser.error('--photometric is only wired for idrid')
        splits = [s.strip() for s in args.train_splits.split(',') if s.strip()]
        if 'test' in splits:
            parser.error('--train-splits must never include test')
        # is_train=True explicitly: DDRDataset defaults it to (split == 'train'),
        # which would feed 'valid' through the TEST pipeline, without augmentation.
        parts = [DDRDataset(s, root=data_root, is_train=True) for s in splits]
        train_ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)
        test_ds = DDRDataset('test', root=data_root, full_res_eval=args.full_res_eval)
        if splits != ['train']:
            print(f'  ** training on {"+".join(splits)} '
                  f'({" + ".join(str(len(p)) for p in parts)}): NOT the paper protocol **')
    if args.dataset == 'idrid' and args.train_splits != 'train':
        parser.error('--train-splits is only wired for ddr')
    print(f'{args.dataset}: {len(train_ds)} train / {len(test_ds)} test '
          f'| {spec["size"]} | classes {CLASSES}')
    if args.photometric:
        print('  ** --photometric ON: deviation from the paper text **')
    _assert_masks_sane(test_ds)

    # spawn, not fork: forked workers deadlock against OpenCV/torch thread
    # pools (same reason train.py:258 does this).
    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,
        multiprocessing_context='spawn' if args.workers > 0 else None,
    )
    # Evaluation runs single-process on purpose. Decoding one test sample costs
    # ~0.03 s, so 225 images is ~6 s -- not worth spawning workers that then sit
    # alive alongside the persistent training workers. That combination (plus
    # the memory the final eval accumulates) segfaulted a worker at the end of
    # a 100k DDR run, losing the eval of an otherwise finished model.
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=True)

    if args.model == 'wfdenet':
        model = build_wfdenet_paper(
            num_classes=len(CLASSES), pretrained=not args.no_pretrained,
            backbone_variant=args.backbone,
            use_lfb=not args.no_lfb, use_hfb=not args.no_hfb,
            use_ccfam=not args.no_ccfam,
        ).to(device)
        print(f'backbone: {args.backbone}')
        ablated = [n for n, off in (('LFB', args.no_lfb), ('HFB', args.no_hfb),
                                    ('CCFAM', args.no_ccfam)) if off]
        print(f'ablation: {" + ".join("no-" + a for a in ablated) if ablated else "none (full model)"}')
        n_params = sum(p.numel() for p in model.parameters())
        ref = ' | paper reports 9.51M' if not ablated else ' | ablated (full model is 9.51M)'
        print(f'WFDENet: {n_params:,} params ({n_params / 1e6:.2f}M){ref}')
    else:
        # Our UNet. Config carries only what build_model reads; the training
        # recipe (SGD/poly/iters/loss) comes from this script, not from Config,
        # so none of the FGADR hyperparameters leak in.
        from config import Config
        from models.factory import build_model
        hilo = args.model == 'unet-hilo'
        our_cfg = Config(
            classes=CLASSES,
            encoder_name=args.encoder,
            encoder_weights=None if args.no_pretrained else 'imagenet',
            wavelet_family='haar', wavelet_level=1,
            wavelet_skip_indices=(0, 1, 2, 3) if hilo else (),
            wavelet_fusion='hilo' if hilo else 'passive',
        )
        model = build_model(our_cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'model: {args.model} ({args.encoder}, '
              f'{"imagenet" if not args.no_pretrained else "scratch"})')
        print(f'params: {n_params:,} ({n_params / 1e6:.2f}M)')
        if any((args.no_lfb, args.no_hfb, args.no_ccfam)):
            parser.error('--no-lfb/--no-hfb/--no-ccfam only apply to --model wfdenet')

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state['model'] if 'model' in state else state)
        print(f'loaded checkpoint {args.ckpt}')

    if args.eval_only:
        results = evaluate(model, test_loader, device,
                           keep_probs=not args.full_res_eval, amp=args.amp,
                           per_image_out=out_dir / 'test_scores.npz')
        print('\n' + format_comparison(results, paper_ref, ref_label))
        (out_dir / 'test_results.json').write_text(json.dumps(results, indent=2))
        scores = dict(np.load(out_dir / 'test_scores.npz', allow_pickle=True))
        if all(f'area_gt_{c}' in scores for c in CLASSES):
            rep = size_breakdown(scores)
            print('\n' + format_size_breakdown(rep))
            (out_dir / 'size_report.json').write_text(json.dumps(rep, indent=2))
        return

    start_iter = int(resume_state['iter']) if resume_state is not None else 0
    # The loader order is seeded too. On resume the stream is re-seeded with
    # seed+start_iter: training continues exactly from the saved weights and
    # optimizer, but the sample order after the resume point is not the one the
    # uninterrupted run would have drawn.
    loader_gen = torch.Generator().manual_seed(args.seed + start_iter)
    _seed_transforms(train_ds, args.seed + start_iter)   # workers=0 path
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        persistent_workers=args.workers > 0, generator=loader_gen,
        worker_init_fn=_seed_worker if args.workers > 0 else None, **loader_kwargs,
    )
    batches = infinite_loader(train_loader)

    criterion = WFDENetLoss(aux_weight=0.5)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    if resume_state is not None:
        model.load_state_dict(resume_state['model'])
        optimizer.load_state_dict(resume_state['optimizer'])
        print(f'resumed from {args.resume} at iter {start_iter}')

    metrics_csv = out_dir / 'train_log.csv'
    if resume_state is None or not metrics_csv.exists():
        with open(metrics_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['iter', 'lr', 'loss', 'loss_main', 'loss_aux0', 'loss_aux1'])

    model.train()
    running, t0 = 0.0, time.time()
    eff_batch = args.batch_size * args.accum
    print(f'\ntraining {args.iters} iters, batch {args.batch_size}'
          f'{f" x accum {args.accum} = eff {eff_batch}" if args.accum > 1 else ""}'
          f'{" bf16-amp" if args.amp else ""}, '
          f'SGD lr={args.lr} poly^{POLY_POWER} min_lr={args.min_lr}\n')

    for it in range(start_iter + 1, args.iters + 1):
        # One iteration == one optimizer step == one point on the poly LR curve,
        # regardless of accumulation. accum micro-batches make up the effective
        # batch, so the schedule stays identical to the paper's 40k iters.
        lr = poly_lr(args.lr, it - 1, args.iters, min_lr=args.min_lr)
        for g in optimizer.param_groups:
            g['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.accum):
            batch = next(batches)
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp):
                outputs = model(images)
                loss, parts = criterion(outputs, masks)
            # bf16 has fp32 dynamic range, so no GradScaler is needed.
            (loss / args.accum).backward()     # average grads over micro-batches
            step_loss += loss.item() / args.accum
        optimizer.step()

        running += step_loss

        if it % args.log_interval == 0:
            avg = running / args.log_interval
            running = 0.0
            speed = args.log_interval / (time.time() - t0)
            t0 = time.time()
            eta = (args.iters - it) / speed / 3600
            print(f'iter {it:6d}/{args.iters}  lr {lr:.5f}  loss {avg:.4f}  '
                  f'main {parts["loss_main"]:.4f}  '
                  f'{speed:.2f} it/s  eta {eta:.1f}h', flush=True)
            with open(metrics_csv, 'a', newline='') as f:
                csv.writer(f).writerow([
                    it, f'{lr:.6f}', f'{avg:.5f}',
                    f'{parts["loss_main"]:.5f}',
                    f'{parts.get("loss_aux0", float("nan")):.5f}',
                    f'{parts.get("loss_aux1", float("nan")):.5f}',
                ])

        if args.eval_interval and it % args.eval_interval == 0 and it < args.iters:
            r = evaluate(model, test_loader, device, keep_probs=False, amp=args.amp)
            print(f'  [monitor @ {it}] mAUPR {r["mAUPR"]:.2f}  '
                  f'mDice {r["mDice"]:.2f}  mIoU {r["mIoU"]:.2f}', flush=True)

        if args.ckpt_interval and it % args.ckpt_interval == 0:
            torch.save({'iter': it, 'model': model.state_dict()},
                       out_dir / f'iter_{it}.pth')

        if args.resume_interval and it % args.resume_interval == 0 and it < args.iters:
            _atomic_save({'iter': it, 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(), 'args': vars(args)},
                         out_dir / 'last.pth')

    final_ckpt = out_dir / 'final.pth'
    torch.save({'iter': args.iters, 'model': model.state_dict()}, final_ckpt)

    print(f'\n=== final model, {args.dataset} test set ({len(test_ds)} images) ===')
    try:
        results = evaluate(model, test_loader, device,
                           keep_probs=not args.full_res_eval, amp=args.amp,
                           per_image_out=out_dir / 'test_scores.npz')
    except Exception as exc:
        # The checkpoint above is already on disk, so a failure here costs the
        # scoring, never the training. Say so loudly and print the one command
        # that recovers it.
        print(f'\n!! evaluation failed ({type(exc).__name__}: {exc})')
        print(f'!! TRAINING IS SAFE -- {final_ckpt} holds the {args.iters}-iter model.')
        print(f'!! recover the scores with:\n'
              f'     python {Path(__file__).name} --dataset {args.dataset} --eval-only \\\n'
              f'         --ckpt {final_ckpt} --workers 0'
              f'{" --amp" if args.amp else ""}'
              f'{" --no-lfb" if args.no_lfb else ""}'
              f'{" --no-hfb" if args.no_hfb else ""}'
              f'{" --no-ccfam" if args.no_ccfam else ""} \\\n'
              f'         --out-dir {out_dir}')
        raise

    print(format_comparison(results, paper_ref, ref_label))
    (out_dir / 'test_results.json').write_text(json.dumps(results, indent=2))
    scores = dict(np.load(out_dir / 'test_scores.npz', allow_pickle=True))
    if all(f'area_gt_{c}' in scores for c in CLASSES):
        rep = size_breakdown(scores)
        print('\n' + format_size_breakdown(rep))
        (out_dir / 'size_report.json').write_text(json.dumps(rep, indent=2))
    print(f'\nsaved {final_ckpt}, test_results.json and test_scores.npz in {out_dir}')


if __name__ == '__main__':
    main()
