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
import sys
import time
from pathlib import Path

import cv2  # noqa: F401  -- must precede torch (CXXABI clash in this env)
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replication.ddr import DDRDataset
from replication.idrid import CLASSES, IDRiDDataset
from replication.loss import WFDENetLoss
from replication.metrics_mmseg import PAPER_TARGETS, SegEvaluator, format_comparison
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


def poly_lr(base_lr: float, it: int, max_iters: int, power: float = POLY_POWER) -> float:
    return base_lr * (1 - it / max_iters) ** power


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


@torch.no_grad()
def evaluate(model, loader, device, keep_probs: bool = True, amp: bool = False,
             per_image_out: Path = None) -> dict:
    model.eval()
    evaluator = SegEvaluator(CLASSES, keep_probs=keep_probs)
    filenames = []
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(images)      # eval mode -> single tensor
        logits = logits.float()
        evaluator.update(logits, masks)
        filenames.extend(batch['filename'])
    model.train()

    if per_image_out is not None:
        # Per-image Dice for paired comparisons between runs (compare_runs.py).
        # Same test_scores.npz convention the FGADR series uses.
        per_img = evaluator.per_image_dice()
        if per_img:
            np.savez(per_image_out, filenames=np.array(filenames), **per_img)

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
                        help="timm variant. Default matches the authors' backbone "
                             '(TF SAME padding + BN eps 1e-3). Use '
                             "'efficientnet_b1' to reproduce the earlier runs.")
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
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    if args.iters is None:
        args.iters = spec['iters']
    paper_ref = PAPER_TARGETS[args.dataset]

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'idrid':
        train_ds = IDRiDDataset('train', photometric=args.photometric)
        test_ds = IDRiDDataset('test')      # test pipeline never augments
    else:
        if args.photometric:
            parser.error('--photometric is only wired for idrid')
        train_ds = DDRDataset('train')
        test_ds = DDRDataset('test')        # Table 6 evaluates on test, not valid
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
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **loader_kwargs)

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

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state['model'] if 'model' in state else state)
        print(f'loaded checkpoint {args.ckpt}')

    if args.eval_only:
        results = evaluate(model, test_loader, device, amp=args.amp,
                           per_image_out=out_dir / 'test_scores.npz')
        print('\n' + format_comparison(results, paper_ref))
        (out_dir / 'test_results.json').write_text(json.dumps(results, indent=2))
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        persistent_workers=args.workers > 0, **loader_kwargs,
    )
    batches = infinite_loader(train_loader)

    criterion = WFDENetLoss(aux_weight=0.5)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    metrics_csv = out_dir / 'train_log.csv'
    with open(metrics_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['iter', 'lr', 'loss', 'loss_main', 'loss_aux0', 'loss_aux1'])

    model.train()
    running, t0 = 0.0, time.time()
    eff_batch = args.batch_size * args.accum
    print(f'\ntraining {args.iters} iters, batch {args.batch_size}'
          f'{f" x accum {args.accum} = eff {eff_batch}" if args.accum > 1 else ""}'
          f'{" bf16-amp" if args.amp else ""}, '
          f'SGD lr={args.lr} poly^{POLY_POWER}\n')

    for it in range(1, args.iters + 1):
        # One iteration == one optimizer step == one point on the poly LR curve,
        # regardless of accumulation. accum micro-batches make up the effective
        # batch, so the schedule stays identical to the paper's 40k iters.
        lr = poly_lr(args.lr, it - 1, args.iters)
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

    final_ckpt = out_dir / 'final.pth'
    torch.save({'iter': args.iters, 'model': model.state_dict()}, final_ckpt)

    print(f'\n=== final model, {args.dataset} test set ({len(test_ds)} images) ===')
    results = evaluate(model, test_loader, device, amp=args.amp,
                       per_image_out=out_dir / 'test_scores.npz')
    print(format_comparison(results, paper_ref))
    (out_dir / 'test_results.json').write_text(json.dumps(results, indent=2))
    print(f'\nsaved {final_ckpt}, test_results.json and test_scores.npz in {out_dir}')


if __name__ == '__main__':
    main()
