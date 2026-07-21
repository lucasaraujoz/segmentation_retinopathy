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
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replication.idrid import CLASSES, IDRiDDataset
from replication.loss import WFDENetLoss
from replication.metrics_mmseg import SegEvaluator, format_comparison
from replication.wfdenet_paper import build_wfdenet_paper

# Paper §4.2.2
BASE_LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
POLY_POWER = 0.9
MAX_ITERS = 40000
BATCH_SIZE = 4


def poly_lr(base_lr: float, it: int, max_iters: int, power: float = POLY_POWER) -> float:
    return base_lr * (1 - it / max_iters) ** power


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def evaluate(model, loader, device, keep_probs: bool = True) -> dict:
    model.eval()
    evaluator = SegEvaluator(CLASSES, keep_probs=keep_probs)
    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        logits = model(images)          # eval mode -> single tensor
        evaluator.update(logits, masks)
    model.train()
    return evaluator.compute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=MAX_ITERS)
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
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = IDRiDDataset('train')
    test_ds = IDRiDDataset('test')
    print(f'IDRiD: {len(train_ds)} train / {len(test_ds)} test | classes {CLASSES}')

    # spawn, not fork: forked workers deadlock against OpenCV/torch thread
    # pools (same reason train.py:258 does this).
    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,
        multiprocessing_context='spawn' if args.workers > 0 else None,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **loader_kwargs)

    model = build_wfdenet_paper(
        num_classes=len(CLASSES), pretrained=not args.no_pretrained
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'WFDENet: {n_params:,} params ({n_params / 1e6:.2f}M) | paper reports 9.51M')

    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state['model'] if 'model' in state else state)
        print(f'loaded checkpoint {args.ckpt}')

    if args.eval_only:
        results = evaluate(model, test_loader, device)
        print('\n' + format_comparison(results))
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
          f'{f" x accum {args.accum} = eff {eff_batch}" if args.accum > 1 else ""}, '
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

            outputs = model(images)
            loss, parts = criterion(outputs, masks)
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
            r = evaluate(model, test_loader, device, keep_probs=False)
            print(f'  [monitor @ {it}] mAUPR {r["mAUPR"]:.2f}  '
                  f'mDice {r["mDice"]:.2f}  mIoU {r["mIoU"]:.2f}', flush=True)

        if args.ckpt_interval and it % args.ckpt_interval == 0:
            torch.save({'iter': it, 'model': model.state_dict()},
                       out_dir / f'iter_{it}.pth')

    final_ckpt = out_dir / 'final.pth'
    torch.save({'iter': args.iters, 'model': model.state_dict()}, final_ckpt)

    print('\n=== final model, IDRiD test set (27 images) ===')
    results = evaluate(model, test_loader, device)
    print(format_comparison(results))
    (out_dir / 'test_results.json').write_text(json.dumps(results, indent=2))
    print(f'\nsaved {final_ckpt} and {out_dir / "test_results.json"}')


if __name__ == '__main__':
    main()
