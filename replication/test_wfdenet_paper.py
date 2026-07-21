"""Structural fidelity checks for the WFDENet replication.

Run this before burning GPU hours. The parameter-count check is the strongest
single signal that the port matches the paper: any missing module, wrong
channel width or dropped conv shifts it immediately.

    python replication/test_wfdenet_paper.py
"""

import sys
from pathlib import Path

import cv2  # noqa: F401  -- must precede torch in this env
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replication.idrid import CLASSES, IDRiDDataset
from replication.loss import WFDENetLoss
from replication.metrics_mmseg import SegEvaluator
from replication.wfdenet_paper import DWT, IDWT, build_wfdenet_paper

PAPER_PARAMS = 9.51e6
failures = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'  [{"OK " if ok else "FAIL"}] {name}{" -- " + detail if detail else ""}')
    if not ok:
        failures.append(name)


print('1. parameter count')
model = build_wfdenet_paper(num_classes=4, pretrained=False)
n = sum(p.numel() for p in model.parameters())
rel = abs(n - PAPER_PARAMS) / PAPER_PARAMS
check('params match paper Table 4', rel < 0.02,
      f'{n:,} ({n / 1e6:.2f}M) vs 9.51M, off by {rel * 100:.2f}%')

print('\n2. DWT/IDWT round-trip (validates the /2 Haar scaling)')
x = torch.randn(2, 8, 32, 48)
ll, hi = DWT()(x)
rec = IDWT()(torch.cat([ll, hi], dim=1))
err = (rec - x).abs().max().item()
check('perfect reconstruction', err < 1e-5, f'max abs err {err:.2e}')
check('subband shapes', ll.shape == (2, 8, 16, 24) and hi.shape == (2, 24, 16, 24),
      f'll {tuple(ll.shape)} hi {tuple(hi.shape)}')

print('\n3. forward pass (eval)')
model.eval()
with torch.no_grad():
    y = model(torch.randn(2, 3, 256, 384))
check('eval returns a single tensor', isinstance(y, torch.Tensor))
check('output shape', tuple(y.shape) == (2, 4, 256, 384), str(tuple(y.shape)))
check('output finite', bool(torch.isfinite(y).all()))

print('\n4. forward/backward (train, deep supervision)')
model.train()
out = model(torch.randn(2, 3, 256, 384))
check('train returns 3 maps (sd, hfb, lfb)', isinstance(out, tuple) and len(out) == 3,
      f'got {type(out).__name__} of len {len(out) if isinstance(out, tuple) else "n/a"}')
check('aux maps at input resolution',
      all(tuple(o.shape) == (2, 4, 256, 384) for o in out))

target = (torch.rand(2, 4, 256, 384) > 0.9).float()
loss, parts = WFDENetLoss(aux_weight=0.5)(out, target)
loss.backward()
check('loss finite', bool(torch.isfinite(loss)), f'{loss.item():.4f}')
n_grad = sum(1 for p in model.parameters() if p.grad is not None)
n_bad = sum(1 for p in model.parameters()
            if p.grad is not None and not torch.isfinite(p.grad).all())
check('all grads finite', n_bad == 0, f'{n_grad} tensors with grads, {n_bad} non-finite')

print('\n5. production input size 960x1440')
# Level 4 lands at 30x45 -- odd width -- so this genuinely exercises
# pad_to_even/unpad, which is why the authors pad that level and only that one.
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval().to(dev)
with torch.no_grad():
    y_full = model(torch.randn(1, 3, 960, 1440, device=dev))
check('960x1440 forward', tuple(y_full.shape) == (1, 4, 960, 1440), str(tuple(y_full.shape)))
check('output finite at full size', bool(torch.isfinite(y_full).all()))
model.to('cpu')

try:
    model(torch.randn(1, 3, 250, 370))
    check('rejects non-multiple-of-32 input', False, 'no error raised')
except ValueError:
    check('rejects non-multiple-of-32 input', True)

print('\n6. real IDRiD batch')
try:
    ds = IDRiDDataset('train')
    sample = ds[0]
    check('image tensor', tuple(sample['image'].shape) == (3, 960, 1440),
          str(tuple(sample['image'].shape)))
    check('mask tensor', tuple(sample['mask'].shape) == (4, 960, 1440),
          str(tuple(sample['mask'].shape)))
    check('mask is binary', set(torch.unique(sample['mask']).tolist()) <= {0.0, 1.0})

    # every class must be non-empty somewhere in the training set
    seen = torch.zeros(4)
    for i in range(len(ds)):
        seen += (ds[i]['mask'].flatten(1).sum(1) > 0).float()
    check('all 4 classes present in train split', bool((seen > 0).all()),
          ', '.join(f'{c}:{int(seen[i])}/{len(ds)}' for i, c in enumerate(CLASSES)))
except FileNotFoundError as e:
    check('IDRiD data available', False, str(e))

print('\n7. evaluator sanity (perfect and inverted predictions)')
gt = (torch.rand(3, 4, 64, 64) > 0.8).float()
ev = SegEvaluator(CLASSES, keep_probs=False)
ev.update(torch.where(gt > 0, 10.0, -10.0), gt)      # near-perfect logits
res = ev.compute()
check('perfect prediction -> Dice ~100', abs(res['mDice'] - 100) < 0.5,
      f'mDice {res["mDice"]:.2f}')
check('perfect prediction -> IoU ~100', abs(res['mIoU'] - 100) < 0.5,
      f'mIoU {res["mIoU"]:.2f}')

ev = SegEvaluator(CLASSES, keep_probs=False)
ev.update(torch.where(gt > 0, -10.0, 10.0), gt)      # inverted
res_bad = ev.compute()
check('inverted prediction -> Dice ~0', res_bad['mDice'] < 1.0,
      f'mDice {res_bad["mDice"]:.2f}')

print('\n' + '=' * 50)
if failures:
    print(f'{len(failures)} CHECK(S) FAILED: {", ".join(failures)}')
    sys.exit(1)
print('all checks passed')
sys.exit(0)
