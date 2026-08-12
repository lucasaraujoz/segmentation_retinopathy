"""Is replication/wfdenet_paper.py numerically identical to the authors' head?

This is the test that settles the audit. Instead of reading the two
implementations side by side and arguing, we instantiate BOTH, copy the weights
from theirs into ours, feed the same tensor, and compare the outputs.

    ours == theirs  =>  our port is correct, and the ablation result
                        (+1.0 mDice) is a property of the method, not a bug.
    ours != theirs  =>  the per-module report below says exactly where.

The authors' code is imported from replication_2/upstream/ (git clone, pinned
at 38b0b16) WITHOUT a single edit -- `git -C upstream status` must stay clean.
Everything they publish is used as-is; only the weight-name mapping lives here,
and that mapping is itself verified (every key consumed, every shape matched).

Requires: mmengine, mmcv-lite==2.1.0 (mmseg pins <2.2.0), mmpretrain.
"""

import cv2  # noqa: F401  -- must precede torch in this env (libstdc++ CXXABI)
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM = ROOT / 'replication_2' / 'upstream'
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT))

from mmseg.models.decode_heads.wfdenet_head import WFDEHead  # noqa: E402  (theirs)
from replication.wfdenet_paper import build_wfdenet_paper, resize  # noqa: E402  (ours)

# From configs/_base_/models/wfdenet_b1.py. SyncBN -> BN because this runs on a
# single process; they are the same op with one GPU (and identical parameters).
HEAD_CFG = dict(
    in_channels=[16, 24, 40, 112, 1280],
    in_index=(0, 1, 2, 3, 4),
    channels=64,
    dropout_ratio=0,
    norm_cfg=dict(type='BN', requires_grad=True),
    align_corners=False,
    num_classes=4,
    loss_decode=dict(type='Loss', loss_name='bdice', use_sigmoid=True,
                     loss_weight=1.0, eps=1e-5),
)

# Small enough to be instant, and deliberately chosen so that level 4 comes out
# 2x3 -- odd width -- which exercises pad_to_even/unpad. A size that never pads
# would leave that branch untested.
INPUT_HW = (64, 96)
BATCH = 2          # their head only emits the aux outputs when batch != 1


class _FixedFeats(nn.Module):
    """Stands in for the backbone so both models see identical features.

    Lets us compare the head alone without touching replication/wfdenet_paper.py:
    its forward calls self.backbone(x) and proceeds, so swapping this in isolates
    exactly the code under test.
    """

    def __init__(self, feats):
        super().__init__()
        self._feats = list(feats)      # a plain list: stays out of state_dict

    def forward(self, x):
        return self._feats


def our_key_to_theirs(key: str) -> str:
    """Translate a key of our state_dict into the authors' naming.

    Two systematic differences, both cosmetic:
      * they number sibling modules by suffix (nf0, CCFAM0); we use ModuleLists.
      * mmcv ConvModule nests as `.conv`/`.bn`; our nn.Sequential as `.0`/`.1`.
    """
    k = key
    k = re.sub(r'^nf\.(\d)\.', r'nf\1.', k)
    k = re.sub(r'^ccfam\.(\d)\.', r'CCFAM\1.', k)

    # lateral_convs.i = Sequential(1x1 bare conv, conv_bn_relu)
    k = re.sub(r'^lateral_convs\.(\d)\.0\.(weight|bias)$',
               r'lateral_convs.\1.0.conv.\2', k)
    k = re.sub(r'^lateral_convs\.(\d)\.1\.0\.', r'lateral_convs.\1.1.conv.', k)
    k = re.sub(r'^lateral_convs\.(\d)\.1\.1\.', r'lateral_convs.\1.1.bn.', k)

    # FPNHead.scale_heads.i = Sequential(conv_bn_relu) vs Sequential(ConvModule)
    k = re.sub(r'^(fpn_\w+)\.scale_heads\.(\d)\.0\.', r'\1.scale_heads.\2.0.conv.', k)
    k = re.sub(r'^(fpn_\w+)\.scale_heads\.(\d)\.1\.', r'\1.scale_heads.\2.0.bn.', k)

    # NeighborFuse.fuse, same ConvModule-vs-Sequential difference
    k = re.sub(r'^(nf\d)\.fuse\.0\.', r'\1.fuse.0.conv.', k)
    k = re.sub(r'^(nf\d)\.fuse\.1\.', r'\1.fuse.0.bn.', k)
    return k


def copy_weights(ours: nn.Module, theirs: nn.Module) -> None:
    """Load the authors' weights into our model, asserting the mapping is total."""
    theirs_sd = theirs.state_dict()
    ours_sd = ours.state_dict()

    new_sd, missing, mismatched = {}, [], []
    for k, v in ours_sd.items():
        tk = our_key_to_theirs(k)
        if tk not in theirs_sd:
            missing.append((k, tk))
            continue
        if theirs_sd[tk].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tk, tuple(theirs_sd[tk].shape)))
            continue
        new_sd[k] = theirs_sd[tk]

    if missing or mismatched:
        for k, tk in missing:
            print(f'  NO MATCH   ours={k}  ->  theirs={tk} (absent)')
        for k, s, tk, ts in mismatched:
            print(f'  BAD SHAPE  ours={k}{s}  vs  theirs={tk}{ts}')
        raise SystemExit('weight mapping is incomplete -- the models differ structurally')

    unused = set(theirs_sd) - {our_key_to_theirs(k) for k in ours_sd}
    if unused:
        # A parameter they have and we never built is a missing module on our side.
        for tk in sorted(unused):
            print(f'  UNUSED     theirs={tk} has no counterpart in ours')
        raise SystemExit('our model is missing parameters the authors have')

    ours.load_state_dict(new_sd, strict=True)
    print(f'  mapped {len(new_sd)} tensors, all shapes agree, none left over')


def main() -> None:
    torch.manual_seed(0)

    print('=' * 68)
    print('WFDENet: our port vs. the authors\' released head')
    print('=' * 68)

    theirs = WFDEHead(**HEAD_CFG).eval()
    ours = build_wfdenet_paper(pretrained=False).eval()

    n_theirs = sum(p.numel() for p in theirs.parameters())
    n_ours_head = sum(p.numel() for p in ours.parameters()) - \
        sum(p.numel() for p in ours.backbone.parameters())
    print(f'\nparameters (head only): theirs {n_theirs:,} | ours {n_ours_head:,} '
          f'| {"MATCH" if n_theirs == n_ours_head else "DIFFER"}')

    H, W = INPUT_HW
    feats = [
        torch.randn(BATCH, 16, H // 2, W // 2),
        torch.randn(BATCH, 24, H // 4, W // 4),
        torch.randn(BATCH, 40, H // 8, W // 8),
        torch.randn(BATCH, 112, H // 16, W // 16),
        torch.randn(BATCH, 1280, H // 32, W // 32),
    ]
    print(f'feature sizes: {[tuple(f.shape[1:]) for f in feats]}')
    print(f'level 4 is {tuple(feats[4].shape[2:])} -> pad_to_even path exercised')

    ours.backbone = _FixedFeats(feats)

    print('\ncopying their weights into our model:')
    copy_weights(ours, theirs)

    # train() so both emit the auxiliary heads. BN then uses batch statistics --
    # deterministic here, since both models see the same input and the same
    # weights, and we never step the optimizer.
    ours.train()
    theirs.train()
    dummy = torch.zeros(BATCH, 3, H, W)

    with torch.no_grad():
        out_ours = ours(dummy)
        out_theirs = theirs(feats)

    print(f'\noutputs: ours {len(out_ours)} tensors, theirs {len(out_theirs)} tensors')

    names = ['main (SD)', 'aux (HFB)', 'aux (LFB)']
    worst = 0.0
    print(f'\n{"output":<12}{"shape":<20}{"max |diff|":>14}{"rel":>12}')
    print('-' * 58)
    for name, a, b in zip(names, out_ours, out_theirs):
        # Their head returns logits at native resolution; ours upsamples to the
        # input size inside forward. Same op, applied here for a like-for-like
        # comparison.
        b_up = resize(b, size=a.shape[2:], align_corners=False)
        diff = (a - b_up).abs().max().item()
        denom = b_up.abs().max().item()
        worst = max(worst, diff)
        print(f'{name:<12}{str(tuple(a.shape)):<20}{diff:>14.3e}'
              f'{diff / max(denom, 1e-12):>12.3e}')

    print('-' * 58)
    tol = 1e-5
    verdict = 'IDENTICAL' if worst < tol else 'DIVERGENT'
    print(f'worst max|diff| = {worst:.3e}   (tolerance {tol:.0e})   =>  {verdict}')

    check_loss()

    if worst >= tol:
        raise SystemExit(1)


def check_loss() -> None:
    """Compare our loss against theirs on identical inputs.

    The config passes eps=1e-5, overriding the 1e-3 default in the function
    signature -- which is why our loss.py uses 1e-5.
    """
    print('\n' + '=' * 68)
    print('loss: replication/loss.py vs. their mmseg/models/losses/loss.py')
    print('=' * 68)

    from mmseg.models.losses.loss import binary_dice_loss  # theirs
    from replication.loss import binary_dice_loss as ours_bdice

    torch.manual_seed(1)
    logits = torch.randn(2, 4, 32, 32)
    target = (torch.rand(2, 4, 32, 32) > 0.9).float()
    # A class that is absent everywhere is the case where eps actually decides
    # the value, so make one channel empty on purpose.
    target[:, 2] = 0

    mine = ours_bdice(logits, target, eps=1e-5)
    theirs_val = binary_dice_loss(logits.sigmoid(), target, weight=None, eps=1e-5)
    print(f'  ours   {mine.item():.10f}')
    print(f'  theirs {theirs_val.item():.10f}')
    print(f'  diff   {abs(mine.item() - theirs_val.item()):.3e}')

    # What the config would have produced with the signature default instead.
    alt = binary_dice_loss(logits.sigmoid(), target, weight=None, eps=1e-3)
    print(f'\n  (their default eps=1e-3 would give {alt.item():.10f} -- '
          f'the config overrides it to 1e-5)')

    try:
        from mmseg.models.losses.loss import Loss
        Loss(loss_name='bdice', use_sigmoid=True, eps=1e-5)(logits, target)
        print('\n  Loss(loss_name="bdice") runs.')
    except TypeError as exc:
        print(f'\n  NOTE: Loss(loss_name="bdice") raises TypeError: {exc}')
        print('  Their Loss.forward passes naive=... but binary_dice_loss has no')
        print('  such parameter, so the released wrapper cannot run for bdice.')
        print('  Consistent with an inference-only release: the training path is')
        print('  not exercised by their test.py. We call binary_dice_loss directly.')


if __name__ == '__main__':
    main()
