"""
Dump per-image test predictions to a portable .npz, for offline detection-metric
and statistical analysis (compare_experiments.py) without re-running training.

Ensembles all fold checkpoints found in the experiment dir (matches train.py's test
eval) and optionally applies TTA. Predictions and GT are at model resolution.

Usage:
    python dump_preds.py --exp H2L2 --suffixes 3        # same split as training
    python dump_preds.py --exp H0   --suffixes 3 --no-tta

Writes: outputs/exp_<id>_<name>/test_preds.npz  with arrays
    probs [N,H,W] float16, gts [N,H,W] uint8, files [N] str
Requires torch (run on the training machine).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import EXPERIMENTS
from dataset import build_dataset, load_dataframe, create_splits, filter_by_suffix
from models import build_model
from train import _tta_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, help='Experiment ID from config.EXPERIMENTS')
    ap.add_argument('--suffixes', default=None, help='Same --suffixes used in training (e.g. "3")')
    ap.add_argument('--class', dest='klass', default='Hemorrhage',
                    help='Which class channel to dump (name from config.classes). '
                         'For multilabel exps pick e.g. Hemorrhage or HardExudate.')
    ap.add_argument('--device', default=None)
    ap.add_argument('--no-tta', action='store_true')
    args = ap.parse_args()

    if args.exp not in EXPERIMENTS:
        print(f'Unknown experiment "{args.exp}". Available: {list(EXPERIMENTS.keys())}')
        sys.exit(1)

    config = EXPERIMENTS[args.exp]
    if args.suffixes:
        config.allowed_suffixes = tuple(s.strip() for s in args.suffixes.split(','))

    if args.klass not in config.classes:
        print(f'Class "{args.klass}" not in exp {args.exp} classes {config.classes}. '
              f'Use --class with one of them.')
        sys.exit(1)
    ch = list(config.classes).index(args.klass)   # channel index for that class
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    use_tta = not args.no_tta

    # Same data + split as training (create_splits reuses the cached json)
    df = load_dataframe(config)
    df = filter_by_suffix(df, config.allowed_suffixes)
    suffix_tag = ''.join(sorted(config.allowed_suffixes)) if config.allowed_suffixes else 'all'
    splits = create_splits(df, config, str(Path(config.output_dir) / f'cv_splits_{suffix_tag}.json'))
    test_df = df.iloc[splits['test']].reset_index(drop=True)

    test_ds = build_dataset(config, test_df, is_train=False)
    loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    ckpts = sorted(config.exp_dir.glob('model_best_fold*.pth'))
    if not ckpts:
        print(f'No checkpoints in {config.exp_dir}/ (train the experiment first).')
        sys.exit(1)
    print(f'Ensembling {len(ckpts)} checkpoint(s); TTA={use_tta}; {len(test_df)} test images; '
          f'class={args.klass} (channel {ch})')

    # Accumulate ensembled probabilities across folds
    probs_sum, gts, files = None, [], []
    for ci, ckpt_path in enumerate(ckpts):
        model = build_model(config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        batch_probs = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch['image'].to(device)
                logits = _tta_predict(model, imgs) if use_tta else model(imgs)
                p = torch.sigmoid(logits)[:, ch].cpu().numpy()   # selected class channel
                batch_probs.append(p)
                if ci == 0:
                    gts.append(batch['mask'][:, ch].numpy().astype(np.uint8))
                    files.extend(batch['filename'])
        fold_probs = np.concatenate(batch_probs, axis=0)
        probs_sum = fold_probs if probs_sum is None else probs_sum + fold_probs

    probs = (probs_sum / len(ckpts)).astype(np.float16)
    gts = np.concatenate(gts, axis=0).astype(np.uint8)
    files = np.array(files, dtype=object)

    out = config.exp_dir / f'test_preds_{args.klass}.npz'
    np.savez_compressed(out, probs=probs, gts=gts, files=files, cls=args.klass)
    print(f'Saved {probs.shape[0]} predictions → {out}  '
          f'(probs {probs.shape} {probs.dtype}, gts {gts.shape}, class={args.klass})')


if __name__ == '__main__':
    main()
