"""How much do the lesion masks overlap, and does the M2MRF flattening matter?

M2MRF (and therefore WFDENet, which reuses its data prep) stores the ground
truth as ONE channel with values 0-4, built by tools/prepare_labels.py:

    for i, c in enumerate(['EX', 'HE', 'SE', 'MA']):
        ann[label > 0] = i + 1        # later class overwrites earlier

So a pixel claimed by two lesion files ends up assigned to the *highest-index*
class only (MA > SE > HE > EX). Both their loss (_make_one_hot, a scatter_ over
that single channel) and their evaluator (`label = raw_label == i`) consume it
that way.

Our datasets build four independent binary masks from the four files, so an
overlapping pixel counts in both classes. That makes our per-class positives a
superset of theirs, which shifts every absolute metric while cancelling in the
ablation delta.

This script measures the size of that effect before we act on it.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ('EX', 'HE', 'SE', 'MA')      # prepare_labels.py order: index 4 wins

DDR_ROOT = Path('/home/lucas/datasets/ddr/DDR-dataset/lesion_segmentation')
IDRID_ROOT = Path('/home/lucas/datasets/idrid/A. Segmentation')

IDRID_DIRS = {
    'EX': '3. Hard Exudates', 'HE': '2. Haemorrhages',
    'SE': '4. Soft Exudates', 'MA': '1. Microaneurysms',
}
IDRID_SUFFIX = {'EX': 'EX', 'HE': 'HE', 'SE': 'SE', 'MA': 'MA'}


def _read_mask(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    m = np.array(Image.open(path))
    if m.ndim == 3:                      # IDRiD_81_EX.tif ships as RGBA
        m = m[..., :3].max(axis=-1)
    return m > 0


def ddr_cases(split: str):
    img_dir = DDR_ROOT / split / 'image'
    lab_dir = DDR_ROOT / split / ('label' if split != 'valid' else 'segmentation label')
    for img in sorted(img_dir.glob('*.jpg')):
        yield img.stem, {c: lab_dir / c / f'{img.stem}.tif' for c in CLASSES}


def idrid_cases(split: str):
    sub = 'a. Training Set' if split == 'train' else 'b. Testing Set'
    img_dir = IDRID_ROOT / '1. Original Images' / sub
    gt_dir = IDRID_ROOT / '2. All Segmentation Groundtruths' / sub
    for img in sorted(img_dir.glob('*.jpg')):
        yield img.stem, {
            c: gt_dir / IDRID_DIRS[c] / f'{img.stem}_{IDRID_SUFFIX[c]}.tif'
            for c in CLASSES
        }


def report(name: str, cases) -> None:
    n_img = 0
    lesion_px = 0                        # pixels belonging to >=1 class (union)
    multi_px = 0                         # pixels belonging to >=2 classes
    per_class = Counter()                # our count (independent masks)
    per_class_flat = Counter()           # M2MRF count (after overwrite)
    pairs = Counter()

    for stem, paths in cases:
        masks = {c: _read_mask(p) for c, p in paths.items()}
        masks = {c: m for c, m in masks.items() if m is not None}
        if not masks:
            continue
        n_img += 1

        shape = next(iter(masks.values())).shape
        count = np.zeros(shape, dtype=np.uint8)
        flat = np.zeros(shape, dtype=np.uint8)   # prepare_labels.py semantics
        for i, c in enumerate(CLASSES):
            m = masks.get(c)
            if m is None:
                continue
            count += m.astype(np.uint8)
            flat[m] = i + 1
            per_class[c] += int(m.sum())

        lesion_px += int((count > 0).sum())
        multi_px += int((count > 1).sum())

        for i, c in enumerate(CLASSES):
            per_class_flat[c] += int((flat == i + 1).sum())

        if (count > 1).any():
            present = [c for c in CLASSES if masks.get(c) is not None]
            for a_i, a in enumerate(present):
                for b in present[a_i + 1:]:
                    both = int((masks[a] & masks[b]).sum())
                    if both:
                        pairs[f'{a}+{b}'] += both

    print(f'\n{"=" * 62}\n{name}  ({n_img} imagens)\n{"=" * 62}')
    if lesion_px == 0:
        print('  nenhum pixel de lesao encontrado -- checar os caminhos')
        return

    pct = 100 * multi_px / lesion_px
    print(f'  pixels de lesao (uniao) : {lesion_px:,}')
    print(f'  pixels em >=2 classes   : {multi_px:,}  ({pct:.4f}% da uniao)')

    if pairs:
        print('\n  pares que colidem:')
        for k, v in pairs.most_common():
            print(f'    {k:<8} {v:>12,}')

    print(f'\n  {"classe":<8}{"nosso GT":>14}{"GT M2MRF":>14}{"perda":>12}{"perda %":>10}')
    for c in CLASSES:
        ours, theirs = per_class[c], per_class_flat[c]
        lost = ours - theirs
        share = 100 * lost / ours if ours else 0.0
        print(f'  {c:<8}{ours:>14,}{theirs:>14,}{lost:>12,}{share:>9.3f}%')

    print(f'\n  => {"MATERIAL: vale reavaliar com GT single-label" if pct >= 0.5 else "DESPREZIVEL: a convencao de GT nao explica o offset"}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['ddr', 'idrid', 'both'], default='both')
    args = ap.parse_args()

    if args.dataset in ('ddr', 'both'):
        report('DDR / test', ddr_cases('test'))
    if args.dataset in ('idrid', 'both'):
        report('IDRiD / test', idrid_cases('test'))


if __name__ == '__main__':
    main()
