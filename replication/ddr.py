"""DDR lesion-segmentation dataset, following the WFDENet paper protocol.

DDR is the dataset the authors ran their ablations on (paper §4.5), so this is
what confronts their Table 6 claim directly. It also has 7x more training and
8x more test images than IDRiD, which is what makes the ablation interpretable.

Data layout (verified on disk at /home/lucas/datasets/ddr/DDR-dataset):
    lesion_segmentation/{train,valid,test}/image/<stem>.jpg
    lesion_segmentation/{train,test}/label/{EX,HE,MA,SE}/<stem>.tif
    lesion_segmentation/valid/'segmentation label'/{EX,HE,MA,SE}/<stem>.tif
                                ^ note the space -- valid differs from train/test

Counts match the paper exactly: 383 train / 149 valid / 225 test.

Unlike IDRiD, the masks here are clean: PIL mode 'L' with values {0, 255}, all
four present for every image (some empty). No palette/RGBA trap.

The big structural difference from IDRiD: images come in 27 distinct resolutions
(aspect 0.995-1.765), so the multi-scale factor cannot be precomputed -- see
build_transform.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from replication.idrid import _PAD_FILL  # albumentations 1.x/2.x fill-arg compat

# OpenCV's thread pool deadlocks inside forked DataLoader workers.
cv2.setNumThreads(0)

DDR_ROOT = Path('/home/lucas/datasets/ddr/DDR-dataset/lesion_segmentation')

# Same order as IDRiD and as the paper's Table 1/2 columns -- NOT the
# alphabetical order the label directories happen to be in (EX HE MA SE).
CLASSES: Tuple[str, ...] = ('EX', 'HE', 'SE', 'MA')

# train/test store labels under 'label'; valid under 'segmentation label'.
_LABEL_DIRS = {
    'train': 'label',
    'valid': 'segmentation label',
    'test': 'label',
}

# From configs/WFDENet/ours_ddr.py -- DDR statistics, on the 0-255 scale.
DDR_MEAN = (81.205, 50.636, 21.216)
DDR_STD = (76.252, 48.798, 21.625)

# Paper §4.2.2: DDR is trained at 1024x1024 (square, divisible by 32).
CROP = 1024
RATIO_RANGE = (0.5, 2.0)

# mmseg's Resize(img_scale=(1024,1024), ratio_range=(0.5,2.0), keep_ratio=True)
# samples r ~ U(0.5, 2), builds scale=(int(1024r), int(1024r)) and calls
# mmcv.imrescale, whose factor is
#     min(max(scale)/max(h,w), min(scale)/min(h,w)) = int(1024r)/max(h,w)
# i.e. the LONGEST side becomes int(1024r), aspect preserved. Sampling the
# longest side uniformly over these integers is equivalent to uniform r.
_LONGEST_SIDES = list(range(int(CROP * RATIO_RANGE[0]), int(CROP * RATIO_RANGE[1]) + 1))


class DDRDataset(Dataset):
    def __init__(self, split: str = 'train', root: Path = DDR_ROOT,
                 classes: Tuple[str, ...] = CLASSES, is_train: bool = None):
        assert split in _LABEL_DIRS, f'unknown split {split!r}'
        self.split = split
        self.root = Path(root)
        self.classes = classes
        self.is_train = (split == 'train') if is_train is None else is_train

        img_dir = self.root / split / 'image'
        self.images: List[Path] = sorted(img_dir.glob('*.jpg'))
        if not self.images:
            raise FileNotFoundError(f'no .jpg images under {img_dir}')

        self.gt_dir = self.root / split / _LABEL_DIRS[split]
        self.transform = build_transform(self.is_train)
        # No cache: measured IO is ~0.026 s/sample (1 jpg + 4 tif) against a
        # ~1.1 s GPU step, so workers hide it entirely. Reading native every
        # time also means the multi-scale rescales from the ORIGINAL image,
        # exactly like mmcv -- no double-resize approximation.

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path(self, stem: str, cls: str) -> Path:
        return self.gt_dir / cls / f'{stem}.tif'

    def _load_mask(self, stem: str, shape: Tuple[int, int]) -> np.ndarray:
        channels = []
        for cls in self.classes:
            p = self._mask_path(stem, cls)
            if p.exists():
                m = np.array(Image.open(p))
                if m.ndim == 3:                       # defensive; DDR is mode 'L'
                    m = m[..., :3].max(axis=-1)
                channels.append((m > 0).astype(np.uint8))
            else:
                channels.append(np.zeros(shape, dtype=np.uint8))
        return np.stack(channels, axis=-1)            # [H, W, C]

    def _read_raw(self, img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'failed to read {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, self._load_mask(img_path.stem, image.shape[:2])

    def __getitem__(self, idx: int) -> dict:
        img_path = self.images[idx]
        image, mask = self._read_raw(img_path)
        out = self.transform(image=image, mask=mask)
        return {
            'image': out['image'],                               # [3, H, W] float
            'mask': out['mask'].permute(2, 0, 1).float(),        # [C, H, W]
            'filename': img_path.name,
        }


def build_transform(is_train: bool) -> A.Compose:
    """Replicates the M2MRF/mmseg DDR train_pipeline.

    Reference (M2MRF configs/_base_/datasets/ddr.py):
        Resize(img_scale=(1024,1024), ratio_range=(0.5,2.0))
        RandomCrop(crop_size=(1024,1024), cat_max_ratio=0.75)
        RandomFlip(flip_ratio=0)          # flips/rotations are OFFLINE there
        PhotoMetricDistortion()
        Normalize(**img_norm_cfg)
        Pad(size=(1024,1024), pad_val=0, seg_pad_val=0)

    Two deliberate differences from replication/idrid.py, both toward mmseg:

    * LongestMaxSize over the NATIVE image reproduces mmcv.imrescale exactly.
      IDRiD could precompute a fixed scale factor because every image is
      2848x4288; DDR has 27 distinct resolutions, so the factor is per-image.
    * Normalize BEFORE Pad, as mmseg does, so the border is 0 in normalised
      space = the dataset mean colour. This matters far more on DDR than on
      IDRiD: with aspect ~1.5 the 1024 crop only fits without padding when
      r >= 1.5, so roughly two thirds of training samples are padded.

    PhotoMetricDistortion is omitted: paper §4.2.2 enumerates exactly three
    augmentations (rotation, flipping, multi-scaling) and none is photometric.
    cat_max_ratio is omitted too -- with lesions under a few percent of the
    frame the background always exceeds 75%, so mmseg falls back to a plain
    random crop after 10 tries.

    mmseg does RandomCrop then Pad; albumentations' RandomCrop requires the
    image to be at least the crop size, so we pad first with position
    'top_left', which places the retina in the same corner mmseg leaves it.
    """
    normalize = A.Normalize(mean=DDR_MEAN, std=DDR_STD, max_pixel_value=1.0)

    if not is_train:
        # Official test pipeline: Resize(scale, keep_ratio=False) -> 1024x1024.
        return A.Compose([A.Resize(CROP, CROP), normalize, ToTensorV2()])

    return A.Compose([
        A.LongestMaxSize(max_size=_LONGEST_SIDES,
                         interpolation=cv2.INTER_LINEAR, p=1.0),
        # Rotation/flips before the crop: 90/270 transposes the image, and the
        # crop is what pins the output to a fixed 1024x1024 for collation.
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        normalize,                                    # BEFORE pad (mmseg order)
        A.PadIfNeeded(min_height=CROP, min_width=CROP, position='top_left',
                      border_mode=cv2.BORDER_CONSTANT, **_PAD_FILL),
        A.RandomCrop(height=CROP, width=CROP),
        ToTensorV2(),
    ])


def inspect() -> None:
    """Sanity report: counts, mask emptiness, and the multi-scale size spread."""
    for split in ('train', 'test'):
        ds = DDRDataset(split=split)
        print(f'\n=== {split}: {len(ds)} images ===')

        nonempty = {c: 0 for c in CLASSES}
        for p in ds.images:
            for c in CLASSES:
                q = ds._mask_path(p.stem, c)
                if q.exists() and (np.array(Image.open(q)) > 0).any():
                    nonempty[c] += 1
        print('  masks non-empty:', {c: f'{n}/{len(ds)}' for c, n in nonempty.items()})

        sample = ds[0]
        img, mask = sample['image'], sample['mask']
        print(f'  sample {sample["filename"]}: image {tuple(img.shape)} '
              f'[{img.min():.2f}, {img.max():.2f}]  mask {tuple(mask.shape)}')
        print(f'  mask values: {sorted(set(mask.unique().tolist()))}')

        shapes = {tuple(ds[i]['image'].shape) for i in range(min(12, len(ds)))}
        print(f'  output shapes over 12 samples: {shapes}')

        if split == 'train':
            # How often does padding kick in? (long side < CROP after rescale)
            raw = [cv2.imread(str(p)).shape[:2] for p in ds.images[:60]]
            pads = sum(1 for h, w in raw
                       for s in [np.mean(_LONGEST_SIDES)]
                       if min(h, w) * (s / max(h, w)) < CROP)
            print(f'  padded at the mean scale: ~{pads}/60 samples '
                  f'(expected: aspect>1 needs r>=aspect to avoid padding)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        inspect()
