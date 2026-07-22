"""IDRiD segmentation dataset, following the WFDENet paper protocol.

Data layout (verified on disk at /home/lucas/datasets/idrid/A. Segmentation):
    1. Original Images/{a. Training Set,b. Testing Set}/IDRiD_NN.jpg   2848x4288
    2. All Segmentation Groundtruths/<split>/<n. Lesion>/IDRiD_NN_XX.tif

Two traps, both verified:
  * The .tif masks are PIL palette images. np.array(Image.open(p)) gives {0,1};
    cv2.IMREAD_GRAYSCALE gives {0,76}. The repo's usual bin_threshold=127
    (dataset.py:161) would silently zero every mask. We binarise with > 0.
  * Soft Exudates are sparse: only 26/54 train and 14/27 test images have an
    SE mask. A missing mask means "no lesion of this class", i.e. a zero
    channel -- not a reason to drop the image.

The official split is fixed: 54 train / 27 test, no validation set.
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

# OpenCV's internal thread pool deadlocks inside forked DataLoader workers.
# train.py works around the same problem with multiprocessing_context='spawn';
# disabling the pool is the cheaper fix and is safe here.
cv2.setNumThreads(0)

IDRID_ROOT = Path('/home/lucas/datasets/idrid/A. Segmentation')

# Order matters: it is the column order of Table 1 in the paper.
CLASSES: Tuple[str, ...] = ('EX', 'HE', 'SE', 'MA')

_MASK_DIRS = {
    'EX': '3. Hard Exudates',
    'HE': '2. Haemorrhages',
    'SE': '4. Soft Exudates',
    'MA': '1. Microaneurysms',
}

_SPLIT_DIRS = {
    'train': 'a. Training Set',
    'test':  'b. Testing Set',
}

# From configs/WFDENet/ours_idrid.py -- IDRiD statistics, NOT ImageNet.
IDRID_MEAN = (116.513, 56.437, 16.309)
IDRID_STD = (80.206, 41.232, 13.293)

# The paper resizes to 1440x960 (W x H); the mmseg config carries it as
# size=(960, 1440), i.e. (H, W).
CROP_H, CROP_W = 960, 1440


class IDRiDDataset(Dataset):
    def __init__(self, split: str = 'train', root: Path = IDRID_ROOT,
                 classes: Tuple[str, ...] = CLASSES, is_train: bool = None,
                 cache: bool = True):
        assert split in _SPLIT_DIRS, f'unknown split {split!r}'
        self.split = split
        self.root = Path(root)
        self.classes = classes
        self.is_train = (split == 'train') if is_train is None else is_train

        img_dir = self.root / '1. Original Images' / _SPLIT_DIRS[split]
        self.images: List[Path] = sorted(img_dir.glob('*.jpg'))
        if not self.images:
            raise FileNotFoundError(f'no .jpg images under {img_dir}')

        self.gt_dir = self.root / '2. All Segmentation Groundtruths' / _SPLIT_DIRS[split]
        self.cache = cache
        self.transform = build_transform(self.is_train, include_base_resize=not cache)
        self._cache: List[Tuple[np.ndarray, np.ndarray]] = []
        if cache:
            self._build_cache()

    def _build_cache(self) -> None:
        """Pre-decode every image/mask once, resized to the 960x1440 base scale.

        This is exactly the A.Resize step that would otherwise run per sample
        (linear for the image, nearest for the mask, as albumentations does),
        so it changes nothing numerically -- all augmentation happens after it.
        It just avoids re-decoding a 12MP JPEG plus four 12MP TIFs on every
        one of 40k iterations. Costs ~520MB RAM for the 54 training images.
        """
        base = A.Resize(CROP_H, CROP_W)
        for path in self.images:
            image, mask = self._read_raw(path)
            out = base(image=image, mask=mask)
            self._cache.append((out['image'], out['mask']))

    def __len__(self) -> int:
        return len(self.images)

    def _mask_path(self, stem: str, cls: str) -> Path:
        return self.gt_dir / _MASK_DIRS[cls] / f'{stem}_{cls}.tif'

    def _load_mask(self, stem: str, shape: Tuple[int, int]) -> np.ndarray:
        channels = []
        for cls in self.classes:
            p = self._mask_path(stem, cls)
            if p.exists():
                m = np.array(Image.open(p))
                if m.ndim == 3:
                    # Almost every GT file is palette mode, but IDRiD_81_EX.tif
                    # ships as RGBA. Its alpha channel is 255 everywhere, so a
                    # max() over all four channels marks the entire image as
                    # lesion -- which silently wrecks the aggregated EX scores
                    # (that one image was 79% of all EX ground-truth pixels in
                    # the test set). Drop alpha and keep only the colour planes.
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
        if self.cache:
            image, mask = self._cache[idx]
        else:
            image, mask = self._read_raw(img_path)

        out = self.transform(image=image, mask=mask)
        return {
            'image': out['image'],                               # [3, H, W] float
            'mask': out['mask'].permute(2, 0, 1).float(),        # [C, H, W]
            'filename': img_path.name,
        }


def build_transform(is_train: bool, include_base_resize: bool = True) -> A.Compose:
    """Paper §4.2.2: rotation (90/180/270), flipping (h/v), multi-scaling
    (0.5-2.0), then the image is resized to 1440x960.

    Read literally, a fixed resize after random scaling would cancel the
    scaling out. The authors' data_preprocessor carries size=(960,1440) with
    pad_val/seg_pad_val, which only makes sense with the standard
    mmseg/M2MRF recipe, equivalent to mmseg's
        Resize(scale=(1440,960), ratio_range=(0.5,2.0)) -> RandomCrop(960,1440)
    Note the random scale is applied to the *1440x960 base scale*, not to the
    native 2848x4288 image -- otherwise every crop would be a native-resolution
    close-up, which contradicts "resized to 1440x960".

    Normalisation runs on the raw 0-255 values (max_pixel_value=1.0), matching
    mmseg's SegDataPreProcessor, whose mean/std are on the 0-255 scale.
    """
    normalize = A.Normalize(mean=IDRID_MEAN, std=IDRID_STD, max_pixel_value=1.0)
    # Skipped when the dataset caches its samples already at the base scale.
    base = [A.Resize(CROP_H, CROP_W)] if include_base_resize else []

    if not is_train:
        return A.Compose([*base, normalize, ToTensorV2()])

    return A.Compose([
        *base,                                          # base scale 1440x960
        A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),  # ratio_range 0.5-2.0
        # Rotation/flips come before the crop: a 90/270 rotation transposes the
        # image, and the crop is the only step that guarantees a fixed
        # 960x1440 output for batch collation.
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.PadIfNeeded(min_height=CROP_H, min_width=CROP_W,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.RandomCrop(height=CROP_H, width=CROP_W),
        normalize,
        ToTensorV2(),
    ])


def inspect() -> None:
    """Sanity report over both splits: counts, per-class presence, value ranges."""
    for split in ('train', 'test'):
        ds = IDRiDDataset(split=split)
        print(f'\n=== {split}: {len(ds)} images ===')

        present = {c: 0 for c in CLASSES}
        for p in ds.images:
            for c in CLASSES:
                if ds._mask_path(p.stem, c).exists():
                    present[c] += 1
        print('  masks on disk:', {c: f'{n}/{len(ds)}' for c, n in present.items()})

        sample = ds[0]
        img, mask = sample['image'], sample['mask']
        print(f'  sample {sample["filename"]}: image {tuple(img.shape)} '
              f'[{img.min():.2f}, {img.max():.2f}]  mask {tuple(mask.shape)}')
        print('  positive pixels per class:',
              {c: int(mask[i].sum()) for i, c in enumerate(CLASSES)})

        # aggregate lesion prevalence, useful to sanity-check class imbalance
        totals = np.zeros(len(CLASSES))
        for i in range(len(ds)):
            m = ds[i]['mask']
            totals += m.flatten(1).sum(1).numpy()
        frac = totals / (len(ds) * CROP_H * CROP_W)
        print('  mean positive fraction:',
              {c: f'{frac[i]:.5f}' for i, c in enumerate(CLASSES)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        inspect()
