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

Augmentation = paper §4.2.2 exactly: rotation (90/180/270), flipping (h/v),
multi-scaling (0.5-2.0), then crop to 960x1440. NO photometric distortion (the
paper lists only those three techniques). This is the e1f2f3b baseline (best,
mDice 63.63) plus ONE change (D1): the multi-scale runs on the NATIVE 2848x4288
image -- a real downscale of native detail -- instead of on a pre-shrunk
960x1440 image that upscaling would blur. Grounded in M2MRF (paper's [7]):
tools/prepare_labels.py does NOT resize, so mmseg's
Resize(img_scale=(1440,960), ratio_range=(0.5,2.0)) rescales the native image.
Everything else (pad-then-normalize order, flips, crop) is kept identical to
e1f2f3b so this is a single-variable experiment isolating the detail question.
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
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

# All 81 IDRiD images are exactly this size (verified). mmcv keep-ratio resize
# is therefore deterministic per ratio, so a fixed RandomScale factor on the
# native image reproduces it exactly.
NATIVE_H, NATIVE_W = 2848, 4288
_IMG_SCALE = (1440, 960)       # mmseg img_scale (W, H)
_RATIO_RANGE = (0.5, 2.0)


def _mmcv_scale_factor(ratio: float) -> float:
    """mmcv.imrescale scale_factor for target (1440*r, 960*r) on the native
    image, keeping aspect ratio (edge-based, like mmcv)."""
    long_edge = max(_IMG_SCALE) * ratio
    short_edge = min(_IMG_SCALE) * ratio
    return min(long_edge / max(NATIVE_H, NATIVE_W),
               short_edge / min(NATIVE_H, NATIVE_W))


# RandomScale multiplies the native size by (1 + limit). We want the output to
# match mmcv's ratio-r size, i.e. factor = scale_factor(r). Since scale_factor
# is linear in r, uniform-in-factor == uniform-in-r (matching mmseg).
#   r=0.5 -> factor 0.168 -> 478x720   r=1 -> 0.336 -> 956x1440   r=2 -> 0.672 -> 1913x2880
_NATIVE_SCALE_LIMIT = (_mmcv_scale_factor(_RATIO_RANGE[0]) - 1.0,
                       _mmcv_scale_factor(_RATIO_RANGE[1]) - 1.0)

# albumentations 2.0 renamed PadIfNeeded's fill arguments (value/mask_value ->
# fill/fill_mask). The wrong pair is only a UserWarning, silently falling back
# to defaults, so select by version and keep the pad value explicit on both.
# Pad value 0 (raw black, applied BEFORE Normalize) matches e1f2f3b.
_PAD_FILL = ({'fill': 0, 'fill_mask': 0}
             if int(A.__version__.split('.')[0]) >= 2
             else {'value': 0, 'mask_value': 0})


class IDRiDDataset(Dataset):
    def __init__(self, split: str = 'train', root: Path = IDRID_ROOT,
                 classes: Tuple[str, ...] = CLASSES, is_train: bool = None,
                 cache: bool = True, photometric: bool = False,
                 full_res_eval: bool = False):
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
        # Full-resolution scoring needs the untouched mask, so the 960x1440
        # test cache (which pre-resizes both) has to be bypassed.
        self.full_res_eval = full_res_eval
        if full_res_eval and not self.is_train:
            cache = False
        self.cache = cache
        self.transform = build_transform(self.is_train,
                                         include_base_resize=not cache,
                                         photometric=photometric)
        self._cache: List[Tuple[np.ndarray, np.ndarray]] = []
        if cache:
            self._build_cache()

    def _build_cache(self) -> None:
        """Pre-decode every image/mask once to avoid re-reading a 12MP JPEG plus
        four 12MP TIFs on every one of 40k iterations.

        Train keeps NATIVE 2848x4288 so the multi-scale is a true downscale of
        real detail (the D1 fix). Test caches at the final 960x1440 inference
        size (no scale aug there). Native cache costs ~85MB/sample x 54 ~= 4.6GB;
        with spawn each worker copies it, so prefer --workers 0 (single copy) or
        keep worker count low.
        """
        if self.is_train:
            for path in self.images:
                self._cache.append(self._read_raw(path))
        else:
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

        if self.full_res_eval and not self.is_train:
            # mmseg scores against the native-resolution ground truth; see
            # BaseSegmentor.postprocess_result. Resize the image only.
            out = self.transform(image=image)
            mask_t = torch.from_numpy(mask).permute(2, 0, 1).float()
            return {'image': out['image'], 'mask': mask_t,
                    'filename': img_path.name}

        out = self.transform(image=image, mask=mask)
        return {
            'image': out['image'],                               # [3, H, W] float
            'mask': out['mask'].permute(2, 0, 1).float(),        # [C, H, W]
            'filename': img_path.name,
        }


def build_transform(is_train: bool, include_base_resize: bool = True,
                    photometric: bool = False) -> A.Compose:
    """Paper §4.2.2: rotation (90/180/270), flipping (h/v), multi-scaling
    (0.5-2.0), then crop to 960x1440. No photometric augmentation.

    photometric=True is a DELIBERATE DEVIATION from the paper, off by default.
    The paper states verbatim: "we use three data augmentation techniques
    including rotation (90, 180, and 270), flipping (horizontal and vertical),
    and multi-scaling (0.5-2.0)" -- three, enumerated, no colour jitter. The
    counter-argument is that M2MRF (ref [7], cited by the paper for the resize)
    does run PhotoMetricDistortion in its train_pipeline, and the WFDENet repo
    publishes no training pipeline at all to settle it. So the flag exists to
    test that hypothesis, never as the faithful default.

    D1 (vs e1f2f3b): the RandomScale runs on the NATIVE image, so ratio 2.0 is a
    real 0.67x downscale of the 4288-wide original (sharp), not a 2x upscale of a
    960x1440 image (blurred). _NATIVE_SCALE_LIMIT reproduces mmcv's per-ratio
    output sizes. For a fundus image the tiny lesions (MA ~5px) only survive if
    they are downsampled from native rather than interpolated up.

    Everything else matches e1f2f3b: pad (raw 0) THEN normalize, online
    flips/rot90 (same symmetry set as M2MRF's offline 6x). Normalisation runs on
    the raw 0-255 values (max_pixel_value=1.0), matching mmseg's
    SegDataPreProcessor, whose mean/std are on the 0-255 scale.
    """
    normalize = A.Normalize(mean=IDRID_MEAN, std=IDRID_STD, max_pixel_value=1.0)

    if not is_train:
        # Test: resize to the 960x1440 inference size (unless already cached).
        base = [A.Resize(CROP_H, CROP_W)] if include_base_resize else []
        return A.Compose([*base, normalize, ToTensorV2()])

    # Train input is always native 2848x4288 (cached raw, or read raw); the
    # multi-scale itself produces the 960x1440-scale crops.
    return A.Compose([
        A.RandomScale(scale_limit=_NATIVE_SCALE_LIMIT, p=1.0),   # ratio 0.5-2.0 on native
        # Rotation/flips before the crop: a 90/270 rotation transposes the
        # image, and the crop is the only step that guarantees a fixed
        # 960x1440 output for batch collation.
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.PadIfNeeded(min_height=CROP_H, min_width=CROP_W,
                      border_mode=cv2.BORDER_CONSTANT, **_PAD_FILL),
        A.RandomCrop(height=CROP_H, width=CROP_W),
        # Off by default -- see the docstring. Approximates mmseg's
        # PhotoMetricDistortion (whose brightness is additive +-32, not
        # multiplicative, so this is close but not identical).
        *([A.ColorJitter(brightness=0.125, contrast=(0.5, 1.5),
                         saturation=(0.5, 1.5), hue=0.05, p=1.0)]
          if photometric else []),
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
