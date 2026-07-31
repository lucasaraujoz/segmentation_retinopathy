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

# Training multi-scale, following the M2MRF config the paper cites for the
# resize (fcn_hr48-M2MRF-C_40k_idrid_bdice.py):
#     Resize(img_scale=(1440,960), ratio_range=(0.5,2.0), keep_ratio=True)
# mmcv.imrescale keeps aspect, so for the uniform 2848x4288 IDRiD images the
# per-ratio output size is deterministic: scale_factor = 0.3358 * r, giving
# r=0.5 -> 478x720, r=1.0 -> 956x1440, r=2.0 -> 1913x2880. It is ALWAYS a
# downscale of the native detail -- never an upscale.
RATIO_RANGE = (0.5, 2.0)

# We cache each training image once at the r=2.0 target (the largest size any
# ratio needs) and realise the random scale as a downscale from there. Aspect
# 1913/2880 = 0.6642 matches the native 2848/4288, so no distortion.
HI_H, HI_W = 1913, 2880

# RandomScale multiplies the *cached* size. Since the cache is the r=2.0 image,
# reproducing ratio r means multiplying by r/2, i.e. a factor in [0.25, 1.0].
# albumentations samples the factor uniformly, matching mmseg's uniform-in-r.
_SCALE_LIMIT = (RATIO_RANGE[0] / RATIO_RANGE[1] - 1.0,   # 0.5/2 - 1 = -0.75
                RATIO_RANGE[1] / RATIO_RANGE[1] - 1.0)   # 2/2   - 1 =  0.0

# albumentations 2.0 renamed PadIfNeeded's fill arguments (value/mask_value ->
# fill/fill_mask). The wrong pair is only a UserWarning, silently falling back
# to defaults, so select by version and keep the pad value explicit on both.
_PAD_FILL = ({'fill': 0, 'fill_mask': 0}
             if int(A.__version__.split('.')[0]) >= 2
             else {'value': 0, 'mask_value': 0})


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
        # Train caches at the r=2.0 high-res target so the random scale is a real
        # downscale; test caches directly at the final 960x1440 (no scale aug).
        self.cache_h, self.cache_w = (HI_H, HI_W) if self.is_train else (CROP_H, CROP_W)
        self.transform = build_transform(self.is_train, include_base_resize=not cache)
        self._cache: List[Tuple[np.ndarray, np.ndarray]] = []
        if cache:
            self._build_cache()

    def _build_cache(self) -> None:
        """Pre-decode every image/mask once at the cache resolution.

        Train: 1913x2880 (the r=2.0 target) so the per-sample RandomScale only
        ever downscales real detail -- caching at 960x1440 and scaling up was
        the bug that blurred the tiny MA/SE lesions. Test: 960x1440, the final
        inference size. Resize is linear for the image, nearest for the mask
        (albumentations default), matching mmcv. Avoids re-decoding a 12MP JPEG
        plus four 12MP TIFs on every one of 40k iterations.
        Cost: ~38MB/sample x 54 ~= 2.1GB (train); ~260MB (test).
        """
        base = A.Resize(self.cache_h, self.cache_w)
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
    """Replicates the M2MRF/mmseg train_pipeline the WFDENet paper cites.

    Reference (fcn_hr48-M2MRF-C_40k_idrid_bdice.py):
        Resize(img_scale=(1440,960), ratio_range=(0.5,2.0))   # downscale of the ORIGINAL
        RandomCrop(crop_size=(960,1440), cat_max_ratio=0.75)  # near-noop here, omitted
        RandomFlip(flip_ratio=0)                              # flips/rot are OFFLINE, 6x
        PhotoMetricDistortion()
        Normalize(**img_norm_cfg)                             # BEFORE pad
        Pad(size=crop_size, pad_val=0, seg_pad_val=0)

    Four things this fixes vs. the previous version (see plan Iteracao 2):
      1. RandomScale runs on the r=2.0 cache (a real downscale of the 4288-wide
         original), not on a pre-shrunk 960x1440 image that upscaling would blur.
         _SCALE_LIMIT maps the factor to [0.25, 1.0], reproducing ratio 0.5-2.0.
      2. Normalize before Pad, so the padded border is 0 in normalised space =
         the dataset mean colour, not black (-1.45 sigma).
      3. ColorJitter approximates PhotoMetricDistortion (brightness/contrast/
         saturation/hue).
      4. Pad position 'top_left' matches mmseg's crop-then-pad placement (image
         top-left, border bottom-right); we pad-then-crop because albumentations
         RandomCrop requires image >= crop.

    Documented approximations: ColorJitter's brightness is multiplicative vs
    PMD's additive; the r<2 downscale starts from the 2880 cache, not the 4288
    native (negligible antialiasing difference); online flips/rot cover the same
    symmetry set as the offline 6x; cat_max_ratio omitted.

    Normalisation runs on the raw 0-255 values (max_pixel_value=1.0), matching
    mmseg's SegDataPreProcessor, whose mean/std are on the 0-255 scale.
    """
    normalize = A.Normalize(mean=IDRID_MEAN, std=IDRID_STD, max_pixel_value=1.0)

    if not is_train:
        # Test: resize to the 960x1440 inference size. Skipped when the dataset
        # already cached at that size.
        base = [A.Resize(CROP_H, CROP_W)] if include_base_resize else []
        return A.Compose([*base, normalize, ToTensorV2()])

    # Without a cache the sample arrives native (2848x4288); bring it to the
    # r=2.0 base first so the scale factor means the same thing.
    base = [A.Resize(HI_H, HI_W)] if include_base_resize else []

    return A.Compose([
        *base,
        A.RandomScale(scale_limit=_SCALE_LIMIT, p=1.0),   # ratio 0.5-2.0 as downscale
        # Flips/rotation before the crop: a 90/270 rotation transposes the image,
        # and the crop is what pins the output to a fixed 960x1440 for collation.
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter(brightness=0.125, contrast=(0.5, 1.5),
                      saturation=(0.5, 1.5), hue=0.05, p=1.0),   # ~ PhotoMetricDistortion
        normalize,                                        # BEFORE pad
        A.PadIfNeeded(min_height=CROP_H, min_width=CROP_W, position='top_left',
                      border_mode=cv2.BORDER_CONSTANT, **_PAD_FILL),
        A.RandomCrop(height=CROP_H, width=CROP_W),
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
