import cv2
import json
import time
import warnings
import numpy as np
import pandas as pd
import pywt
import torch
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config


_MASK_DIRS = {
    'HardExudate': 'HardExudate_Masks',
    'Hemorrhage':  'Hemohedge_Masks',
}


def extract_suffix(fname: str) -> str:
    """'0563_3.png' → '3'"""
    return fname.rsplit('_', 1)[1].split('.')[0]


def filter_by_suffix(df: pd.DataFrame, allowed: 'tuple | None') -> pd.DataFrame:
    """Keep only rows whose filename suffix is in *allowed*. None = keep all."""
    if not allowed:
        return df
    keep = df['filename'].apply(extract_suffix).isin(set(allowed))
    return df[keep].reset_index(drop=True)


class FGADRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: Config, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.fgadr = Path(config.fgadr_path)
        self.transform = self._build_transform(is_train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # A transient PNG read failure (libpng CRC / truncated read under heavy I/O)
        # must NOT kill a multi-hour run. Retry the sample, then fall back to a
        # neighbour so the epoch keeps going; only give up if many in a row fail.
        for attempt in range(5):
            try:
                return self._get_sample(idx)
            except (RuntimeError, cv2.error) as e:
                warnings.warn(
                    f'[dataset] sample {idx} ({self.df.iloc[idx]["filename"]}) failed: {e}; '
                    f'retry/fallback {attempt + 1}/5'
                )
                time.sleep(0.2)
                idx = (idx + 1) % len(self.df)
        raise RuntimeError(f'Could not load any sample after 5 attempts (started near idx {idx})')

    def _get_sample(self, idx):
        row = self.df.iloc[idx]
        fname = row['filename']

        img  = self._load_image(fname)    # [H, W, 3] uint8
        mask = self._load_mask(fname, img.shape[:2])  # [H, W, C] uint8

        augmented = self.transform(image=img, mask=mask)
        image_t = augmented['image']                           # [3, H, W] float32
        mask_t  = augmented['mask'].permute(2, 0, 1).float()  # [C, H, W]

        if self.config.input_preproc == 'wavelet_channels':
            image_t = self._append_wavelet_channels(image_t)   # [3,H,W] -> [6,H,W]

        return {'image': image_t, 'mask': mask_t, 'filename': fname}

    @staticmethod
    def _imread_retry(path: Path, flags: int = cv2.IMREAD_COLOR, tries: int = 3) -> np.ndarray:
        # cv2.imread returns None on a truncated/CRC-failed PNG read (can happen
        # transiently under heavy I/O). Retry with a short backoff before giving up
        # so a single glitchy read doesn't crash a whole epoch.
        for attempt in range(tries):
            img = cv2.imread(str(path), flags)
            if img is not None:
                return img
            time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f'Failed to read after {tries} retries: {path}')

    def _load_image(self, fname: str) -> np.ndarray:
        path = self.fgadr / 'Original_Images' / fname
        img = self._imread_retry(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.config.apply_clahe:
            img = self._apply_clahe(img)
        if self.config.input_preproc == 'wavelet_illumnorm':
            img = self._wavelet_illumnorm(img)
        return img

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ── Wavelet input preprocessing ───────────────────────────────────────────

    @staticmethod
    def _haar_lowfreq(chan: np.ndarray, level: int) -> np.ndarray:
        """Coarse low-frequency (Haar LL) of a 2D channel, upsampled back to its size."""
        coeffs = pywt.wavedec2(chan, 'haar', level=level, mode='periodization')
        ll = coeffs[0]
        return cv2.resize(ll, (chan.shape[1], chan.shape[0]), interpolation=cv2.INTER_LINEAR)

    def _wavelet_illumnorm(self, img: np.ndarray) -> np.ndarray:
        """Flatten uneven illumination: divide each channel by its coarse Haar-LL background.
        Keeps 3 channels, uint8. Makes dark diffuse lesions (hemorrhage) more consistent."""
        out = np.empty(img.shape, dtype=np.float32)
        for c in range(3):
            chan = img[:, :, c].astype(np.float32)
            bg = self._haar_lowfreq(chan, level=5)          # slow illumination component
            out[:, :, c] = chan / (bg + 1e-3) * float(bg.mean())
        return np.clip(out, 0, 255).astype(np.uint8)

    def _append_wavelet_channels(self, image_t: torch.Tensor) -> torch.Tensor:
        """Append [LL, detail-magnitude, illumnorm] of the (augmented) green channel → 6ch.
        Computed post-transform so geometry matches the mask; each map z-scored per image."""
        # de-normalize green back to ~[0,1] so divisions are stable (ImageNet g: mean .456, std .224)
        g = (image_t[1] * 0.224 + 0.456).clamp(0, 1).numpy().astype(np.float32)
        H, W = g.shape

        c = pywt.wavedec2(g, 'haar', level=2, mode='periodization')
        ll = pywt.waverec2([c[0]] + [tuple(np.zeros_like(d) for d in det) for det in c[1:]],
                           'haar', mode='periodization')[:H, :W]          # low-freq (full res)

        c1 = pywt.wavedec2(g, 'haar', level=1, mode='periodization')
        cH, cV, cD = c1[1]
        detail = cv2.resize(np.abs(cH) + np.abs(cV) + np.abs(cD), (W, H))  # high-freq magnitude

        bg = self._haar_lowfreq(g, level=4)
        illum = g / (bg + 1e-3)                                            # illumination-normalized

        maps = []
        for m in (ll, detail, illum):
            m = (m - m.mean()) / (m.std() + 1e-6)                          # z-score per image
            m = np.clip(m, -5.0, 5.0)                                      # tame border spikes (bg≈0)
            maps.append(torch.from_numpy(m.astype(np.float32)))
        extra = torch.stack(maps, dim=0)                                  # [3, H, W]
        return torch.cat([image_t, extra], dim=0)                        # [6, H, W]

    def _load_mask(self, fname: str, img_shape: tuple) -> np.ndarray:
        # Load masks at original resolution — albumentations Resize handles
        # image and mask together, so they must share the same input shape.
        H0, W0 = img_shape
        channels = []
        for cls in self.config.classes:
            mask_path = self.fgadr / _MASK_DIRS[cls] / fname
            if mask_path.exists():
                m = self._imread_retry(mask_path, cv2.IMREAD_GRAYSCALE)
                channels.append((m > self.config.bin_threshold).astype(np.uint8))
            else:
                channels.append(np.zeros((H0, W0), dtype=np.uint8))

        return np.stack(channels, axis=-1)  # [H, W, 2]

    def _build_transform(self, is_train: bool) -> A.Compose:
        H, W = self.config.image_size
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        if is_train:
            return A.Compose([
                A.Resize(H, W),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                                   rotate_limit=45, border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.OneOf([
                    A.ElasticTransform(p=1),
                    A.GridDistortion(p=1),
                ], p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.4),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(H, W),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])


# ── Split utilities ───────────────────────────────────────────────────────────

def _load_fgadr_df(config: Config) -> pd.DataFrame:
    presence_path = Path(config.presence_csv)
    df = pd.read_csv(presence_path)

    dr_csv = Path(config.fgadr_path) / 'DR_Seg_Grading_Label.csv'
    dr = pd.read_csv(dr_csv, header=None, names=['filename', 'dr_grade'])
    df = df.merge(dr, on='filename', how='left')
    return df


def create_splits(df: pd.DataFrame, config: Config, splits_path: str) -> dict:
    splits_file = Path(splits_path)
    if splits_file.exists():
        with open(splits_file) as f:
            return json.load(f)

    rng = np.random.default_rng(config.random_seed)

    # Hold out test patients (15%, stratified by DR grade)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=config.test_fraction, random_state=config.random_seed
    )
    train_val_idx, test_idx = next(splitter.split(
        df, groups=df['patient_id'], y=df['dr_grade']
    ))
    df_trainval = df.iloc[train_val_idx].reset_index(drop=True)
    test_indices = df.iloc[test_idx].index.tolist()

    # 5-fold CV on remaining patients
    kf = GroupKFold(n_splits=config.n_folds)
    folds = []
    for train_i, val_i in kf.split(df_trainval, groups=df_trainval['patient_id']):
        folds.append({
            'train': df_trainval.iloc[train_i].index.tolist(),
            'val':   df_trainval.iloc[val_i].index.tolist(),
        })

    splits = {
        'test': test_indices,
        'folds': folds,
        'trainval_indices': df_trainval.index.tolist(),
    }

    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f'Splits saved → {splits_file}')
    print(f'  Train+val: {len(df_trainval)} images ({len(df_trainval.patient_id.unique())} patients)')
    print(f'  Test:      {len(test_idx)} images ({len(df.iloc[test_idx].patient_id.unique())} patients)')
    return splits


# ── Dataset & loader registries ───────────────────────────────────────────────
# To add a new dataset: (1) write a Dataset subclass and a loader function,
# (2) register both below, (3) set dataset_name in Config. train.py stays unchanged.

_DATASET_REGISTRY: dict = {
    'fgadr': FGADRDataset,
}

_LOADER_REGISTRY: dict = {
    'fgadr': _load_fgadr_df,
}


def build_dataset(config: Config, df: pd.DataFrame, is_train: bool) -> Dataset:
    """Instantiate the dataset class registered for config.dataset_name."""
    cls = _DATASET_REGISTRY[config.dataset_name]
    return cls(df, config, is_train)


def load_dataframe(config: Config) -> pd.DataFrame:
    """Load the raw DataFrame for the dataset registered under config.dataset_name."""
    loader = _LOADER_REGISTRY[config.dataset_name]
    return loader(config)
