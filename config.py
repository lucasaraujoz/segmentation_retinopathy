from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence
from pathlib import Path


@dataclass
class Config:
    # --- Dataset ---
    dataset_name: str = 'fgadr'
    fgadr_path: str = '/home/lucas/datasets/fgadr/Seg-set'
    presence_csv: str = 'outputs/fgadr_class_presence.csv'   # all images, no filter
    image_size: Tuple[int, int] = (512, 512)
    classes: Tuple[str, ...] = ('HardExudate', 'Hemorrhage')
    bin_threshold: int = 127
    apply_clahe: bool = False
    allowed_suffixes: Optional[Tuple[str, ...]] = None        # None = all (_1, _2, _3)

    # --- Model ---
    encoder_name: str = 'efficientnet-b4'
    encoder_weights: str = 'imagenet'
    # Wavelet configuration
    wavelet_family: str = 'haar'            # haar | db2 | db4 | sym4
    wavelet_level: int = 1                  # decomposition depth
    wavelet_skip_indices: Tuple[int, ...] = ()  # empty = baseline (no wavelet)

    # --- Training ---
    batch_size: int = 4                     # EfficientNet-B4 @ 512x512 needs ~3GB/sample
    num_workers: int = 0                    # 0 = safe on Linux (cv2+fork deadlock); set >0 for throughput
    num_epochs: int = 150                   # exp00 val Dice ainda subia na ep90 → mais runway p/ convergir
    learning_rate: float = 3e-4            # OneCycleLR max_lr; 1e-4 causou underfitting (val Dice caiu), 3e-4 converge melhor
    scheduler_pct_start: float = 0.15      # warmup curto: modelo está underfitting, deixar mais steps p/ aprender/anelar
    weight_decay: float = 1e-5
    # pos_weight for BCEWithLogitsLoss: [HardExudate, Hemorrhage]
    # computed on full dataset with mask > 127
    pos_weight: Tuple[float, float] = (136.8, 89.4)
    accumulation_steps: int = 8             # effective batch = batch_size * accumulation_steps = 4 * 8 = 32 (igual ao paper)

    # --- Loss ---
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    focal_gamma: float = 2.0

    # --- Split (5-fold CV + fixed test) ---
    n_folds: int = 5
    test_fraction: float = 0.15             # held-out test set by patient
    random_seed: int = 42

    # --- Task mode ---
    task: str = 'multilabel'             # 'multilabel' | 'multiclass'

    # --- Experiment metadata ---
    exp_id: str = '00'
    exp_name: str = 'baseline'

    # --- Logging ---
    use_wandb: bool = True
    wandb_project: str = 'fgadr-wavelet-ablation'
    output_dir: str = 'outputs'

    @property
    def out_channels(self) -> int:
        """Output channels: one per class (multilabel) or classes+1 for multiclass (incl. background)."""
        if self.task == 'multiclass':
            return len(self.classes) + 1
        return len(self.classes)

    @property
    def exp_dir(self) -> Path:
        return Path(self.output_dir) / f'exp_{self.exp_id}_{self.exp_name}'

    @property
    def has_wavelet(self) -> bool:
        return len(self.wavelet_skip_indices) > 0


# ── Experiment registry ──────────────────────────────────────────────────────

EXPERIMENTS: dict[str, Config] = {
    # Baseline: standard UNet + EfficientNet-B4, no wavelet
    '00': Config(
        exp_id='00', exp_name='baseline',
        wavelet_skip_indices=(),
    ),

    # Wavelet families — first skip only, level 1
    '01': Config(
        exp_id='01', exp_name='haar_L1_skip0',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0,),
    ),
    '02': Config(
        exp_id='02', exp_name='db2_L1_skip0',
        wavelet_family='db2', wavelet_level=1,
        wavelet_skip_indices=(0,),
    ),
    '03': Config(
        exp_id='03', exp_name='db4_L1_skip0',
        wavelet_family='db4', wavelet_level=1,
        wavelet_skip_indices=(0,),
    ),
    '04': Config(
        exp_id='04', exp_name='sym4_L1_skip0',
        wavelet_family='sym4', wavelet_level=1,
        wavelet_skip_indices=(0,),
    ),

    # Decomposition depth — Haar, first skip only
    '05': Config(
        exp_id='05', exp_name='haar_L2_skip0',
        wavelet_family='haar', wavelet_level=2,
        wavelet_skip_indices=(0,),
    ),
    '06': Config(
        exp_id='06', exp_name='haar_L3_skip0',
        wavelet_family='haar', wavelet_level=3,
        wavelet_skip_indices=(0,),
    ),

    # Skip positions — Haar level 1, increasing coverage
    '07': Config(
        exp_id='07', exp_name='haar_L1_skip01',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1),
    ),
    '08': Config(
        exp_id='08', exp_name='haar_L1_skip012',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2),
    ),
    '09': Config(
        exp_id='09', exp_name='haar_L1_all_skips',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
    ),
}
