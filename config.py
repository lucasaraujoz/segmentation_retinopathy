import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Tuple, Sequence
from pathlib import Path


# Short codes for single-lesion runs, used to name the first level under outputs/.
_CLASS_CODES = {
    'Hemorrhage': 'hem',
    'HardExudate': 'ex',
    'SoftExudate': 'se',
    'Microaneurysms': 'ma',
    'IRMA': 'irma',
    'Neovascularization': 'nv',
}


@dataclass
class Config:
    # --- Dataset ---
    dataset_name: str = 'fgadr'
    fgadr_path: str = '/home/lucas/fgadr/Seg-set'
    presence_csv: str = 'outputs/fgadr_class_presence.csv'   # all images, no filter
    image_size: Tuple[int, int] = (512, 512)
    classes: Tuple[str, ...] = ('HardExudate', 'Hemorrhage')
    bin_threshold: int = 127
    apply_clahe: bool = False
    # Wavelet-based input preprocessing (applied to the image BEFORE the encoder):
    #   'none'              — raw RGB
    #   'wavelet_illumnorm' — divide by coarse Haar-LL background (flatten illumination), stays 3ch
    #   'wavelet_channels'  — append [LL, detail-mag, illumnorm] of the green channel → 6ch input
    input_preproc: str = 'none'
    allowed_suffixes: Optional[Tuple[str, ...]] = None        # None = all (_1, _2, _3)

    # --- Model ---
    encoder_name: str = 'efficientnet-b4'
    encoder_weights: Optional[str] = 'imagenet'   # None = train encoder from scratch (random init)
    # Wavelet configuration
    wavelet_family: str = 'haar'            # haar | db2 | db4 | sym4
    wavelet_level: int = 1                  # decomposition depth
    wavelet_skip_indices: Tuple[int, ...] = ()  # empty = baseline (no wavelet)
    wavelet_include_ll: bool = False        # also inject the final approximation (LL) band, not only details
    # Skip-connection fusion style:
    #   'passive'  — WaveletSkipConnection: DWT → upsample details → concat → 1x1 (original)
    #   'idwt'     — ActiveWaveletFusion: DWT → per-band 1x1 → IDWT reconstruct → residual add
    #   'idwt_enh' — 'idwt' + LL conv booster + CBAM-lite denoising on the detail bands
    #   'hilo'     — HiLoWaveletSkip: LF (LL) and HF (cat[LH,HL,HH]) boosted separately,
    #                recomposed by IDWT instead of upsample+concat (WFDENet, sem atenção)
    wavelet_fusion: str = 'passive'
    # Deep supervision: auxiliary Dice head on each wavelet-enhanced skip (WFDENet-style), λ below.
    deep_supervision: bool = False
    deepsup_weight: float = 0.5
    # Architecture selector: 'unet' (smp UNet, optionally wavelet skips) | 'wfdenet' (faithful port)
    arch: str = 'unet'
    # WFDENet module ablation toggles (only used when arch='wfdenet')
    wfdenet_use_lfb: bool = True
    wfdenet_use_hfb: bool = True
    wfdenet_use_ccfam: bool = True
    wfdenet_use_sd: bool = True
    # Asymmetric Wavelet Skip (AWS) — used when wavelet_fusion='asym' (our contribution)
    aws_use_gate: bool = True         # vessel-suppression gate from the oriented HF bands
    aws_symmetric: bool = False       # True = enhance/reconstruct ALL bands (the "fixed H5" control)
    # Deep-supervision placement + bottleneck wavelet attention (WA family)
    deepsup_indices: Tuple[int, ...] = ()   # which wavelet skips get aux heads (empty = all selected)
    bottleneck_attn: bool = False           # wavelet self-attention on the bottleneck (FP suppression)
    attn_heads: int = 4

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
    loss_type: str = 'dice_focal'          # 'dice_focal' | 'dice_focal_alpha' | 'focal_tversky'
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75              # class balance for loss_type='dice_focal_alpha' (replaces pos_weight)
    tversky_alpha: float = 0.3             # false-positive weight (focal_tversky)
    tversky_beta: float = 0.7              # false-negative weight; beta>alpha favors recall
    tversky_gamma: float = 1.333           # focal exponent (Abraham & Khan 2019)

    # --- Split (5-fold CV + fixed test) ---
    n_folds: int = 5
    test_fraction: float = 0.15             # held-out test set by patient
    random_seed: int = 42

    # --- Task mode ---
    task: str = 'multilabel'             # 'multilabel' | 'multiclass'

    # --- Experiment metadata ---
    exp_id: str = '00'
    exp_name: str = 'baseline'
    # Output grouping. Empty = derived from `classes`/`task` (see the `group` property):
    # one class -> its short code (hem/ex/se/ma); several -> multi<N>sig | multi<N>soft.
    # Set explicitly only to override that derivation.
    task_group: str = ''
    # TRAINING seed (model init, augmentation, dataloader order). Distinct from
    # `random_seed`, which seeds the patient SPLIT and must stay 42 forever, or every
    # result stops being comparable with the ones already in outputs/.
    seed: int = 42

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
    def in_channels(self) -> int:
        """Model input channels: 6 when appending wavelet maps, else 3 (RGB)."""
        return 6 if self.input_preproc == 'wavelet_channels' else 3

    @property
    def class_codes(self) -> tuple:
        """Short per-class labels for compact console lines: ex, hem, se, ma."""
        return tuple(_CLASS_CODES.get(c, c.lower()) for c in self.classes)

    @property
    def group(self) -> str:
        """Task bucket this run belongs to: the first level under outputs/."""
        if self.task_group:
            return self.task_group
        if len(self.classes) == 1:
            return _CLASS_CODES.get(self.classes[0], self.classes[0].lower())
        head = 'soft' if self.task == 'multiclass' else 'sig'
        return f'multi{len(self.classes)}{head}'

    def derive(self, **overrides) -> 'Config':
        """A copy with fields replaced — for defining a variant without retyping the base.

        `HL0.derive(exp_id='HL1', wavelet_level=2)` is safer than copy-pasting the block,
        because the fields you did not name are guaranteed identical to the parent.
        """
        return dataclasses.replace(self, **overrides)

    @property
    def exp_dir(self) -> Path:
        """outputs/<group>/<id>_<name>/seed<seed> — one directory per run, never shared."""
        return (Path(self.output_dir) / self.group
                / f'{self.exp_id}_{self.exp_name}' / f'seed{self.seed}')

    @property
    def has_wavelet(self) -> bool:
        return len(self.wavelet_skip_indices) > 0

# ── Experiment registry ──────────────────────────────────────────────────────
# The ~50 definitions live in the `experiments/` package, one module per family.
# Resolved lazily: `experiments/*.py` imports Config from here, so importing the package
# eagerly at module level would deadlock whichever of the two gets imported first.
# PEP 562 __getattr__ defers it to first use, which keeps `from config import EXPERIMENTS`
# working for train.py and the test_*.py scripts without the cycle.


def __getattr__(name):
    if name == 'EXPERIMENTS':
        from experiments import EXPERIMENTS as _registry
        return _registry
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = ['Config', 'EXPERIMENTS']
