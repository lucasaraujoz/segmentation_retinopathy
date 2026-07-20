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

    # Loss ablation (baseline architecture, no wavelet) — exp00 is the control
    '10': Config(
        exp_id='10', exp_name='dicefocal_alpha',
        loss_type='dice_focal_alpha',
        wavelet_skip_indices=(),
    ),
    '11': Config(
        exp_id='11', exp_name='focal_tversky',
        loss_type='focal_tversky',
        wavelet_skip_indices=(),
    ),

    # ── Hemorrhage-only testbed (binary: 0 background, 1 Hemorrhage) ──────────
    # Isolates the wavelet's effect. Evidence (wavelet_haar_multilevel.ipynb):
    # hemorrhage signal lives in the LL (approximation), which the detail-only
    # WaveletSkipConnection discards — so H2 (with LL) is the real hypothesis test.
    # Reference: user baseline HEM Dice 0.603; FGADR paper U-Net 0.570, DenseU-Net 0.617.
    'H0': Config(
        exp_id='H0', exp_name='hem_baseline',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_skip_indices=(),
    ),
    'H1': Config(
        exp_id='H1', exp_name='hem_haar_detail',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1),
        wavelet_include_ll=False,
    ),
    'H2': Config(
        exp_id='H2', exp_name='hem_haar_ll',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1),
        wavelet_include_ll=True,
    ),

    # ── Wavelet as INPUT preprocessing (before the encoder), hemorrhage-only ──
    # Feature-map wavelet (H1/H2) was ~null; pixel-level evidence for LL/hemorrhage
    # is strong, so apply the wavelet on the raw image instead. Compare vs H0.
    'H3': Config(
        exp_id='H3', exp_name='hem_illumnorm',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        input_preproc='wavelet_illumnorm',   # divide by coarse Haar-LL background (3ch)
    ),
    'H4': Config(
        exp_id='H4', exp_name='hem_wavelet_channels',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        input_preproc='wavelet_channels',    # RGB + [LL, detail, illumnorm] of green (6ch)
    ),

    # ── Level sweep on H2 (LL in features), hemorrhage-only ──────────────────
    # Only wavelet_level changes; skips=(0,1) + include_ll fixed, so any effect is
    # attributable to depth. Ceiling 3 (deeper → feature maps too small). Note: extra
    # levels mostly add DETAIL bands, which hemorrhage barely uses → modest expectation.
    'H2L2': Config(
        exp_id='H2L2', exp_name='hem_haar_ll_L2',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=2,
        wavelet_skip_indices=(0, 1),
        wavelet_include_ll=True,
    ),
    'H2L3': Config(
        exp_id='H2L3', exp_name='hem_haar_ll_L3',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=3,
        wavelet_skip_indices=(0, 1),
        wavelet_include_ll=True,
    ),

    # ── Skip POSITION on the best depth (L2 + LL), hemorrhage-only ────────────
    # Every H* arm fixed skips=(0,1); position is the one untested axis. Depth is
    # already swept (L1/L2/L3 all tied). Keep the only combo with a rationale
    # (L2 + LL) and open coverage to all skips (0,1,2,3) — deeper skips operate on
    # coarser feature maps, matching hemorrhage's large low-frequency scale.
    # Index 4 (features[5]) is the bottleneck → excluded. Compare GAP vs H0/H2L2.
    'H2L2A': Config(
        exp_id='H2L2A', exp_name='hem_haar_ll_L2_allskips',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=2,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_include_ll=True,
    ),
    # ── Closing grid: POSITION × DEPTH on the passive module, hemorrhage-only ─
    # The original ablation (exps 01-09) predates the hemorrhage-only testbed: it is
    # multilabel + dice_focal + detail-only (include_ll=False), so it cannot be compared
    # against H0/H2L2A. These arms re-run that same 2x3 grid inside the current protocol:
    # passive module (all 4 subbands → upsample → concat → 1x1), haar, +LL, dice_focal_alpha.
    # Position: first skip (0,) — the original bet, finest resolution — vs all skips.
    # Depth: L1/L2/L3. H2L2A is the all-skips/L2 cell, so only 5 arms are missing.
    # Expectation: flat. Extra levels add mostly DETAIL bands and hemorrhage lives in LL
    # (AUC 0.23 vs ~0.6 detail) → depth should be null, with a mechanism, at 5 folds.
    'H2L1A': Config(
        exp_id='H2L1A', exp_name='hem_haar_ll_L1_allskips',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_include_ll=True,
    ),
    'H2L3A': Config(
        exp_id='H2L3A', exp_name='hem_haar_ll_L3_allskips',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=3,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_include_ll=True,
    ),
    'H2L1F': Config(
        exp_id='H2L1F', exp_name='hem_haar_ll_L1_skip0',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0,),
        wavelet_include_ll=True,
    ),
    'H2L2F': Config(
        exp_id='H2L2F', exp_name='hem_haar_ll_L2_skip0',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=2,
        wavelet_skip_indices=(0,),
        wavelet_include_ll=True,
    ),
    'H2L3F': Config(
        exp_id='H2L3F', exp_name='hem_haar_ll_L3_skip0',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=3,
        wavelet_skip_indices=(0,),
        wavelet_include_ll=True,
    ),

    # ── Active wavelet fusion (WFDENet-style), hemorrhage-only ───────────────
    # Passive skips (H1–H2L2A) were null: the branch is upsample+concat, no IDWT
    # reconstruction / no per-band enhancement / no aux loss, so the net ignores it
    # (project_wfdenet_analysis). H5→H7 add the missing active components one at a
    # time (isolable ablation). All inherit H0's setup + L2/all-skips/LL like H2L2A.
    'H5': Config(
        exp_id='H5', exp_name='hem_awf_idwt',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=2,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_include_ll=True,
        wavelet_fusion='idwt',              # DWT → per-band 1x1 → IDWT → residual
    ),
    'H6': Config(
        exp_id='H6', exp_name='hem_awf_enhanced',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=2,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_include_ll=True,
        wavelet_fusion='idwt_enh',          # H5 + LL booster + CBAM-lite HF denoising
    ),
    'H7': Config(
        exp_id='H7', exp_name='hem_awf_deepsup',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=2,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_include_ll=True,
        wavelet_fusion='idwt_enh',
        deep_supervision=True,              # H6 + auxiliary Dice heads on wavelet skips
    ),

    # ── Faithful WFDENet port (Li et al., PatRec 2026), hemorrhage-only ───────
    # H5-H7 were a per-skip approximation; these run the REAL WFDENet (own decoder):
    # WHLFD → LFB (FPN low-freq) + HFB/CCFAM (Fourier complex attention high-freq)
    # → SD (IDWT + adjacent-level fusion). Same encoder/loss/splits as H0 for a fair
    # gap. Module ablations (W_LFB…W_noSD) mirror the paper's Tables 6/8 to isolate
    # which frequency half helps hemorrhage. Success = gap vs H0 in AUPR/FROC, not Dice.
    'W0': Config(
        exp_id='W0', exp_name='wfdenet_full',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        arch='wfdenet',
        deep_supervision=True,
    ),
    'W_LFB': Config(
        exp_id='W_LFB', exp_name='wfdenet_lfb_only',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        arch='wfdenet',
        deep_supervision=True,
        wfdenet_use_hfb=False,              # low-freq/semantics only
    ),
    'W_HFB': Config(
        exp_id='W_HFB', exp_name='wfdenet_hfb_only',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        arch='wfdenet',
        deep_supervision=True,
        wfdenet_use_lfb=False,              # high-freq/details only
    ),
    'W_noCCFAM': Config(
        exp_id='W_noCCFAM', exp_name='wfdenet_no_ccfam',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        arch='wfdenet',
        deep_supervision=True,
        wfdenet_use_ccfam=False,            # HFB = FPN only (isolate the Fourier attention)
    ),
    'W_noSD': Config(
        exp_id='W_noSD', exp_name='wfdenet_no_sd',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        arch='wfdenet',
        deep_supervision=True,
        wfdenet_use_sd=False,               # SD adjacent-fusion → residual block
    ),

    # ── Asymmetric Wavelet Skip (AWS), hemorrhage-only — OUR CONTRIBUTION ─────
    # Paradigm inversion motivated by the frequency-localization evidence + the W0
    # ablation (WFDENet's hemorrhage gain is 100% low-freq; the HFB is dead weight):
    # LL is ENHANCED (semantics/FP↓), the oriented HF bands become a vessel-SUPPRESSION
    # gate (not lesion detail). Lightweight, plugs into the stock smp UNet decoder
    # (WaveletUnet path), same recipe as H0 for a fair gap. A_sym / A_noGate isolate
    # the asymmetry as the source of the gain. Success = gap vs H0/W0 in AUPR/FROC/FP.
    'A0': Config(
        exp_id='A0', exp_name='aws_asym_gate',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_fusion='asym',
        aws_use_gate=True, aws_symmetric=False,
    ),
    'A_noGate': Config(
        exp_id='A_noGate', exp_name='aws_asym_nogate',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_fusion='asym',
        aws_use_gate=False, aws_symmetric=False,   # LL-enhance + IDWT only (isolates the gate)
    ),
    'A_sym': Config(
        exp_id='A_sym', exp_name='aws_symmetric',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_fusion='asym',
        aws_symmetric=True,                        # enhance ALL bands = the "fixed H5" control
    ),

    # ── AWS + cross-level multi-scale fusion, hemorrhage-only ────────────────
    # A0 landed at baseline (its vessel gate never fired: FP went UP, AUPR flat) while W0 clearly
    # won → the active ingredient is multi-scale LOW-FREQ aggregation ACROSS levels, not per-skip
    # band manipulation. A_MS isolates exactly that: keep the asymmetry (reconstruct low-freq only),
    # add the FPN-style cross-level fusion, drop everything else W0 has (no CCFAM/Fourier, no custom
    # decoder, no deep-sup, no gate). If A_MS ≈ W0 → a light module matches the heavy SOTA.
    'A_MS': Config(
        exp_id='A_MS', exp_name='aws_multiscale',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_fusion='asym_ms',
    ),

    # ── H2L1A incrementado: deep-sup wavelet + atenção global wavelet ────────
    # H2L1A (melhor wavelet: passivo, all-skips, L1, +LL) ganha do baseline mas perde pro W0 por
    # ~1% (supressão de FP). O ganho do W0 vem de deep-sup + contexto global, não do skip. Aqui,
    # sobre a MESMA base H2L1A: (WA_ds) deep-sup só nas skips grossas (conserta o H7 que supervisava
    # todas); (WA_at) self-attention wavelet no bottleneck (FP=vaso=longo alcance); (WA) os dois.
    'WA_ds': Config(
        exp_id='WA_ds', exp_name='h2l1a_deepsup_coarse',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_include_ll=True,
        deep_supervision=True,
        deepsup_indices=(2, 3),             # só skips grossas (64² e 32²), sem o ruído das rasas
    ),
    'WA_at': Config(
        exp_id='WA_at', exp_name='h2l1a_wavelet_attn',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_include_ll=True,
        bottleneck_attn=True,
    ),
    'WA': Config(
        exp_id='WA', exp_name='h2l1a_deepsup_attn',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_include_ll=True,
        deep_supervision=True,
        deepsup_indices=(2, 3),
        bottleneck_attn=True,
    ),

    # ── Wavelet WITHOUT pretraining, hemorrhage-only ─────────────────────────
    # Pretrained arms (H0-H4) all tied → hypothesis: ImageNet already supplies the
    # low-freq/edge prior wavelet would give. From scratch the prior may matter:
    # expect gap(S2-S0) > 0 while gap(H2-H0) ≈ 0. Compare the GAP, not absolute Dice.
    'S0': Config(
        exp_id='S0', exp_name='hem_scratch_baseline',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        encoder_weights=None,                # random init
    ),
    'S2': Config(
        exp_id='S2', exp_name='hem_scratch_haar_ll',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        encoder_weights=None,
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1),
        wavelet_include_ll=True,
    ),
    'S4': Config(
        exp_id='S4', exp_name='hem_scratch_channels',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        encoder_weights=None,
        input_preproc='wavelet_channels',
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
