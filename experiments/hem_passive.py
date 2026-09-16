"""Hemorrhage-only, passive wavelet skip (WaveletSkipConnection).

H0 is THE baseline every later arm is measured against. H1-H4 vary the input side,
H2L* is the POSITION x DEPTH grid, S* drops ImageNet pretraining.
"""

from config import Config


EXPERIMENTS: dict[str, Config] = {
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
}
