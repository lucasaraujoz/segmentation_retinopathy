"""Hemorrhage-only, active fusion: per-band enhancement then IDWT (ActiveWaveletFusion).
"""

from config import Config


EXPERIMENTS: dict[str, Config] = {
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
}
