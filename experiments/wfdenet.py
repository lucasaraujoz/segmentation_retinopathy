"""Faithful WFDENet port (Li et al., PatRec 2026) inside our FGADR testbed.

W0 is the only arm that has ever cleared the noise floor against H0. W_* are the
module ablations mirroring the paper's Tables 6/8 — written, never run.
"""

from config import Config


EXPERIMENTS: dict[str, Config] = {
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
}
