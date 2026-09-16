"""Original EX+HE multilabel ablation (exps 00-11 and 01-09).

Predates the hemorrhage-only testbed: multilabel, dice_focal, detail-only
(include_ll=False). NOT comparable against H0/H2L1A/HL0 — kept for the record and
for the loss ablation (10, 11), which is where dice_focal_alpha was chosen.
"""

from config import Config


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
