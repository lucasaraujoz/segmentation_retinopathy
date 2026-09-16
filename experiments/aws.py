"""Asymmetric Wavelet Skip and the bottleneck-attention family — our earlier bets.

All landed inside the noise band around H0. Kept because the null result is itself
evidence: per-skip band manipulation is exhausted.
"""

from config import Config


EXPERIMENTS: dict[str, Config] = {
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
}
