"""All four lesions in one model.

Two output heads are possible and only one is implemented today:

  sigmoid (multilabel)  — 4 independent channels, a pixel may carry several lesions.
                          `all4-H0` and `all4-HL0` below. Works now.
  softmax (multiclass)  — 5 channels (4 lesions + background), a pixel gets exactly one.
                          NOT registered here: `task='multiclass'` raises NotImplementedError
                          in losses.py:147 and metrics.py:92, and `_load_mask` still returns
                          multilabel channels. See §1.3 of docs/plano_experimentos.md.

Class order is (EX, HE, SE, MA), matching the WFDENet papers' Tables 1/2 column order and
our own replication, so per-class numbers line up with them without reshuffling.

Softmax note for later: measured on 60 `_3` images, only 0.09% of lesion pixels belong to
more than one class, so forcing a single label is cheap — but it does need a documented
priority order for those pixels.
"""

from config import Config

CLASSES = ('HardExudate', 'Hemorrhage', 'SoftExudate', 'Microaneurysms')

# Same two arms as the single-lesion families, so "one model per lesion" vs "one model for
# all four" is a clean comparison: identical architecture, identical recipe, only the head
# width and the targets change.
_ARMS = {
    'H0': dict(
        exp_name='baseline_sigmoid',
        loss_type='dice_focal_alpha',
        wavelet_skip_indices=(),
    ),
    'HL0': dict(
        exp_name='hilo_idwt_L1_allskips_sigmoid',
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_fusion='hilo',
    ),
}

EXPERIMENTS: dict[str, Config] = {}
for _arm, _fields in _ARMS.items():
    _id = f'all4-{_arm}'
    # dice_focal_alpha takes the alpha path, so the 2-element `pos_weight` default
    # (valid only for EX+HE) is ignored rather than silently broadcast over 4 channels.
    EXPERIMENTS[_id] = Config(exp_id=_id, classes=CLASSES, task='multilabel', **_fields)
