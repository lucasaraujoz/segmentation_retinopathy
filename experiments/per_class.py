"""Single-lesion sigmoid arms for HardExudate, SoftExudate and Microaneurysms.

Generated in a loop instead of hand-written. The lesions differ only in `classes`, so a
loop makes them *provably* identical in every other field — hand-copying four blocks is
how a stray `wavelet_level` ends up in one arm and not the others.

Hemorrhage is deliberately NOT generated here: it keeps its historical ids (`H0`, `HL0`)
so the runs already sitting in outputs/_legado stay citable under the same name. Those
live in `hem_passive.py` and `hilo.py`. So the HL0 architecture appears in three modules
— `HL0` (hemorrhage), `ex-HL0`/`se-HL0`/`ma-HL0` (here), `all4-HL0` (multi_lesion.py) —
because the modules are organised by origin, not by architecture.
Run `python -m experiments` to see every arm grouped by output bucket.

Ids are `<code>-<arm>`: ex-H0, ex-HL0, se-H0, ... Each lands in its own bucket,
outputs/<code>/<id>_<name>/seed<N>, because `Config.group` derives the code from
`classes`. Both arms use dice_focal_alpha, which takes the alpha path and therefore
ignores `pos_weight` — the 2-element pos_weight default is only valid for (EX, HE) and
would broadcast silently against a 1-class target.
"""

from config import Config, _CLASS_CODES

# Lesions with enough annotated images in the `_3` suffix to train on.
# Counts at suffix _3 (588 images): EX 523, SE 311, MA 445.
# IRMA (11) and Neovascularization (14) are excluded — too few to fold.
LESIONS = ('HardExudate', 'SoftExudate', 'Microaneurysms')

# The two arms every lesion gets: the plain UNet, and the HL0 mechanism on top of it.
# Kept byte-identical to H0 / HL0 except for `classes`, so the per-lesion gap is
# measured the same way the hemorrhage gap was.
_ARMS = {
    'H0': dict(
        exp_name='baseline',
        loss_type='dice_focal_alpha',
        wavelet_skip_indices=(),
    ),
    'HL0': dict(
        exp_name='hilo_idwt_L1_allskips',
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_fusion='hilo',
    ),
}

EXPERIMENTS: dict[str, Config] = {}
for _cls in LESIONS:
    _code = _CLASS_CODES[_cls]
    for _arm, _fields in _ARMS.items():
        _id = f'{_code}-{_arm}'
        EXPERIMENTS[_id] = Config(exp_id=_id, classes=(_cls,), **_fields)
