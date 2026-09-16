"""Experiment registry, split by family.

`config.py` holds the Config dataclass; the ~50 experiment definitions live here, one
module per family, so adding an arm touches one small file instead of a 550-line one.

To add a family: write `experiments/<name>.py` exposing `EXPERIMENTS: dict[str, Config]`
and list the module in `_MODULES` below. Duplicate ids across families are an error at
import time, not a silent last-one-wins.
"""

from importlib import import_module

from config import Config

_MODULES = (
    'legacy_multilabel',   # 00-11, 01-09 — original EX+HE ablation
    'hem_passive',         # H0-H4, H2L*, S* — hemorrhage, passive skip
    'hem_active',          # H5-H7 — per-band enhancement + IDWT
    'wfdenet',             # W0, W_* — faithful port and its module ablations
    'aws',                 # A*, WA* — asymmetric skip, bottleneck attention
    'hilo',                # HL* — the current line of work
    'per_class',           # ex-*, se-*, ma-* — single-lesion arms, generated
    'multi_lesion',        # all4-* — the four lesions in one model
)

EXPERIMENTS: dict[str, Config] = {}
_ORIGIN: dict[str, str] = {}

for _mod_name in _MODULES:
    _mod = import_module(f'{__name__}.{_mod_name}')
    for _id, _cfg in _mod.EXPERIMENTS.items():
        if _id in EXPERIMENTS:
            raise ValueError(
                f'duplicate experiment id {_id!r}: defined in {_ORIGIN[_id]} '
                f'and again in {_mod_name}'
            )
        if _id != _cfg.exp_id:
            raise ValueError(
                f'{_mod_name}: registry key {_id!r} does not match exp_id {_cfg.exp_id!r} '
                f'— the key names the CLI (--exp) and exp_id names the directory, '
                f'so they must agree'
            )
        EXPERIMENTS[_id] = _cfg
        _ORIGIN[_id] = _mod_name


def origin(exp_id: str) -> str:
    """Which family module defines this experiment. For error messages and reports."""
    return _ORIGIN[exp_id]


__all__ = ['EXPERIMENTS', 'origin']
