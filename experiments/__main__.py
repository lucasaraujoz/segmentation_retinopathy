"""`python -m experiments` — list every registered arm, so nobody has to grep config files.

Grouped by output bucket, which is also the first level under outputs/, so the listing
reads in the same shape as the results directory.
"""

from collections import defaultdict

from experiments import EXPERIMENTS, origin

_by_group = defaultdict(list)
for _id, _cfg in EXPERIMENTS.items():
    _by_group[_cfg.group].append((_id, _cfg))

print(f'{len(EXPERIMENTS)} experimentos\n')
for group in sorted(_by_group):
    print(f'── {group} ' + '─' * (62 - len(group)))
    for exp_id, cfg in _by_group[group]:
        arch = cfg.arch if cfg.arch != 'unet' else (cfg.wavelet_fusion if cfg.has_wavelet else 'unet cru')
        print(f'  --exp {exp_id:<12} {arch:<10} {origin(exp_id):<18} {cfg.exp_dir}')
    print()
