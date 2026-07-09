"""
Paired comparison of two (or more) experiments on the detection metrics that
train.py now writes at test time. NO torch — reads only the sidecar files:
  - outputs/exp_<id>_*/test_scores.npz   (per-image pixel/lesion Dice, per-lesion hits)
  - outputs/exp_<id>_*/test_results.json (aggregate metrics + FROC curve)

What it adds on top of the per-run numbers (which are already in test_results.json):
  - a side-by-side table,
  - a FROC overlay (outputs/froc_compare.png),
  - paired statistics between the FIRST two experiments:
      Wilcoxon signed-rank on per-image pixel Dice and lesion Dice,
      exact McNemar on per-lesion hit/miss.

Usage:
    python compare_experiments.py [--class Hemorrhage] H0 H2L2 [H2 ...]
"""

from __future__ import annotations
import sys
import glob
import json
import numpy as np


def load_scores(exp_id: str, cls: str) -> dict:
    npz = glob.glob(f'outputs/exp_{exp_id}_*/test_scores.npz')
    js  = glob.glob(f'outputs/exp_{exp_id}_*/test_results.json')
    if not npz or not js:
        raise FileNotFoundError(
            f'missing sidecars for exp {exp_id} (run: python train.py --exp {exp_id} --eval-only ...)')
    d = np.load(npz[0], allow_pickle=True)
    j = json.load(open(js[0]))
    if f'pixdice_{cls}' not in d:
        raise KeyError(f'class "{cls}" not in {npz[0]} (classes present: '
                       f'{[k[8:] for k in d.files if k.startswith("pixdice_")]})')
    return {
        'files': d['files'],
        'pixdice': d[f'pixdice_{cls}'], 'lesdice': d[f'lesdice_{cls}'], 'hits': d[f'hits_{cls}'],
        'froc': j.get('froc', {}).get(cls, []),
        'json': j,
    }


# ── paired statistics (validated earlier) ───────────────────────────────────────

def wilcoxon_paired(a, b) -> dict:
    from scipy.stats import wilcoxon
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.allclose(a - b, 0):
        return {'p_value': 1.0, 'median_diff': 0.0, 'note': 'no differences'}
    try:
        _, p = wilcoxon(a, b)
    except ValueError as e:
        return {'p_value': float('nan'), 'median_diff': float(np.median(a - b)), 'note': str(e)}
    return {'p_value': float(p), 'median_diff': float(np.median(a - b)),
            'mean_A': float(a.mean()), 'mean_B': float(b.mean())}


def mcnemar_exact(hits_a, hits_b) -> dict:
    from scipy.stats import binomtest
    a = np.asarray(hits_a).astype(bool); b = np.asarray(hits_b).astype(bool)
    n10 = int((a & ~b).sum())      # A hit, B miss
    n01 = int((~a & b).sum())      # A miss, B hit
    n = n01 + n10
    if n == 0:
        return {'n01': 0, 'n10': 0, 'n_discordant': 0, 'p_value': 1.0}
    p = binomtest(min(n01, n10), n, 0.5, alternative='two-sided').pvalue
    return {'A_only_hits': n10, 'B_only_hits': n01, 'n_discordant': n, 'p_value': float(p)}


# ── report ──────────────────────────────────────────────────────────────────────

def main(exp_ids: list[str], cls: str = 'Hemorrhage'):
    print(f'classe: {cls}')
    exps = {e: load_scores(e, cls) for e in exp_ids}

    ref = exps[exp_ids[0]]
    for e in exp_ids[1:]:
        if not np.array_equal(exps[e]['files'], ref['files']):
            raise ValueError(f'exp {e} test files differ from {exp_ids[0]} (different split?)')
        if len(exps[e]['hits']) != len(ref['hits']):
            raise ValueError(f'exp {e} has a different number of GT lesions — not the same test set')

    # table (pull aggregates from each run's json when available, else recompute)
    print(f'\n{"exp":8s} | pixDice | lesDice | lesHD95 | sens  | FP/img')
    print('-' * 58)
    for e in exp_ids:
        j = exps[e]['json']
        pd = j.get(f'test_pixel_dice_{cls}', float(np.mean(exps[e]['pixdice'])))
        ld = j.get(f'test_lesion_dice_{cls}', float(np.nanmean(exps[e]['lesdice'])))
        hd = j.get(f'test_lesion_hd95_{cls}', float('nan'))
        se = j.get(f'test_sensitivity_{cls}', float('nan'))
        fp = j.get(f'test_fp_per_image_{cls}', float('nan'))
        print(f'{e:8s} | {pd:.4f}  | {ld:.4f}  | {hd:6.1f}  | {se:.3f} | {fp:5.2f}')

    # FROC overlay
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        for e in exp_ids:
            fr = exps[e]['froc']
            if fr:
                plt.plot([p['fp_per_image'] for p in fr], [p['sensitivity'] for p in fr],
                         marker='o', label=e)
        plt.xlabel('FP por imagem'); plt.ylabel('sensibilidade por lesão')
        plt.title(f'FROC — {cls}'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig('outputs/froc_compare.png', dpi=120, bbox_inches='tight')
        print('\nFROC → outputs/froc_compare.png')
    except Exception as ex:
        print(f'\n[FROC plot skipped: {ex}]')

    # paired stats between the first two
    if len(exp_ids) >= 2:
        A, B = exp_ids[0], exp_ids[1]
        print(f'\n── Testes pareados: {A} vs {B} ──')
        wp = wilcoxon_paired(exps[A]['pixdice'], exps[B]['pixdice'])
        wl = wilcoxon_paired(exps[A]['lesdice'], exps[B]['lesdice'])
        mc = mcnemar_exact(exps[A]['hits'], exps[B]['hits'])
        print(f'Wilcoxon pixel Dice : p={wp["p_value"]:.4f}  (mediana Δ={wp.get("median_diff",0):+.4f})')
        print(f'Wilcoxon lesion Dice: p={wl["p_value"]:.4f}  (mediana Δ={wl.get("median_diff",0):+.4f})')
        print(f'McNemar por-lesão   : p={mc["p_value"]:.4f}  '
              f'({A}-só {mc.get("A_only_hits",0)}, {B}-só {mc.get("B_only_hits",0)}, '
              f'discordantes {mc["n_discordant"]})')
        print('(p<0.05 = diferença significativa; senão, dentro do ruído)')


if __name__ == '__main__':
    args = sys.argv[1:]
    cls = 'Hemorrhage'
    if '--class' in args:
        i = args.index('--class'); cls = args[i + 1]; del args[i:i + 2]
    if not args:
        print('uso: python compare_experiments.py [--class Hemorrhage] H0 H2L2 [H2 ...]')
        sys.exit(1)
    main(args, cls)
