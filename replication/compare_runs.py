"""Compare two replication runs with a paired test over the 27 test images.

Both runs are scored on exactly the same images, so a paired Wilcoxon on the
per-image Dice is far more powerful than eyeballing the aggregated numbers --
which matters here because MA covers ~0.1% of the pixels and a handful of
images dominate the aggregate.

    python replication/compare_runs.py <dir_A> <dir_B>

Reads test_results.json (aggregated, the Table-1-comparable numbers) and
test_scores.npz (per-image Dice) from each directory.

Caveat: the aggregated Dice is NOT the mean of the per-image Dice -- it pools
TP/FP/FN over the dataset first. Report the aggregated value; use the per-image
column only for the significance test.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compare_experiments import wilcoxon_paired  # noqa: E402  -- reuse, don't reimplement
from replication.idrid import CLASSES  # noqa: E402


def load_run(d: Path) -> tuple:
    results_path = d / 'test_results.json'
    scores_path = d / 'test_scores.npz'
    if not results_path.exists():
        raise FileNotFoundError(f'{results_path} not found')
    results = json.loads(results_path.read_text())
    scores = np.load(scores_path, allow_pickle=True) if scores_path.exists() else None
    if scores is None:
        print(f'warning: {scores_path} missing -- paired test unavailable for {d.name}.\n'
              f'         Regenerate with: python replication/train_idrid.py '
              f'--eval-only --ckpt {d}/final.pth --out-dir {d}')
    return results, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_a', type=str, help='baseline / reference run')
    parser.add_argument('dir_b', type=str, help='run to compare against it')
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args()

    a_dir, b_dir = Path(args.dir_a), Path(args.dir_b)
    res_a, sc_a = load_run(a_dir)
    res_b, sc_b = load_run(b_dir)

    print(f'A = {a_dir.name}')
    print(f'B = {b_dir.name}\n')

    paired = sc_a is not None and sc_b is not None

    header = f'{"metric":<12}{"A":>9}{"B":>9}{"B-A":>9}'
    if paired:
        header += f'{"p (pareado)":>14}{"":>4}'
    print(header)
    print('-' * len(header))

    for cls in CLASSES:
        key = f'Dice_{cls}'
        if key not in res_a or key not in res_b:
            continue
        va, vb = res_a[key], res_b[key]
        line = f'{key:<12}{va:>9.2f}{vb:>9.2f}{vb - va:>+9.2f}'

        if paired and cls in sc_a.files and cls in sc_b.files:
            da, db = sc_a[cls].astype(float), sc_b[cls].astype(float)
            # Dice is nan where a class is absent from both pred and GT
            # (e.g. SE in 13 of the 27 test images) -- drop those pairs.
            ok = ~(np.isnan(da) | np.isnan(db))
            if ok.sum() >= 2:
                w = wilcoxon_paired(da[ok], db[ok])
                p = w['p_value']
                mark = '*' if (p == p and p < args.alpha) else ''
                line += f'{p:>14.4f}{mark:>4}  (n={int(ok.sum())})'
            else:
                line += f'{"n/a":>14}{"":>4}'
        print(line)

    print('-' * len(header))
    for key in ('mDice', 'mAUPR', 'mIoU'):
        if key in res_a and key in res_b:
            va, vb = res_a[key], res_b[key]
            print(f'{key:<12}{va:>9.2f}{vb:>9.2f}{vb - va:>+9.2f}')

    if paired:
        print(f'\n* = p < {args.alpha} (Wilcoxon pareado sobre o Dice por imagem).')
        print('Dice agregado (as colunas A/B) NAO e a media do Dice por imagem: '
              'ele agrupa TP/FP/FN no dataset inteiro, que e a convencao da '
              'Tabela 1. O teste pareado usa o por-imagem so para significancia.')
    else:
        print('\nSem test_scores.npz nos dois runs -- so a comparacao agregada '
              'esta disponivel (sem suporte estatistico).')


if __name__ == '__main__':
    main()
