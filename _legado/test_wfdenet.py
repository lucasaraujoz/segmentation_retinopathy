"""Sanity checks for the faithful WFDENet port (models/wfdenet.py)."""
import sys, torch
sys.path.insert(0, __import__('os').path.dirname(__file__))
from models.factory import build_model
from config import EXPERIMENTS

torch.manual_seed(0)
ok_all = True

def check(name, cond):
    global ok_all
    ok_all = ok_all and cond
    print(f'{"OK " if cond else "FAIL"} {name}')

# 1. Full model: eval → tensor of the right shape; no NaN/Inf.
model = build_model(EXPERIMENTS['W0']).eval()
n_params = sum(p.numel() for p in model.parameters())
with torch.no_grad():
    y = model(torch.randn(2, 3, 512, 512))
check(f'W0 eval → tensor {tuple(y.shape)} (params {n_params/1e6:.1f}M)',
      torch.is_tensor(y) and y.shape == (2, 1, 512, 512) and torch.isfinite(y).all())

# 2. Full model: train → (logits, [aux_lfb, aux_hfb]); backward flows.
model.train()
out = model(torch.randn(2, 3, 512, 512))
is_tuple = isinstance(out, tuple) and len(out) == 2 and len(out[1]) == 2
check(f'W0 train → tuple with {len(out[1]) if is_tuple else "?"} aux heads', is_tuple)
if is_tuple:
    logits, aux = out
    loss = logits.mean() + sum(a.mean() for a in aux)
    loss.backward()
    # Healthy backward = no NaN/Inf grads. A few params legitimately get no grad
    # (boundary neighbour convs unused at the finest/coarsest level; EfficientNet's
    # classifier head that smp's depth-5 extractor drops) — those are expected.
    trained = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    nonfinite = sum(not torch.isfinite(p.grad).all() for p in trained)
    check(f'W0 backward: {len(trained)} params updated, {nonfinite} non-finite grads',
          nonfinite == 0 and len(trained) > 550)
    check('W0 aux heads at level-1 (finest booster) resolution',
          aux[0].shape[-1] == 128 and aux[1].shape[-1] == 128)

# 3. Ablation toggles: each builds, forwards (eval), correct #aux (train).
expected_aux = {'W_LFB': 1, 'W_HFB': 1, 'W_noCCFAM': 2, 'W_noSD': 2}
for exp, n_aux in expected_aux.items():
    m = build_model(EXPERIMENTS[exp])
    m.eval()
    with torch.no_grad():
        ye = m(torch.randn(1, 3, 512, 512))
    m.train()
    ot = m(torch.randn(1, 3, 512, 512))
    got = len(ot[1]) if isinstance(ot, tuple) else 0
    check(f'{exp}: eval tensor {tuple(ye.shape)}, train {got} aux (exp {n_aux})',
          torch.is_tensor(ye) and ye.shape == (1, 1, 512, 512)
          and torch.isfinite(ye).all() and got == n_aux)

print('done —', 'ALL OK' if ok_all else 'SOME FAILED')
sys.exit(0 if ok_all else 1)
