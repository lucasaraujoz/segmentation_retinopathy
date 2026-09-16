"""Sanity checks for the Asymmetric Wavelet Skip (models/wavelet.py + factory 'asym')."""
import sys, torch
sys.path.insert(0, __import__('os').path.dirname(__file__))
from models.wavelet import AsymmetricWaveletSkip
from models.factory import build_model
from config import EXPERIMENTS
from losses import build_loss

torch.manual_seed(0)
ok_all = True
def check(name, cond):
    global ok_all; ok_all = ok_all and bool(cond)
    print(f'{"OK " if cond else "FAIL"} {name}')

# 1. Module: output shape preserved, gate in [0,1], residual gentle at init.
m = AsymmetricWaveletSkip(in_channels=160, wavelet='haar', level=1, use_gate=True).eval()
x = torch.randn(2, 160, 32, 32)
with torch.no_grad():
    y = m(x)
check(f'module out shape {tuple(y.shape)} == in', y.shape == x.shape and torch.isfinite(y).all())
# recompute the gate directly to check its range
with torch.no_grad():
    u = m.proj_in(x); _, lh, hl, hh = m._dwt_one_level(u)
    g = torch.sigmoid(m.gate_conv(torch.cat([lh.abs(), hl.abs(), hh.abs()], 1)))
check(f'vessel gate g in [0,1] (min {g.min():.3f}, max {g.max():.3f})', g.min() >= 0 and g.max() <= 1)

# 2. Ablation wiring: correct submodules present per variant.
mods = {e: build_model(EXPERIMENTS[e]) for e in ('A0', 'A_noGate', 'A_sym')}
def sub(mm):  # grab one AsymmetricWaveletSkip from the WaveletUnet
    return mm.wavelet_modules['0']
check('A0 has gate, no hf_conv',
      hasattr(sub(mods['A0']), 'gate_conv') and not hasattr(sub(mods['A0']), 'hf_conv'))
check('A_noGate has neither gate nor hf_conv (LL-enhance only)',
      not hasattr(sub(mods['A_noGate']), 'gate_conv') and not hasattr(sub(mods['A_noGate']), 'hf_conv'))
check('A_sym has hf_conv, no gate (symmetric control)',
      hasattr(sub(mods['A_sym']), 'hf_conv') and not sub(mods['A_sym']).use_gate)

# 3. Full model: eval tensor, train tensor (no deep-sup), backward healthy, integrates with loss.
for e, mm in mods.items():
    mm.eval()
    with torch.no_grad():
        ye = mm(torch.randn(1, 3, 512, 512))
    check(f'{e} eval → {tuple(ye.shape)} finite', torch.is_tensor(ye)
          and ye.shape == (1, 1, 512, 512) and torch.isfinite(ye).all())

cfg = EXPERIMENTS['A0']
model = build_model(cfg).train()
criterion = build_loss(cfg)
imgs = torch.randn(2, 3, 512, 512)
masks = (torch.rand(2, 1, 512, 512) > 0.9).float()
out = model(imgs)
logits = out[0] if isinstance(out, tuple) else out       # A-series: plain tensor (no deep-sup)
loss = criterion(logits, masks)
loss.backward()
trained = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
nonfinite = sum(not torch.isfinite(p.grad).all() for p in trained)
check(f'A0 train+loss={loss.item():.4f}, {len(trained)} params, {nonfinite} non-finite grads',
      torch.isfinite(loss) and nonfinite == 0 and len(trained) > 400)
# alpha (residual scale) receives a gradient
alpha = model.wavelet_modules['0'].alpha
check(f'AWS residual α gets grad (α={alpha.item():.3f})', alpha.grad is not None)

print('done —', 'ALL OK' if ok_all else 'SOME FAILED')
sys.exit(0 if ok_all else 1)
