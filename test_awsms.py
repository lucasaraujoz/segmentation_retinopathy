"""Sanity checks for the multi-scale asymmetric wavelet skip (A_MS)."""
import sys, torch
sys.path.insert(0, __import__('os').path.dirname(__file__))
from models.wavelet import MultiScaleAsymWaveletSkip
from models.factory import build_model
from config import EXPERIMENTS
from losses import build_loss

torch.manual_seed(0)
ok_all = True
def check(name, cond):
    global ok_all; ok_all = ok_all and bool(cond)
    print(f'{"OK " if cond else "FAIL"} {name}')

# EfficientNet-B4 skips at 512²: (48,256²) (32,128²) (56,64²) (160,32²) — finest → coarsest.
CH = [48, 32, 56, 160]
RES = [256, 128, 64, 32]
skips = [torch.randn(2, c, r, r) for c, r in zip(CH, RES)]

# 1. Module: shapes preserved per level (channels AND resolution).
m = MultiScaleAsymWaveletSkip(CH, 'haar').eval()
with torch.no_grad():
    out = m(skips)
shapes_ok = all(o.shape == s.shape for o, s in zip(out, skips))
check(f'module returns {len(out)} skips with original shapes {[tuple(o.shape[1:]) for o in out]}',
      len(out) == 4 and shapes_ok and all(torch.isfinite(o).all() for o in out))

# 2. FPN alignment: LL_i must be exactly 2× the resolution of LL_{i+1}.
with torch.no_grad():
    lls = [m._dwt_one_level(m.proj_in[i](skips[i]))[0] for i in range(4)]
sizes = [ll.shape[-1] for ll in lls]
aligned = all(sizes[i] == 2 * sizes[i + 1] for i in range(len(sizes) - 1))
check(f'LL pyramid aligns for Up2 fusion: {sizes}', aligned)

# 3. Full model: eval tensor, backward healthy, integrates with the real loss.
model = build_model(EXPERIMENTS['A_MS'])
model.eval()
with torch.no_grad():
    ye = model(torch.randn(1, 3, 512, 512))
check(f'A_MS eval → {tuple(ye.shape)} finite',
      torch.is_tensor(ye) and ye.shape == (1, 1, 512, 512) and torch.isfinite(ye).all())

cfg = EXPERIMENTS['A_MS']
model = build_model(cfg).train()
criterion = build_loss(cfg)
out = model(torch.randn(2, 3, 512, 512))
logits = out[0] if isinstance(out, tuple) else out
loss = criterion(logits, (torch.rand(2, 1, 512, 512) > 0.9).float())
loss.backward()
trained = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
nonfinite = sum(not torch.isfinite(p.grad).all() for p in trained)
check(f'A_MS train loss={loss.item():.4f}, {len(trained)} params, {nonfinite} non-finite grads',
      torch.isfinite(loss) and nonfinite == 0 and len(trained) > 400)

alpha = model.ms_module.alpha
check(f'per-level residual α gets grad (α={[round(a, 3) for a in alpha.tolist()]})',
      alpha.grad is not None and torch.isfinite(alpha.grad).all())

# 4. The cross-level fusion is real: perturbing ONLY the coarsest skip must change the finest output.
model.eval()
with torch.no_grad():
    base = model.ms_module(skips)[0].clone()
    perturbed = [s.clone() for s in skips]
    perturbed[-1] = perturbed[-1] + 10.0          # only the coarsest level
    after = model.ms_module(perturbed)[0]
check('coarsest skip influences the finest output (cross-level fusion active)',
      not torch.allclose(base, after, atol=1e-6))

print('done —', 'ALL OK' if ok_all else 'SOME FAILED')
sys.exit(0 if ok_all else 1)
