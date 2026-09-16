"""Sanity checks for HiLoWaveletSkip (models/wavelet.py + factory 'hilo', exp HL0).

The claim under test: the skip is recomposed by a real IDWT over separately-boosted
LF/HF paths — not by upsampling the bands and concatenating them (H2L1A's way), and not
by dropping HF entirely (AsymmetricWaveletSkip's way).
"""
import sys, torch
sys.path.insert(0, __import__('os').path.dirname(__file__))
from models.wavelet import HiLoWaveletSkip, AsymmetricWaveletSkip
from models.factory import build_model
from config import EXPERIMENTS
from losses import build_loss

torch.manual_seed(0)
ok_all = True
def check(name, cond):
    global ok_all; ok_all = ok_all and bool(cond)
    print(f'{"OK " if cond else "FAIL"} {name}')

# 1. Shape preserved (the decoder must see an unchanged skip) and output finite.
m = HiLoWaveletSkip(in_channels=160, level=1).eval()
x = torch.randn(2, 160, 32, 32)
with torch.no_grad():
    y = m(x)
check(f'out shape {tuple(y.shape)} == in, finite',
      y.shape == x.shape and torch.isfinite(y).all())

# 2. Starts as a gentle residual (α=0.1), i.e. close to identity rather than overwriting.
rel = (y - x).norm() / x.norm()
check(f'α starts at 0.1 and output stays near identity (‖Δ‖/‖x‖ = {rel:.3f})',
      abs(m.alpha.item() - 0.1) < 1e-6 and rel < 0.5)

# 3. IDWT∘DWT round-trip is exact for the module's own primitives (orthonormal haar).
z = torch.randn(1, 8, 16, 16)
ll, lh, hl, hh = m._dwt_one_level(z)
rec = m._idwt_one_level(ll, lh, hl, hh, z.shape[-2:])
err = (rec - z).abs().max().item()
check(f'IDWT(DWT(z)) == z (max err {err:.2e})', err < 1e-4)

# 4. THE key structural test: the HF path actually reaches the output.
#    Zeroing the HF boosters degenerates the module to LF-only reconstruction; if the
#    output is unchanged, HF was never contributing (the failure mode of A0/A_sym).
with torch.no_grad():
    y_full = m(x)
    for p in m.hf_convs.parameters():
        p.zero_()
    m.hf_convs.eval()
    y_lfonly = m(x)
d = (y_full - y_lfonly).abs().max().item()
check(f'HF bands participate in the reconstruction (max Δ = {d:.4f})',
      d > 1e-4 and not torch.allclose(y_full, y_lfonly))

# 5. HF is mixed jointly across orientations (one conv over 3C), not per-band like A_sym.
m2 = HiLoWaveletSkip(in_channels=64, level=1)
hf_in = m2.hf_convs[0][0].in_channels
a_sym = AsymmetricWaveletSkip(64, symmetric=True)
check(f'HF conv sees all 3 bands at once (in_channels={hf_in} = 3x64), vs A_sym per-band 1x1',
      hf_in == 3 * 64 and hasattr(a_sym, 'hf_conv') and len(a_sym.hf_conv) == 3)

# 6. Multi-level: one HF booster per level, still shape-preserving.
m3 = HiLoWaveletSkip(in_channels=56, level=2).eval()
with torch.no_grad():
    y3 = m3(torch.randn(1, 56, 64, 64))
check(f'level=2 → {len(m3.hf_convs)} HF boosters, out {tuple(y3.shape)}',
      len(m3.hf_convs) == 2 and y3.shape == (1, 56, 64, 64))

# 7. HL0 end-to-end: builds, eval → plain tensor, train → loss backward, α gets grad.
cfg = EXPERIMENTS['HL0']
model = build_model(cfg)
check(f'HL0 uses the hilo module on all 4 skips ({len(model.wavelet_modules)} modules)',
      len(model.wavelet_modules) == 4
      and all(isinstance(mm, HiLoWaveletSkip) for mm in model.wavelet_modules.values()))

model.eval()
with torch.no_grad():
    ye = model(torch.randn(1, 3, 512, 512))
check(f'HL0 eval → {tuple(ye.shape)} finite, no aux heads',
      torch.is_tensor(ye) and ye.shape == (1, 1, 512, 512) and torch.isfinite(ye).all())

model.train()
criterion = build_loss(cfg)
imgs = torch.randn(2, 3, 512, 512)
masks = (torch.rand(2, 1, 512, 512) > 0.9).float()
out = model(imgs)
logits = out[0] if isinstance(out, tuple) else out
loss = criterion(logits, masks)
loss.backward()
trained = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
nonfinite = sum(not torch.isfinite(p.grad).all() for p in trained)
alphas = [model.wavelet_modules[k].alpha for k in model.wavelet_modules]
check(f'HL0 backward: loss={loss.item():.4f}, {len(trained)} tensors com grad, {nonfinite} não-finitos',
      torch.isfinite(loss) and nonfinite == 0 and len(trained) > 400)
check('α recebe gradiente nas 4 skips',
      all(a.grad is not None and torch.isfinite(a.grad).all() for a in alphas))

# 8. Parameter cost vs H2L1A (the arm it will be compared against).
h2l1a = build_model(EXPERIMENTS['H2L1A'])
n_hl0 = sum(p.numel() for p in model.parameters())
n_h2 = sum(p.numel() for p in h2l1a.parameters())
print(f'     HL0 {n_hl0:,} params vs H2L1A {n_h2:,} ({n_hl0 - n_h2:+,})')

print('done —', 'ALL OK' if ok_all else 'SOME FAILED')
sys.exit(0 if ok_all else 1)
