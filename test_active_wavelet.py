import sys, torch
sys.path.insert(0, __import__('os').path.dirname(__file__))
from models.wavelet import ActiveWaveletFusion
from models.factory import build_model
from config import EXPERIMENTS

torch.manual_seed(0)

# 1. IDWT(DWT(x)) ≈ x for Haar (multi-level roundtrip, no enhancement)
awf = ActiveWaveletFusion(in_channels=8, wavelet='haar', level=2, enhance=False).eval()
x = torch.randn(2, 8, 64, 64)
approx = x
sizes, details = [], []
for _ in range(awf.level):
    sizes.append(approx.shape[-2:])
    approx, LH, HL, HH = awf._dwt_one_level(approx)
    details.append((LH, HL, HH))
ll = approx
for lvl in reversed(range(awf.level)):
    LH, HL, HH = details[lvl]
    ll = awf._idwt_one_level(ll, LH, HL, HH, sizes[lvl])
err = (ll - x).abs().max().item()
print(f'[1] Haar IDWT(DWT(x)) roundtrip max-abs err = {err:.2e}  -> {"OK" if err < 1e-4 else "FAIL"}')

# 2. Forward shape preserved (idwt and idwt_enh)
for enh in (False, True):
    m = ActiveWaveletFusion(16, 'haar', level=2, enhance=enh).eval()
    y = m(torch.randn(2, 16, 32, 32))
    print(f'[2] enhance={enh}: out shape {tuple(y.shape)} -> {"OK" if y.shape == (2,16,32,32) else "FAIL"}')

# 3. build_model for H5/H6/H7: eval -> tensor; train (H7) -> (logits, aux list)
for exp in ('H5', 'H6', 'H7'):
    model = build_model(EXPERIMENTS[exp])
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 512, 512))
    is_tensor = torch.is_tensor(out)
    print(f'[3] {exp} eval -> {"tensor "+str(tuple(out.shape)) if is_tensor else type(out)}  -> {"OK" if is_tensor else "FAIL"}')

model = build_model(EXPERIMENTS['H7']); model.train()
out = model(torch.randn(1, 3, 512, 512))
ok = isinstance(out, tuple) and len(out) == 2 and len(out[1]) == 4
print(f'[3] H7 train -> tuple with {len(out[1]) if isinstance(out,tuple) else "?"} aux heads  -> {"OK" if ok else "FAIL"}')
print('done')
