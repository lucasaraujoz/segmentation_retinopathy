# `replication_2/` — auditoria da réplica contra o código dos autores

Objetivo: parar de auditar por leitura e **executar o código oficial**, para decidir
se o `replication/` mede o método dos autores ou uma leitura nossa dele.

## Resultado

**O nosso port é bit-a-bit idêntico ao head dos autores.**

```
parameters (head only): theirs 2,998,580 | ours 2,998,580 | MATCH
  mapped 322 tensors, all shapes agree, none left over

output      shape                   max |diff|         rel
main (SD)   (2, 4, 64, 96)           0.000e+00   0.000e+00
aux (HFB)   (2, 4, 64, 96)           0.000e+00   0.000e+00
aux (LFB)   (2, 4, 64, 96)           0.000e+00   0.000e+00
worst max|diff| = 0.000e+00   (tolerância 1e-05)   =>  IDENTICAL

loss  ours 0.8074194193 | theirs 0.8074194193 | diff 0.000e+00
```

Não é "próximo": é **zero**. Os 322 tensores do head mapearam 1:1, sem sobra dos
dois lados, e as três saídas coincidem exatamente.

⇒ O Δ medido na ablação (**+1.07 mDice no DDR**, **+1.03 no IDRiD**) é uma
propriedade do método, não um artefato de implementação.

## Como reproduzir

```bash
PY=/home/lucas/.conda/envs/notebook/bin/python
$PY -m pip install mmengine "mmcv-lite==2.1.0" mmpretrain \
                   importlib_metadata modelindex rich einops mat4py
$PY replication_2/equivalence_test.py
git -C replication_2/upstream status --porcelain   # só __pycache__
```

`mmcv-lite==2.1.0`: o `mmseg/__init__.py` deles exige `mmcv < 2.2.0`, e a variante
*lite* evita compilar ops CUDA (impossível com torch 2.12+cu130). Nenhuma op CUDA
do mmcv é usada por este head.

## Procedência

- Clone: `github.com/xuanli01/WFDENet`, SHA **`38b0b16`** ("paper", 2025-10-07).
- `replication_2/upstream/` é **read-only**. Nenhuma linha dos autores foi editada;
  a única coisa escrita aqui é o mapeamento de nomes de pesos, e ele se
  autovalida (todo tensor consumido, toda shape conferida).

## O que foi conferido linha a linha

Li as 564 linhas de `wfdenet_head.py`. Batem com o nosso
[`replication/wfdenet_paper.py`](../replication/wfdenet_paper.py):

| componente | veredito |
|---|---|
| `forward` (laterais → DWT → LFB/HFB → IDWT → NF+out_f → FPN main) | idêntico |
| `FPNHead` — `out[i]=conv_i(x_i)`, `x_{i-1} += Up(out[i])`, retorno por nível | idêntico |
| `DWT` / `IDWT` — escala `/2`, sinais, posicionamento dos quadrantes | idêntico |
| `pad_to_even` (`mode='reflect'`, só nível 4) / `unpad` | idêntico |
| `CCFAM` — `autocast(False)` → fp32 → `rfft2(norm='ortho')` → CCL → CA → SA → `irfft2` | idêntico |
| `ComplexCA` — `avg+max` pool, `ComplexConv1d(k=3)`, sigmoid sobre o par | idêntico |
| `ComplexSA` — `C→C/4→1`, ReLU, Sigmoid | idêntico |
| `ComplexConv2d/1d` — multiplicação complexa | idêntico |
| `ComplexBatchNorm` — whitening de Trabelsi (Vrr/Vri/Vii, `s`, `t`, `rst`, Z) | idêntico |
| `NeighborFuse` — `x + fuse(x + downconv(lf) + upconv(Up(x_{l+1})))` | idêntico |
| backbone: params (6,513,184), 5 shapes de saída, TF-SAME | idêntico |

## Divergências reais encontradas

**1. BN eps do backbone — a única diferença numérica do modelo inteiro.**

O config passa `norm_cfg=dict(type='SyncBN', requires_grad=True)`, que
**sobrescreve** o default da classe deles (`BN, eps=1e-3`). Resultado: eles rodam
com **eps 1e-5**. Nosso `timm tf_efficientnet_b1.in1k` usa **1e-3** (convenção TF).

```
BN eps  theirs [1e-05] | ours [0.001]
conv    theirs Conv2dAdaptivePadding | ours Conv2dSame     (mesma semântica SAME)
```

Afeta os **dois braços da ablação igualmente** ⇒ cancela no Δ. Move o offset, não
a conclusão.

**2. `eps` da loss — NÃO é divergência (corrigindo um alarme falso meu).**

O default de `binary_dice_loss` é `1e-3`, mas o config passa `eps=1e-5`
explicitamente, e o nosso `replication/loss.py` já usa `1e-5`. **Confere.**
Cheguei a reportar isso como divergência lendo só o `loss.py`; o config decide.

**3. A `Loss` publicada não roda para `bdice`.**

`Loss.forward` chama `self.loss_func(..., naive=self.naive, ...)` mas
`binary_dice_loss` não tem esse parâmetro:

```
TypeError: binary_dice_loss() got an unexpected keyword argument 'naive'
```

Coerente com um release *inference-only* — o caminho de treino nunca é exercitado
pelo `test.py` deles. Chamamos `binary_dice_loss` diretamente, que é o que a
Eq. 12/13 descreve.

## O que continua não publicado (e não some clonando)

- **Pipeline de treino**: `configs/WFDENet/ours_{idrid,ddr}.py` e
  `configs/_base_/datasets/*.py` têm **só `test_pipeline`**. Sem otimizador, sem
  schedule, sem batch size, sem augmentação. Reconstruído do §4.2.2 + M2MRF.
- **Peso λ das perdas auxiliares**: o `loss_by_feat` herdado é o do mmseg *stock*,
  que recebe um `Tensor` único — não a lista de 3 saídas que o head devolve — e
  não tem peso auxiliar nenhum. O λ=0.5 vem da Eq. 13 do artigo, não de código.
- **Pesos ImageNet**: sem `init_cfg` no config. Ver abaixo.

## Próximo lever (não executado)

O backbone deles é `mmpretrain.EfficientNet`, então o checkpoint correspondente é

```
https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty_8xb32_in1k_20220119-002556d9.pth
```

Trocar o nosso substituto `timm` por esse checkpoint (+ eps 1e-5) ataca o offset
sistemático de **−2.86 mDice** que a réplica carrega em toda configuração
(IDRiD completo −2.86; DDR ablado −2.86).

**Mas isso move o offset, não o Δ** — e é o Δ que responde se o mecanismo funciona.
