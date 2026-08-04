# Réplica integral do WFDENet no IDRiD

Réplica de primeira ordem do paper:

> Xuan Li, Ding Ma, Xiangqian Wu.
> *WFDENet: Wavelet-based frequency decomposition and enhancement network for
> diabetic retinopathy lesion segmentation.*
> Pattern Recognition 172 (2026) 112492.

**Este pacote é independente do resto do repo.** Nada das nossas configs entra
aqui — nem o `Config`, nem o `train.py`, nem o `models/wfdenet.py`. O objetivo é
responder uma pergunta só: *o WFDENet reproduz os números publicados?*

## Por que não reusar o que já existe

| | nosso pipeline (`train.py`) | protocolo do paper |
|---|---|---|
| dataset | FGADR | IDRiD (54 treino / 27 teste, split oficial) |
| loop | epoch-based, 5-fold CV | iteration-based, split único |
| otimizador | AdamW + OneCycleLR | SGD + poly (power 0.9) |
| lr / wd | 3e-4 / 1e-5 | 0.01 / 5e-4 |
| loss | `dice_focal_alpha` | Dice puro (denominador quadrático) |
| backbone | EfficientNet-B4 (smp) | EfficientNet-B1, canais `[16,24,40,112,1280]` |
| entrada | 512×512 | 960×1440 |
| seleção de modelo | melhor checkpoint por val Dice | **sem validação** — testa o modelo final |
| métricas | média por imagem | agregadas no dataset inteiro |

`models/wfdenet.py` (que sustenta os experimentos W0/W_* no FGADR) **não foi
tocado** — a réplica fiel vive em `wfdenet_paper.py`. Os dois divergem em sete
pontos, listados no docstring desse arquivo.

## Fonte da verdade

O código oficial (`github.com/xuanli01/WFDENet`, mmsegmentation) foi usado como
referência, não só o texto do paper:

- `mmseg/models/decode_heads/wfdenet_head.py` → `wfdenet_paper.py`
- `mmseg/models/losses/loss.py` → `loss.py`
- `mmseg/evaluation/metrics/{aupr,iou_and_dice}.py` → `metrics_mmseg.py`
- `configs/_base_/models/wfdenet_b1.py` + `configs/WFDENet/ours_idrid.py` → hiperparâmetros

## Protocolo (paper §4.2)

```
backbone     EfficientNet-B1 (ImageNet), out = [16,24,40,112,1280] @ strides 2..32
channels     64          wavelet Haar        num_classes 4 (sigmoid, multilabel)
loss         L = Dice(SD) + 0.5·Dice(LFB) + 0.5·Dice(HFB)     eps 1e-5
otimizador   SGD  lr 0.01  momentum 0.9  weight_decay 0.0005
schedule     poly, power 0.9,  40.000 iterações,  batch 4
entrada      960×1440   (o paper escreve 1440×960 = W×H)
normalização mean [116.513, 56.437, 16.309]  std [80.206, 41.232, 13.293]  (0-255, RGB)
```

Note que a normalização usa **estatísticas do IDRiD**, não do ImageNet.

## Alvos (Tabela 1 do paper)

| | EX | HE | SE | MA | média |
|---|---|---|---|---|---|
| AUPR | 86.61 | 69.05 | 81.28 | 49.91 | **71.71** |
| Dice | 80.42 | 65.73 | 76.02 | 50.31 | **68.12** |
| IoU  | 67.25 | 48.95 | 61.31 | 33.61 | **52.78** |

Validação estrutural (Tabela 4): **9.51M parâmetros**.
Nossa porta dá **9,511,764** — bate.

## Resultados

Run **D1** (commit `307b8de`), comando exato:

```bash
python replication/train_idrid.py --iters 40000 --batch-size 4 --amp --workers 0 \
    --out-dir outputs/repro_wfdenet_idrid
```

| classe | Dice ours | paper | Δ | AUPR ours | paper | Δ | IoU ours | paper | Δ |
|---|---|---|---|---|---|---|---|---|---|
| EX | **80.44** | 80.42 | **+0.02** | 86.67 | 86.61 | +0.06 | 67.29 | 67.25 | +0.04 |
| HE | 64.39 | 65.73 | −1.34 | 69.53 | 69.05 | +0.48 | 47.48 | 48.95 | −1.47 |
| SE | 72.31 | 76.02 | −3.71 | 76.16 | 81.28 | −5.12 | 56.63 | 61.31 | −4.68 |
| MA | 41.39 | 50.31 | **−8.92** | 41.07 | 49.91 | −8.84 | 26.10 | 33.61 | −7.51 |
| **média** | **64.63** | 68.12 | −3.49 | **68.36** | 71.71 | −3.35 | **49.37** | 52.78 | −3.41 |

**EX reproduzido exatamente** (+0.02 Dice, +0.06 AUPR); HE dentro do ruído.
SE a ~4 e **MA a ~9** são o resíduo.

### Histórico das iterações

| run | mudança | mDice | Dice_MA |
|---|---|---|---|
| inicial | máscara `IDRiD_81_EX.tif` RGBA envenenando EX | 51.65 | 41.01 |
| `e1f2f3b` | fix da máscara RGBA | 63.63 | 41.01 |
| `8613fdc` | multi-scale 2880 + ColorJitter + batch 2 | 59.92 ⬇ | 36.52 |
| **`307b8de` (D1)** | revert do ColorJitter; multi-scale sobre o nativo; batch 4 + AMP | **64.63** | 41.39 |

O `8613fdc` **regrediu** e o resultado foi inútil: mudei duas coisas grandes ao
mesmo tempo (augmentation **e** batch 4→2) e a augmentation embutia um
`ColorJitter` que eu havia inventado. Lição registrada: **um fator por run**.

O D1 resolveu EX e HE, mas **MA praticamente não se moveu** (41.01 → 41.39), o
que **refuta** a hipótese de que o borrão do upscale explicava o gap de MA — ela
explicava as lesões grandes, não as pequenas.

## Como rodar

```bash
PY=/home/lucas/.conda/envs/notebook/bin/python   # cv2 precisa vir antes de torch

$PY replication/test_wfdenet_paper.py            # checagens de fidelidade
$PY -m replication.idrid --inspect               # sanidade dos dados

# treino (config fiel atual)
$PY replication/train_idrid.py --iters 40000 --batch-size 4 --amp --workers 0 \
    --out-dir outputs/repro_wfdenet_idrid_tfbb

# reavaliar um checkpoint sem retreinar (~1 min)
$PY replication/train_idrid.py --eval-only --ckpt <path>/final.pth --out-dir <path>
```

Flags relevantes: `--backbone` (default `tf_efficientnet_b1.in1k`),
`--photometric` (**desvio**, off por padrão), `--amp`, `--accum`.

### Memória

Medido a 960×1440, treino (fwd+bwd+step), numa RTX 3060 12GB:

| batch | fp32 | bf16 (`--amp`) |
|---|---|---|
| 1 | 6.0 GB | — |
| 2 | 11.8 GB | 6.9 GB |
| 4 | OOM (>12 GB; extrapolado ~23 GB) | **cabe em 16GB** ✱ |

✱ batch 4 + AMP não foi medido diretamente (estoura os 12GB desta placa), mas
**rodou de ponta a ponta numa GPU de 15.57 GiB** — logo o pico está entre 12 e
15.5 GB. Não temos o número exato.

`--amp` (bf16 autocast, sem GradScaler — bf16 tem o mesmo expoente do fp32) faz
o **batch 4 do paper** caber em 16GB. O CCFAM já força `autocast(False)`
internamente, então a FFT continua em fp32. Alternativa sem AMP:
`--batch-size 2 --accum 2` (batch efetivo 4, mas o BatchNorm vê só 2 por forward).

### Cache em RAM

`IDRiDDataset(cache=True)` (padrão) pré-decodifica **uma vez**: treino guarda o
**nativo 2848×4288** (o multi-scale precisa dele), teste guarda 960×1440.
Custo: ~4.6GB (treino), ~260MB (teste). Com `spawn` cada worker copia o cache
⇒ RAM ≈ 4.6GB × (`workers`+1). **Use `--workers 0`** (o gargalo é a GPU).

### DataLoader

Workers forkados travam contra o thread pool do OpenCV (GPU a 0%, processo
parado). Defesas: `cv2.setNumThreads(0)` em `idrid.py` + `spawn` (mesmo
workaround de `train.py:258`).

## Decisões registradas

**1. Augmentation — o que o paper diz, verbatim (§4.2.2).**

> "we use **three** data augmentation techniques including rotation (90, 180,
> and 270), flipping (horizontal and vertical), and multi-scaling (0.5-2.0)"

São **três**, enumeradas, e **não há augmentation fotométrica**. A frase
seguinte — *"Following [6,7], ... the augmented images are resized to
1440×960"* — prende o "Following [6,7]" ao **resize**, não ao pipeline inteiro.
Nosso pipeline implementa exatamente essas três.

O `train_pipeline` do M2MRF (ref [7], `fcn_hr48-M2MRF-C_40k_idrid_bdice.py`)
*tem* `PhotoMetricDistortion`, e o repo do WFDENet **não publica pipeline de
treino nenhum** para desempatar (é release inference-only). Por isso o PMD
existe como `--photometric`, **off por padrão e rotulado como desvio** — testar
a hipótese sem contaminar a configuração fiel. Um teste anterior com ele veio
confundido com a mudança de batch; refazer isoladamente se interessar.

**Multi-scale sobre o nativo (D1).** `mmcv.imrescale(img_scale=(1440,960),
ratio_range=(0.5,2.0))` opera na **imagem original 2848×4288** — verificado que
o `tools/prepare_labels.py` do M2MRF não redimensiona nada antes. Logo
`sf = 0.3358·r`: r=0.5 → 478×720, r=1.0 → 956×1440, r=2.0 → 1913×2880. É
**sempre downscale** de detalhe nativo, nunca upscale. `_NATIVE_SCALE_LIMIT`
reproduz esses tamanhos. Medido: nitidez (var. do Laplaciano) ~7× maior que a
versão que dava upscale num 960×1440.

**Aproximações declaradas:** flips/rot90 online (dihedral-8) ⊇ as 6 variantes
offline do `tools/augment.py`; `cat_max_ratio=0.75` omitido (com lesões <2% do
quadro o fundo sempre excede 75%, então o mmseg cai no crop aleatório após 10
tentativas — quase-noop, **não testado**); ordem pad→normalize mantida do
`e1f2f3b` para os testes serem single-variable.

**2. Backbone — divergência encontrada e corrigida.**
O repo oficial empacota o próprio backbone
(`mmseg/models/backbones/efficientnet.py`), cujos defaults são:

```python
conv_cfg = dict(type='Conv2dAdaptivePadding')   # padding TensorFlow SAME
norm_cfg = dict(type='BN', eps=1e-3)
```

| | padding | BN eps |
|---|---|---|
| WFDENet oficial (mmpretrain) | **TF SAME** | **1e-3** |
| `timm efficientnet_b1` (runs até D1) | simétrico (1,1) | 1e-5 |
| `timm tf_efficientnet_b1` (default atual) | **Conv2dSame** ✔ | **1e-3** ✔ |

As duas variantes dão estrutura de 5 níveis idêntica e **mesmo param count**
(7.794.184), então o total segue **9.511.764 = 9.51M** do paper. Os pesos `tf_`
vêm da mesma conversão do checkpoint TensorFlow que os pesos 3rdparty do
mmpretrain — é a **correspondência mais próxima disponível**, não provadamente o
mesmo arquivo (ver limitação 7).

**3. Máscaras.** Os `.tif` são palette-mode: `PIL` → `{0,1}`,
`cv2.IMREAD_GRAYSCALE` → `{0,76}`. O `bin_threshold=127` do FGADR
(`dataset.py:161`) zeraria **todas**. Binarizamos com `> 0`.

⚠️ **`IDRiD_81_EX.tif` é a exceção**: único arquivo das 261 máscaras que vem em
**RGBA**. O alpha é 255 em toda a imagem, então `max()` sobre os 4 canais marca
**100% dos pixels** como exsudato. Com métricas agregadas, essa imagem sozinha
era **77% do ground truth de EX no teste** e derrubava o Dice de EX para 30.57.
Correção: `m[..., :3].max(axis=-1)`. Há uma trava (`_assert_masks_sane` em
`train_idrid.py`) que aborta se qualquer canal cobrir >30% da imagem.

**4. Soft Exudates ausentes.** Só 26/54 (treino) e 14/27 (teste) têm máscara SE.
Ausência = canal de zeros, não motivo para descartar a imagem.

**5. Ordem das classes** fixada em `(EX, HE, SE, MA)` para casar coluna a coluna
com a Tabela 1.

**6. Métricas agregadas.** AUPR/Dice/IoU acumuladas sobre as 27 imagens e só
então reduzidas, como em `iou_and_dice.py`/`aupr.py` oficiais. AUPR usa 11
thresholds (`linspace(0,1,11)`) + `sklearn.metrics.auc(recall, precision)`.
O `metrics.py` do repo faz média por imagem e daria números **não comparáveis**.
`AP_sklearn_*` é reportado só como ponte com a série FGADR — **não** vale contra
a Tabela 1.

**7. Sem validação.** O protocolo não tem val set nem seleção de melhor
checkpoint: treina 40k e avalia o modelo final. As avaliações intermediárias
(`--eval-interval`) servem só para a curva.

## Verificado como idêntico (sem ação pendente)

`channels=64`, `align_corners=False`, `dropout_ratio=0`, `bias=True` nos
ConvModule (é o que faz o param count bater exato), loss bdice com `eps=1e-5` e
λ=0.5 nas duas auxiliares, aux upsampled à resolução plena, `test_cfg='whole'`
sem TTA, normalização com stats do IDRiD + `bgr_to_rgb`, `drop_path_rate=0`,
SGD/poly-0.9/40k/batch-4, teste em 960×1440. SyncBN→BN é equivalente em 1 GPU.

## Limitações não resolvíveis com o que foi publicado

1. **Pesos ImageNet exatos.** O config publicado não tem `pretrained`/`init_cfg`
   apontando checkpoint — coerente com um release inference-only (os autores
   distribuem os pesos treinados sob pedido por e-mail). Assumimos init ImageNet
   — **suposição explícita**, já que treinar 9.5M params em 54 imagens do zero
   até mDice 68 é implausível.
2. **Config de treino além do §4.2.2.** Não publicada: sem `train.py`, sem
   `configs/_base_/schedules`, e `configs/_base_/datasets/idrid.py` só tem
   `test_pipeline`.
3. **Run único vs. melhor-de-N.** Desconhecido para o paper; aqui é run único.

**Ressalva estatística.** MA é ~0.1% dos pixels em **27 imagens** de teste —
poucas imagens dominam a métrica agregada. Sem múltiplas seeds, variações de
alguns pontos em MA são ruído, não sinal. (MA é a pior nota do próprio paper na
Tabela 1 — é a classe difícil para todos os métodos.)

## Critério de sucesso

Não é *bater* a Tabela 1, é *reproduzi-la*. EX está exato, HE dentro do ruído,
SE a ~4. Se o gap de MA não fechar com o backbone TF-SAME, a conclusão a
documentar é que ele é **irredutível com os artefatos públicos** (pesos exatos +
config de treino não publicados) — uma limitação declarada, não um bug em
aberto.

Fechando o IDRiD, o mesmo pacote roda em DDR trocando o dataset por
`1024×1024` + 100k iterações.
