"""HL family for HEMORRHAGE: WFDENet's ideas in the H2L1A skip, without its attention.

HL0 is the current best non-WFDENet arm. Add HL1, HL2... here.

Only hemorrhage lives here, under the bare id `HL0`, because outputs/_legado and every
number already written up refer to it by that name. The same architecture for the other
three lesions is generated in `per_class.py` as ex-HL0 / se-HL0 / ma-HL0, and for all
four at once in `multi_lesion.py` as all4-HL0. The split is by ORIGIN (historical ids vs
generated), not by architecture — `python -m experiments` lists them all together.
"""

from config import Config


EXPERIMENTS: dict[str, Config] = {
    # ── Família HL: as ideias do WFDENet trazidas para o skip do H2L1A ───────
    # Passo 1 (HL0): só a RECOMPOSIÇÃO muda. O H2L1A sobe cada sub-banda para a resolução
    # cheia, concatena tudo e resolve num 1x1 — a estrutura wavelet se perde ali. Aqui LF (LL)
    # e HF (cat[LH,HL,HH], bandas misturadas por um conv só, como o HFB do WFDENet) são
    # realçados em separado e o skip é remontado por IDWT de verdade. Sem atenção: isso é o
    # passo 2 (HL1, CCFAM no HF), para que o ganho da recomposição não fique confundido com o
    # ganho do realce. Mesma base do H2L1A em todo o resto (haar, L1, todas as skips).
    # Nota: `wavelet_include_ll` não se aplica — a LL sempre volta, faz parte da IDWT.
    'HL0': Config(
        exp_id='HL0', exp_name='hem_hilo_idwt_L1_allskips',
        classes=('Hemorrhage',),
        loss_type='dice_focal_alpha',
        wavelet_family='haar', wavelet_level=1,
        wavelet_skip_indices=(0, 1, 2, 3),
        wavelet_fusion='hilo',
    ),
}
