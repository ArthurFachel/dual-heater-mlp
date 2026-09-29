# Protocolo congelado — generalização da plasticidade efetiva para métodos de penalidade

**Congelado em:** 2026-09-29, antes de qualquer implementação e de qualquer seed.
**Motiva:** Fase 2 de `goals/roadmap_icml_ijcnn.md`.
**Antecedente:** `goals/protocol_plasticity_matched.md` e o R3 de
`docs/lora/lora_confirmation_results.md`.

## A. O problema

`surface_plasticity()` (`src/dual_heater/lora_slowheat.py:103`) mede a fração
média do update preservada sobre a superfície treinável de um braço de
referência. Ela assume que a intervenção é um **escalonamento diagonal** do
update: cada parâmetro tem seu update multiplicado por um fator em [0, 1].

Isso vale para máscara de LoRA e para congelamento. **Não vale para EWC/SI/MAS**,
que somam `λ Ω (θ − θ*)` ao gradiente: o update resultante pode mudar de direção,
não apenas de módulo.

Sem uma métrica comum, a frase "a literatura de CL não verifica pareamento de
plasticidade" não é testável fora do LoRA, e a Seção 4 do ICML não existe.

## B. Métrica primária (congelada)

> **EMENDADO em 2026-09-29** — ver §K linha 2 e §B.2. A métrica declarada
> originalmente (razão média por elemento) é mal-condicionada para penalidades
> aditivas e foi rebaixada a diagnóstica. O texto original está preservado em
> §B.0 porque um pré-registro que apaga o que dizia não é um pré-registro.

Para um passo de otimização em que a penalidade está ativa:

    Δ_native       = θ_depois − θ_antes, com o termo de penalidade na loss
    Δ_unpenalized  = θ_depois − θ_antes, sem o termo, a partir do MESMO
                     estado (parâmetros e estado do otimizador)

    plasticity_norm_ratio = ‖Δ_native‖ / ‖Δ_unpenalized‖

(norma L2 sobre a superfície de referência concatenada, não a média das normas
por tensor).

Um parâmetro congelado tem `Δ_native = 0` e portanto reduz o numerador sem
tocar no denominador — contribui para baixar a razão, que é o comportamento que
o bug R3 não tinha.

### B.0 A métrica original, rebaixada (registro histórico)

    plasticity_ratio = média_{i : |Δ_unpenalized,i| > 0} ( |Δ_native,i| / |Δ_unpenalized,i| )

Continua implementada, continua reportada, **não é mais a primária**. Por quê,
em §B.2.

### B.1 O lema (independe de dados)

Se a intervenção é um escalonamento diagonal **uniforme** `m_i = c`, então
`Δ_native = c · Δ_unpenalized`, e portanto

    plasticity_norm_ratio = c = média(m) = E da máscara

Ou seja: **a métrica primária reduz ao `E` de `surface_plasticity()` no caso do
`lr_control`**, que é exatamente o braço contra o qual o pareamento deste
piloto é feito. Pinado por
`test_norm_ratio_reduces_to_the_mask_mean_for_a_UNIFORM_mask`.

**O que o lema NÃO cobre, dito explicitamente:** para uma máscara
**heterogênea**, `‖m ⊙ Δ‖/‖Δ‖` é uma média quadrática ponderada pela magnitude
do update, e **não** a média aritmética de `m`. Máscara `[1,1,0,0]` sobre
denominador homogêneo dá `sqrt(0,5) = 0,707`, não `0,5`. Logo a comensurabilidade
com o `E` de uma máscara heterogênea de LoRA **foi perdida** nesta emenda. A
razão por elemento (§B.0) preserva essa propriedade e por isso continua
reportada. Pinado por
`test_norm_ratio_does_NOT_equal_the_mask_mean_for_a_heterogeneous_mask`.

### B.2 Por que a primária mudou (evidência mecanismo-only, sem acurácia)

A seed de calibração (`7999991`, fora da banda confirmatória, §I) mediu na
passada 1:

| arm | razão por elemento | norma | cos |
|---|---:|---:|---:|
| `ewc` | **1,0956** | 0,9956 | 0,9994 |
| `si` | 0,9989 | 0,9997 | 1,0000 |
| `mas` | **1,7759** | 0,9455 | 0,9784 |

Dois arms com razão por elemento **maior que 1**. O §C manda construir o
`lr_control` com `lr × E`, e `lr_scale > 1` não é um controle de plasticidade —
é um **aumento** de learning rate. `surface_plasticity()`
(`src/dual_heater/lora_slowheat.py:143`) recusa `lr_scale` fora de `[0, 1]` por
esse motivo. A métrica, como declarada, **não produzia um pareamento
construível**.

A causa é aritmética, não numérica. A regra G3 exclui denominador *exatamente*
zero mas mantém denominadores arbitrariamente pequenos; cada um vira um outlier
ilimitado, e a média por elemento (G4) dá a eles o mesmo peso de um parâmetro
que de fato se moveu. Disseção no MNIST real (MAS, AdamW): mediana `0,98`,
norma `0,97` — o update **encolheu** — com máximo por elemento `3.436,7`, e o
decil de menor denominador (`|Δ| < 5,01e-06`) respondendo por **17,9%** da
média. Em fixture sintética com mais passos o mesmo decil chega a **31,4%**,
com razão média `4,04`.

`direction_cosine` ficou em `0,978–1,000` nos três métodos: a intervenção é
**quase diagonal** neste host, ou seja o pareamento É enunciável e o problema
estava na estatística escolhida, não na geometria do método.

**Nenhum endpoint de acurácia foi lido para tomar esta decisão.** A run de
calibração é `n = 1`, fora da banda confirmatória, e
`scripts/run_penalty_calibration.py` não importa nem imprime acurácia.

## C. Métricas diagnósticas (reportadas junto, nunca promovidas)

| # | métrica | o que captura |
|---|---|---|
| D1 | razão média por elemento (§B.0) | comensurável com o `E` de máscara heterogênea; ilimitada acima |
| D2 | `cos(Δ_native, Δ_unpenalized)` | mudança de direção; é exatamente 1 sob escalonamento diagonal |

D2 é o número que distingue "removeu plasticidade" de "girou o update". Um método
com razão `0,6` e `cos = 1,0` é comparável a um `lr_control`; um com
`cos = 0,3` não é, e o protocolo tem de dizer isso em vez de parear assim mesmo.

## D. A ressalva honesta (vai no artigo, não só aqui)

1. **A generalização não é única.** Para métodos de penalidade o update não é
   escalonamento diagonal. Outras razões defensáveis existem (norma, projeção).
   Declaramos a primária ANTES de ver números; as diagnósticas vão junto.
2. **Sob AdamW o contrafactual é local.** O host MLP usa AdamW
   (`experiments/split_mnist.py:1014`). O estado do otimizador diverge entre os
   dois fluxos de gradiente ao longo do treino, então `Δ_unpenalized` é bem
   definido apenas como "dado o estado atual, qual seria ESTE passo". Medimos por
   passo-sombra a partir do mesmo estado. Não afirmamos que é a trajetória que o
   método teria tido sem penalidade.
3. **Custo.** Cada medição custa 3 passos em vez de 1. Amostramos em vez de medir
   todo passo; a taxa de amostragem entra no manifest.

## E. Decisões congeladas

| # | Decisão | Valor | Justificativa |
|---|---|---|---|
| G1 | ~~Métrica primária~~ | ~~razão média por elemento (§B.0)~~ | **REVOGADA em 29/09, ver G1'** |
| G1' | Métrica primária | razão de normas (§B) | única das duas que produz um `lr_scale` construível (§B.2); reduz ao `E` no caso do `lr_control` (§B.1) |
| G2 | Diagnósticas | D1 (razão por elemento) e D2 (§C) | reportadas sempre, nunca promovidas a primária |
| G3 | Denominador zero | elemento excluído da média de D1 | ausência de evidência ≠ plasticidade zero |
| G4 | Escopo | norma L2 sobre a superfície concatenada | média por tensor pondera errado camadas de tamanhos diferentes |
| G5 | Congelado | `Δ_native = 0`, baixa o numerador | é o defeito R3 que esta métrica existe para não repetir |
| G6 | Contrafactual sob AdamW | passo-sombra do mesmo estado | declarado como local (§D.2) |
| G7 | Amostragem | taxa registrada no manifest | custo 3× por passo medido |
| G8 | Razão primária > 1 | método declarado **não-pareável**; contraste reportado sem `lr_control` | `lr_scale > 1` é aumento de LR, não controle de plasticidade |

## F. O que este protocolo NÃO decide

- Não declara arms, seeds, endpoints nem família confirmatória de nenhum
  experimento. O piloto da Fase 2.4 exige **seu próprio** pré-registro, com banda
  de seeds nova e disjunta.
- Não afirma que métodos de penalidade e LoRA são comparáveis na prática; afirma
  que esta métrica é a que torna a comparação possível de enunciar.

## G. Registro de alterações

| data | alteração |
|---|---|
| 2026-09-29 | documento congelado, antes de qualquer código |
| 2026-09-29 | **Emenda G1 → G1'.** A métrica primária passa de "razão média por elemento" para "razão de normas". **Motivo:** a calibração `n=1` (seed 7999991, fora da banda) mediu razão por elemento `> 1` em `ewc` (1,0956) e `mas` (1,7759), o que torna o `lr_control` do §C não-construível (`lr_scale > 1` é aumento de LR). Causa em §B.2: denominadores pequenos viram outliers ilimitados sob a média aritmética. **Custo aceito:** perde-se a comensurabilidade com o `E` de máscara LoRA *heterogênea* (§B.1), preservada apenas para máscara uniforme — que é o caso do `lr_control`. A métrica original vira D1 e continua reportada. **Nenhuma acurácia foi lida:** o script de calibração não importa nem imprime endpoints de acurácia, e a seed está fora da banda confirmatória. Acrescentado G8 para o caso de a razão primária também passar de 1. Nenhuma seed confirmatória havia sido gasta neste momento. |
