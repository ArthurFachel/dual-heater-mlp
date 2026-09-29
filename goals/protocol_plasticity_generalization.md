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

Para um passo de otimização em que a penalidade está ativa:

    Δ_native       = θ_depois − θ_antes, com o termo de penalidade na loss
    Δ_unpenalized  = θ_depois − θ_antes, sem o termo, a partir do MESMO
                     estado (parâmetros e estado do otimizador)

    plasticity_ratio = média_{i : |Δ_unpenalized,i| > 0} ( |Δ_native,i| / |Δ_unpenalized,i| )

A média é **por elemento** sobre toda a superfície de referência, não a média das
médias por tensor. Elementos com `Δ_unpenalized = 0` são **excluídos**: divisão
por zero é ausência de evidência, não plasticidade zero.

Um parâmetro congelado tem `Δ_native = 0` com `Δ_unpenalized ≠ 0`, logo contribui
com **0** — que é o comportamento que o bug R3 não tinha.

### B.1 O lema (independe de dados)

Se a intervenção é um escalonamento diagonal por máscara `m`, então
`Δ_native = m ⊙ Δ_unpenalized`, e portanto

    plasticity_ratio = média(m) = E da máscara

Ou seja: **a métrica primária reduz ao `E` de `surface_plasticity()` no caso em
que ambas se aplicam.** É isso que torna LoRA e EWC comensuráveis e o que
justifica escolher esta entre as três candidatas. Pinado por teste
(`test_ratio_reduces_to_the_mask_mean_for_diagonal_scaling`).

## C. Métricas diagnósticas (reportadas junto, nunca promovidas)

| # | métrica | o que captura |
|---|---|---|
| D1 | `‖Δ_native‖ / ‖Δ_unpenalized‖` | razão de norma; ignora direção |
| D2 | `cos(Δ_native, Δ_unpenalized)` | mudança de direção; é exatamente 1 sob escalonamento diagonal |

D2 é o número que distingue "removeu plasticidade" de "girou o update". Um método
com `plasticity_ratio = 0,6` e `cos = 1,0` é comparável a um `lr_control`; um com
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
| G1 | Métrica primária | razão média por elemento (§B) | única que reduz ao `E` da máscara (§B.1) |
| G2 | Diagnósticas | D1 e D2 (§C) | reportadas sempre, nunca promovidas a primária |
| G3 | Denominador zero | elemento excluído da média | ausência de evidência ≠ plasticidade zero |
| G4 | Escopo da média | por elemento sobre a superfície de referência | média por tensor pondera errado camadas de tamanhos diferentes |
| G5 | Congelado | conta como ratio 0 | é o defeito R3 que esta métrica existe para não repetir |
| G6 | Contrafactual sob AdamW | passo-sombra do mesmo estado | declarado como local (§D.2) |
| G7 | Amostragem | taxa registrada no manifest | custo 3× por passo medido |

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
