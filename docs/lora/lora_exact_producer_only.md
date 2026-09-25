# Aplicação 2 — LoRA exato produtor-only

**Fonte:** `build_exact_slowheat_lora`, `src/dual_heater/bert.py:1055`
**Braço equivalente no benchmark:** `exact` em `src/dual_heater/lora_slowheat.py`
**Estado:** **único mecanismo com efeito estatisticamente sobrevivente** no
benchmark de 10 domínios × 10 seeds.

## Claim, com o escopo explícito

> Congelando `lora_A` e mascarando linhas de `lora_B`, a linha de update
> `delta_weight[i, :] = B[i, :] @ A` de uma saída `i` com máscara 0 é
> **exatamente invariante** a um passo do otimizador, independentemente do que
> acontece com as outras saídas.

O escopo é essencial: a exatidão vale **por saída**, e ao preço de `A` fixa.

## Por que funciona

Com `A` congelada, `delta_weight[i, :]` depende de exatamente um conjunto de
parâmetros treináveis — a linha `B[i, :]`. Zerar a plasticidade dessa linha
zera o único canal de mudança. O vazamento da [aplicação 1](lora_dualheat_legacy.md)
existia porque `A` era o canal compartilhado; congelando `A`, ele desaparece
por construção, não por aproximação.

Formalmente, para máscara `m[i] = 0`:

```text
B_novo[i, :] = B[i, :] - lr · m[i] · adam_step  =  B[i, :]
A_novo       = A                                      (congelada)
=> delta_weight_novo[i, :] = B[i, :] @ A = delta_weight[i, :]
```

## Custo: metade dos parâmetros treináveis

Congelar `A` remove metade da capacidade do adaptador. No Qwen2.5-0.5B com
`r=16`, medido: **2.174.208 parâmetros treináveis contra 3.452.160** dos demais
braços (−37%, contando a cabeça de classificação que todos treinam).

Esse custo aparece na acurácia da **primeira** tarefa, antes de existir
qualquer coisa a proteger. Na run de 2 domínios, com `r=8`:

| arm | acc. após T0 | acc. após T1 |
|---|---|---|
| exact | 0,860 | 0,440 |
| todos os outros | 0,963 | 0,343–0,437 |

O `exact` chega 10 pontos abaixo ao fim de T0. Ele paga plasticidade de
aquisição desde o passo zero.

## Resultado no benchmark (10 domínios, 10 seeds)

Contraste pareado contra vanilla, teste de sinal exato bicaudal:

| métrica | diferença média | p | vitórias |
|---|---|---|---|
| forgetting | **−0,0629** | **0,0020** | 10/10 |
| FAA | +0,0349 | 0,1094 | 8/10 |

Sob Holm sobre as 8 comparações do sweep, **apenas a redução de forgetting
sobrevive** (0,0020 < 0,00625). O ganho de acurácia final não é significativo.

O sinal é consistente — 10 de 10 seeds na direção esperada, sem exceção.

## Ressalva que impede a leitura causal

Este braço rodou com `E_eff = 0,923`, enquanto `rank` e `leak` rodaram com
~0,65 e `slice` com 0,062. Os braços **não** foram pareados em plasticidade
retida, e a ordenação do FAA segue `E_eff` quase monotonicamente.

Ou seja: parte do resultado do `exact` pode vir de ele ter removido menos
plasticidade, não de a proteção ser melhor distribuída. Somado ao custo de
aquisição documentado acima, há uma leitura alternativa não descartada:
*ele esquece menos porque aprendeu menos*.

A run iso-plasticidade (`results/qwen_lora_iso_10seed/`) foi desenhada para
separar as duas coisas, pareando todos os braços em `E = 0,85` e incluindo o
braço `lr_control`.

## Custo computacional medido

| | valor |
|---|---|
| tempo por seed (10 tarefas) | 17,7 ± 4,1 min |
| sobrecusto vs vanilla | 1,67× |
| pico de memória | 2.827,7 MiB (vs 2.759,8 do vanilla) |

O sobrecusto é dominado pelo passo pontual mascarado do `SlowHeatAdamW`, que
percorre parâmetro a parâmetro em Python sem `foreach`.

## Quando usar

Quando a exatidão por saída for requisito e a perda de metade da capacidade do
adaptador for aceitável — tipicamente com `r` maior para compensar. Se `A`
treinável for necessária, veja o mecanismo `leak` da
[aplicação 3](lora_slowheat_mechanisms.md).

## Referências

- [Índice das aplicações de LoRA](lora_applications.md)
- [Resultados completos do benchmark](lora_qwen_benchmark_results.md)
- [SlowHeat em Transformers, Solução A](../mechanisms/functional_slowheat_transformers.md)
