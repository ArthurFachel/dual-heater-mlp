# Aplicação 2 — LoRA exato produtor-only

**Fonte:** `build_exact_slowheat_lora`, `src/dual_heater/bert.py:1055`
**Braço equivalente no benchmark:** `exact` em `src/dual_heater/lora_slowheat.py`
**Estado:** **único mecanismo com efeito estatisticamente sobrevivente** nas
duas runs de 10 domínios × 10 seeds, incluindo sob pareamento de plasticidade
e contra o controle de learning rate.

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
`r=16`, medido: **4.214.016 parâmetros treináveis contra 6.769.920** dos demais
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

Contrastes pareados, teste de sinal exato bicaudal. A run iso-plasticidade
(todos os braços em `E = 0,85`) é a principal:

| contraste | métrica | diferença média | p | vitórias |
|---|---|---|---|---|
| **vs lr_control** | **forgetting** | **−0,1190** | **0,0020** | **10/10** |
| vs lr_control | FAA | +0,0810 | 0,0215 | 9/10 |
| vs vanilla | forgetting | −0,0807 | 0,0020 | 10/10 |
| vs vanilla | FAA | +0,0478 | 0,0215 | 9/10 |

Sob Holm, **apenas a redução de forgetting sobrevive** em cada família de
comparações. O ganho de acurácia final não atinge o limiar corrigido.

O contraste contra `lr_control` é o que isola o mecanismo: mesma plasticidade
removida, seletiva contra uniforme. Ele é **maior** que o contraste contra
vanilla, e é o único resultado do projeto que sustenta "a distribuição da
proteção importa, não só a quantidade".

## A ressalva que o pareamento resolveu, e a que permanece

**Resolvida.** Na primeira run este braço tinha `E_eff = 0,923` contra 0,062 do
`slice`, e a ordenação do FAA seguia `E_eff` quase monotonicamente — o efeito
podia ser só "removeu menos plasticidade". Com todos os braços em `E = 0,85` o
efeito **aumentou** (forgetting de −0,063 para −0,081 contra vanilla), o que
descarta essa explicação.

**Permanece.** O custo de aquisição não é corrigido pelo pareamento em E. Este
braço treina metade dos parâmetros do adaptador e chega mais baixo ao fim da
primeira tarefa; `E` mede plasticidade retida da máscara, não capacidade do
adaptador. A leitura *ele esquece menos porque aprendeu menos* segue viva, ainda
que enfraquecida pelo fato de o FAA final ser o **maior** de todos os braços.

## Custo computacional medido

| | valor |
|---|---|
| tempo por seed (10 tarefas) | 10,4 min |
| sobrecusto vs vanilla | 1,76× |
| pico de memória | 2.827,3 MiB (vs 2.759,8 do vanilla) |

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
