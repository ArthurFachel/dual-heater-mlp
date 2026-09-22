# Calibração de capacidade do SlowHeat: critério declarado

Estado: aritmética implementada e verificada. O diagnóstico no checkpoint real
ainda não foi executado.

## 1. Por que não se calcula `beta` e `budget` diretamente

Não há derivação de primeiros princípios para `slow_strength` e
`ffn_plasticity_budget`. O que existe é uma escolha entre dois procedimentos
legítimos:

1. calibrar em dataset disjunto (BANKING77, HWU64) e congelar para o CLINC150;
2. declarar um critério que leia **apenas** quantidades de mecanismo, fixá-lo
   antes da run e auditá-lo depois.

Este documento implementa o segundo. Escolher os valores olhando FAA no
CLINC150 e depois reportar FAA no CLINC150 é o post-hoc que
`docs/bert_slowheat_diagnostic_results.md` proíbe explicitamente.

## 2. Correção: contagem de protegidos superestima o dano

O smoke reportou `min=max=3648` protegidas de 4864 por camada, o que parece
dizer "75% da rede congelada". Está errado.

`slow_heat` é normalizado para [0,1] e a modulação é
`lr_scale = 1/(1 + beta*h)`. Uma unidade "protegida" com `h = 0.01` mantém 97%
da taxa de aprendizado em `beta = 3`. A quantidade que governa o quanto o
otimizador ainda consegue mover é a **plasticidade efetiva**:

```text
E(beta, b) = (1/N) * soma_i [ 1 / (1 + beta * h_i) ]
```

Limite inferior para o caso observado, supondo o pior cenário em que todas as
3648 protegidas tivessem `h = 1`:

```text
(1216 * 1.00 + 3648 * 0.25) / 4864 = 0.4375
```

Ou seja, a plasticidade efetiva é no mínimo 0,44, não 0,25. A fração livre
`b = 0.25` é um piso, nunca o valor real.

## 3. Quantidades medidas

| Quantidade | Definição | O que diz |
|---|---|---|
| `positive_fraction(m)` | fração de unidades com `m > 0` | densidade do sinal |
| `participation_ratio(m)` | `(soma m)^2 / soma(m^2)` | número efetivo de unidades que carregam o sinal |
| `effective_plasticity(h, beta)` | `média 1/(1 + beta*h)` | plasticidade que sobra |
| `jaccard` entre máscaras | sobreposição entre tarefas | churn do conjunto protegido |

O churn é o risco que a contagem esconde. Sob consolidação por `max`,
`importance_memory` é monotonicamente não-decrescente. A contagem de livres fica
fixa em `b*N`, mas a **identidade** das livres converge para "unidades que nunca
foram úteis para nada". A contagem não degrada; a qualidade do pool livre
degrada. Por isso o diagnóstico reporta `protection_jaccard_vs_previous` e
`free_pool_turnover` por estágio.

## 4. O critério

```text
escolher o maior beta tal que E(beta, b) >= piso, no budget b declarado
```

Implementado em `select_by_declared_criterion`. Propriedades:

- lê somente mecanismo, nunca acurácia;
- avaliado analiticamente sobre um vetor de importância fixo, então a grade
  inteira custa alguns sorts, não uma run por candidato;
- **falha em vez de relaxar** o piso quando nenhum candidato qualifica. Relaxar
  silenciosamente seria o post-hoc que o critério existe para evitar.

## 5. Limitação importante do critério

O critério pressupõe que `E` seja sensível a `beta` na faixa de interesse. Isso
depende da **forma** da distribuição de importância, e a sensibilidade colapsa
quando a distribuição é de cauda pesada.

Valores de `E` em três formas sintéticas, 24 x 4864 unidades, `b = 0.25`:

| Forma | PR (% das unidades) | E(beta=1) | E(beta=3) | E(beta=30) | E(beta=100) |
|---|---:|---:|---:|---:|---:|
| uniforme | 75,1% | 0,720 | 0,525 | 0,293 | 0,264 |
| lognormal(0,1) | 36,4% | 0,984 | 0,956 | 0,747 | 0,551 |
| lognormal(0,2) | 2,0% | 0,999 | 0,998 | 0,985 | 0,960 |

Leitura: sob cauda muito pesada, quase toda a massa de `h` está em pouquíssimas
unidades, o resto das protegidas tem `h` desprezível, e `E` fica acima de 0,96
até `beta = 100`. Um piso de 0,6 selecionaria `beta = 100` sem que isso
signifique proteção branda — significa que a métrica ficou cega.

Consequências práticas, a decidir **antes** da run:

- o piso só é informativo se a importância real não for de cauda extrema.
  `participation_ratio` mede isso, e é reportado por estágio;
- se `PR / N` for muito baixo (digamos abaixo de 5%), o critério de `E` deve ser
  acompanhado de um segundo guard-rail, por exemplo um teto em `beta` ou um piso
  em `E` calculado apenas sobre o subconjunto protegido;
- a grade e o piso devem ser registrados no manifesto junto com `PR`, para que a
  auditoria posterior mostre se o critério estava em regime informativo.

Essas três formas são sintéticas. A distribuição real do Qwen é desconhecida até
o diagnóstico rodar, e é justamente o que ele mede primeiro.

## 6. Verificação

`tests/test_capacity_calibration.py`, 43 testes, todos com valores esperados
derivados à mão a partir das definições.

Dez mutações injetadas no código de produção; oito foram detectadas:
`floor` -> `ceil`, budget não invertido, fórmula de plasticidade trocada,
razão de participação sem o quadrado, jaccard com denominador errado, critério
escolhendo o menor `beta`, critério relaxando o piso silenciosamente, critério
ignorando o filtro de budget.

Duas sobreviveram e foram investigadas: normalizar por `selected.max()` versus
`max` global, e o teto pela contagem de positivos. Ambas são **equivalências
provadas**, não lacunas: `argsort` descendente garante que o máximo global está
sempre no conjunto selecionado, e entradas extras acima da contagem positiva são
zeros, com `0/max = 0`. Verificado em 2000 casos aleatórios com zeros
deliberados, divergência máxima 0,0. Os dois guards ficam como código defensivo
e a equivalência está fixada em
`test_selected_maximum_equals_the_global_maximum_by_construction`.

## 7. Como rodar

```sh
cd /mnt/B-SSD/fachel/dual-heater-mlp
CUDA_VISIBLE_DEVICES= HF_HOME=.hf-cache PYTHONPATH=. \
  .venv/bin/python experiments/qwen_capacity_diagnostic.py \
    --tasks 2 --budget 0.25 --min-effective-plasticity 0.6
```

Exige o CLINC150 via `datasets` (download ainda não feito). O script grava
`results/qwen_capacity_diagnostic/manifest.json` com critério, protocolo,
estatísticas por estágio, grade completa e o ponto selecionado.

O script não tem acesso a nenhum loop de avaliação: não computa acurácia,
retenção nem FAA. Isso é deliberado.
