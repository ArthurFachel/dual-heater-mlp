# Ablação iso-plasticidade: como a proteção é distribuída importa?

Documento único da ablação Qwen. O corpo principal define o desenho, os braços
e os achados; o **Anexo A** define a aritmética de capacidade e o critério
declarado que alimenta os braços (antes em `qwen_capacity_calibration.md`).

Estado: aritmética implementada e verificada (55 testes na ablação, 43 na
calibração). O diagnóstico no
checkpoint real **foi executado**: `results/qwen_capacity_diagnostic/manifest.json`
(22/09, escopo local, 2 tarefas, seed 0), 6 runs de calibração em
`results/qwen_iso_plasticity/` e 6 runs de ordem/seed em
`results/qwen_layer_anomaly/`.

Ressalvas que impedem tratar esses números como resultado:

- todas as runs existentes usam **30 passos por tarefa**, valor revogado pela
  tabela K de `goals/protocol_iso_plasticity.md` em favor de 120; os endpoints
  de acurácia dessas runs estão obsoletos;
- não há agregação entre seeds, nem `aggregate.json`, nem diferenças pareadas;
- o critério declarado do Anexo A não foi aplicado: o
  manifesto grava `minimum_effective_plasticity: null`,
  `declared_before_run: false` e `selection: null`;
- a confirmação de 10 seeds × 10 domínios × 120 passos ainda não produziu
  nenhum manifesto.

## 1. Por que isto é uma ablação e não uma seleção

Um critério que escolhe um ponto `(beta, budget)` produz uma linha numa tabela e
nada mais. Não dá para discorrer sobre ele: a pergunta "o critério escolheu
bem?" só se responde olhando acurácia, que é exatamente o que o critério
proíbe.

A alternativa é tornar a capacidade o **eixo** do experimento. Isso exige
separar duas coisas que a literatura de CL costuma confundir:

- **quanta** plasticidade foi removida;
- **como** essa remoção foi distribuída entre as unidades.

O SlowHeat tem dois botões, e a ingenuidade é tratá-los como intercambiáveis:
`beta` (força da proteção) e `budget` (quantas unidades ficam livres). A seção 3
mostra que eles não são intercambiáveis, e a seção 4 usa esse fato para montar
braços de custo pareado.

## 2. A confusão que a ablação remove

Comparar `beta = 3` contra `beta = 30` não é um contraste limpo. `beta = 30`
remove mais plasticidade **total**, então qualquer perda de aquisição da tarefa
nova é explicável por "o modelo simplesmente tinha menos capacidade de
aprender". O diagnóstico de BERT tem exatamente esse formato, e é por isso que a
conclusão lá só pode ser "mais proteção troca aquisição por retenção" — uma
afirmação sobre a quantidade, não sobre o mecanismo.

A pergunta mecanística é outra: **dada a mesma quantidade de plasticidade
removida, importa se ela foi tirada de muitas unidades fracamente ou de poucas
unidades fortemente?**

Se importar, o ranking de importância funcional carrega informação estrutural
além da magnitude. Se não importar, o mecanismo é equivalente a uma redução
global de learning rate com máscara arbitrária, e isso é um resultado negativo
publicável e honesto.

## 3. Resultado analítico: `beta` e `budget` não são intercambiáveis

Com `h` normalizado em [0,1] e `lr_scale = 1/(1 + beta*h)`:

```text
E(beta, b) = (1/N) * soma_i [ 1 / (1 + beta * h_i) ]
```

Unidades livres têm `h_i = 0` e contribuem exatamente 1 cada. Logo, para `P`
unidades protegidas:

```text
E(beta, b) >= (N - P) / N   para todo beta
```

O limite é justo: quando `beta -> infinito` todo termo protegido desaparece e
`E -> (N-P)/N`. Verificado em 3000 casos aleatórios, zero violações, e o limite
confere com 6 casas decimais em `beta = 1e12`.

Consequências:

- o **budget** define um piso rígido de plasticidade; `beta` só move `E` dentro
  de `[piso, 1]`;
- uma plasticidade-alvo `E*` é **inalcançável** em qualquer budget cujo piso já
  seja maior que `E*`. Por isso `solve_strength_for_plasticity` retorna `None`
  em vez de saturar em `beta` máximo — saturar reportaria uma configuração que
  não existe;
- portanto existe uma família de configurações com `E` idêntico e distribuição
  diferente, e ela é construtível exatamente, não por busca em grade.

## 4. O design

Para uma plasticidade-alvo `E*` fixada antes da run, resolver `beta` por
bisseção em cada budget alcançável. Cada braço tem a mesma plasticidade efetiva
e uma distribuição diferente:

| Braço | budget | unidades protegidas | beta | E |
|---|---:|---:|---:|---:|
| muitas-fracas | 0,05 | 95% | menor | `E*` |
| intermediário | 0,25 | 75% | médio | `E*` |
| poucas-fortes | 0,50 | 50% | maior | `E*` |

Spread de contagem entre os extremos: `(1-0.05)/(1-0.50) = 1,9x`. O spread de
`beta` é maior e depende da forma da distribuição de importância.

Controles necessários, todos com o mesmo `E*`:

- **hard aleatório:** mesma contagem protegida, identidades sorteadas. Separa
  "o ranking informa" de "proteger qualquer coisa já basta".
- **LR reduzido global:** `lr * E*`, sem máscara nenhuma. Este é o controle que
  o diagnóstico de BERT não tinha e é o mais importante: se o braço vencedor não
  superar uma simples redução uniforme de learning rate na mesma plasticidade
  efetiva, o mecanismo não está fazendo nada estrutural.
- **vanilla:** `E = 1`, âncora de forgetting.

O que se reporta: FAA, retenção, aquisição e drift, por braço, com diferenças
pareadas por seed. A hipótese fica declarada antes: *se a distribuição não
importa, os três braços iso-E ficam dentro do ruído entre seeds, e todos ficam
dentro do ruído do LR reduzido.*

## 5. Limitação que decide a viabilidade

A sensibilidade de `E` a `beta` colapsa sob importância de cauda pesada. Valores
para 24 x 4864 unidades em `b = 0.25`, com `beta` resolvido para `E* = 0.75`:

| Forma da importância | PR/N | beta necessário |
|---|---:|---:|
| uniforme | 75,1% | 0,83 |
| lognormal(0,1) | 36,9% | 32,4 |
| lognormal(0,2) | 2,1% | 1213,2 |

Sob cauda extrema, quase toda a massa de `h` está em pouquíssimas unidades, o
resto das protegidas tem `h` desprezível, e é preciso `beta` na casa dos
milhares para mover `E`. Nesse regime a ablação continua **válida** (os braços
ainda têm `E` idêntico por construção), mas os `beta` ficam absurdos e a
interpretação "força de proteção" perde sentido físico.

`participation_ratio` mede isso e é reportado por estágio. Decisão a tomar antes
da run: se `PR/N` ficar abaixo de uns 5%, a família iso-E deve ser construída
sobre `h` renormalizado (por exemplo `h / percentil_90`) em vez de `h / max`,
ou a ablação deve trocar de eixo para a contagem protegida com `beta` fixo.

As três formas acima são sintéticas. A distribuição real do Qwen é desconhecida
até o diagnóstico rodar, e medi-la é o primeiro passo.

## 6. O que isto permite discorrer

Quatro afirmações possíveis, cada uma com um contraste que a sustenta:

1. **"O budget domina, `beta` é secundário"** — sustentado pelo piso
   `E >= (N-P)/N`, que é analítico e independe de dados.
2. **"A distribuição da proteção importa/não importa"** — braços iso-E entre si.
3. **"O ranking funcional carrega sinal"** — iso-E aprendido contra iso-E
   aleatório.
4. **"O mecanismo não é redutível a LR menor"** — qualquer braço contra LR
   reduzido na mesma plasticidade efetiva.

A afirmação 1 já está provada e é o achado teórico do trabalho. As outras três
dependem da run.

## 7. Honestidade sobre o escopo

Isto não resgata o endpoint do BERT. `docs/results/bert_slowheat_diagnostic_results.md`
concluiu que o SlowHeat não superou replay e que não se deve abrir nova grade
para o mesmo endpoint. A ablação iso-plasticidade pergunta outra coisa: não "o
SlowHeat é competitivo?", mas "o mecanismo de proteção seletiva tem efeito
estrutural além da redução de capacidade que ele causa?". Um resultado negativo
aqui é informativo e publicável, e é por isso que o controle de LR reduzido é
obrigatório e não opcional.

## 8. Verificação

`tests/test_capacity_calibration.py`, 55 testes. Cinco mutações injetadas nas
funções novas (direção da bisseção invertida, alvo inalcançável saturando em vez
de `None`, `free_fraction` contando protegidas, família não descartando budgets
inalcançáveis, limite inferior fixado em zero); todas as cinco detectadas.

## 9. Correção: escopo de capacidade é parte do mecanismo

A primeira execução do diagnóstico no Qwen real reportou `beta` na casa das
centenas (266 a 593 para `E* = 0,75`). Esses valores estavam errados por
aproximadamente duas ordens de magnitude.

**Causa.** `_pooled_importance` concatenava as 24 camadas e a aritmética
normalizava o pool inteiro por um único máximo global. O modelo, porém, roda com
`capacity_scope="local"` (default), que rankeia e normaliza **cada camada
independentemente**.

As camadas do Qwen têm magnitudes de importância muito diferentes. Sob
normalização local, uma camada de baixa magnitude ainda atinge `h = 1,0` dentro
de si mesma; sob normalização agregada, ela recebe `h` próximo de zero e some da
conta. Reproduzido com estrutura sintética equivalente, `budget = 0,25`:

| beta | E agregado (errado) | E local (correto) |
|---:|---:|---:|
| 3 | 0,903 | 0,503 |
| 30 | 0,723 | 0,287 |
| 100 | 0,608 | 0,262 |

A média de heat difere por fator ~8x. Consequência: o mecanismo é muito **mais**
sensível a `beta` do que o diagnóstico indicava, e a preocupação com "beta
absurdo" da seção 5 não se aplica ao regime local.

**O erro conceitual**, não apenas de código: tratei "importância agregada" como
grandeza única, quando o escopo de capacidade faz parte da definição do
mecanismo. Uma função que recebe um vetor único só pode responder sobre **um**
grupo de normalização.

**Correção.** Adicionadas variantes `*_scoped` que recebem a lista de vetores por
camada mais o escopo: `protected_heat_scoped`, `free_fraction_scoped`,
`plasticity_bounds_scoped`, `solve_strength_for_plasticity_scoped`,
`iso_plasticity_family_scoped`, `sweep_capacity_scoped`. O runner usa apenas
essas, expõe `--capacity-scope` e registra o escopo no manifesto. As versões de
vetor único permanecem para estatísticas que não normalizam
(`positive_fraction`, `participation_ratio`).

`test_scoped_heat_matches_the_production_mechanism` compara as três variantes
contra `apply_family_capacity` real, com camadas cujas magnitudes diferem por
quatro ordens de magnitude. Cinco mutações injetadas nas funções novas, todas
detectadas, incluindo a que reintroduz exatamente o bug original.

## 10. Distribuição real medida no Qwen

Duas tarefas do CLINC150 (`banking -> credit_cards`), 30 passos cada,
`max_length = 64`, seed 0. Valores válidos porque não dependem de normalização:

| Quantidade | Valor |
|---|---|
| unidades totais (24 x 4864) | 116.736 |
| densidade de sinal | 1,000 em todas as camadas |
| PR agregado | 13.430 (11,5% das unidades) |
| PR por camada, típico | 3.000 a 4.100 de 4864 (63 a 84%) |
| protegidas por camada, `b = 0,25` | 3.648 (saturado) |
| loss estágio 0 | 16,19 -> 2,06 |
| loss estágio 1 | 11,67 -> 0,55 |
| Jaccard do conjunto protegido | 0,870 |
| turnover do pool livre | 0,345 |

Leituras:

- **densidade 1,000** significa que nenhuma unidade fica sem utilidade. O budget
  é o único limitador da proteção, confirmando o que o smoke sugeriu;
- **PR agregado de 11,5% contra PR por camada de 63-84%** não é contradição: é a
  assinatura de magnitudes heterogêneas entre camadas. Foi esse contraste que
  revelou o bug de escopo;
- **duas camadas anômalas**: camada 3 (PR = 95) e camada 21 (PR = 40) de 4864
  concentram importância em 1-2% das unidades, enquanto as outras 22 usam 63-84%.
  Achado estrutural independente do bug, e ainda sem explicação;
- **Jaccard 0,870 após uma única fronteira** indica que o conjunto protegido
  fossiliza rápido. Com 10 tarefas, o pool livre tende a convergir para unidades
  nunca úteis. É o risco previsto na seção 3 do documento de calibração, agora
  medido.

Os `beta` das famílias iso-plasticidade precisam ser remedidos sob escopo local;
os valores da primeira execução estão descartados.

---

# Anexo A — Calibração de capacidade: o critério declarado

Esta seção era `docs/qwen_capacity_calibration.md`. Ela define a aritmética de
capacidade e o critério que alimenta os braços da ablação acima, e por isso não
faz sentido separada dela.

Estado: aritmética implementada e verificada (43 testes). O critério **nunca foi
aplicado numa run**: `results/qwen_capacity_diagnostic/manifest.json` grava
`minimum_effective_plasticity: null`, `declared_before_run: false` e
`selection: null`.

## A.1. Por que não se calcula `beta` e `budget` diretamente

Não há derivação de primeiros princípios para `slow_strength` e
`ffn_plasticity_budget`. O que existe é uma escolha entre dois procedimentos
legítimos:

1. calibrar em dataset disjunto (BANKING77, HWU64) e congelar para o CLINC150;
2. declarar um critério que leia **apenas** quantidades de mecanismo, fixá-lo
   antes da run e auditá-lo depois.

Este anexo implementa o segundo. Escolher os valores olhando FAA no
CLINC150 e depois reportar FAA no CLINC150 é o post-hoc que
[`bert_slowheat_diagnostic_results.md`](../results/bert_slowheat_diagnostic_results.md)
proíbe explicitamente.

## A.2. Correção: contagem de protegidos superestima o dano

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

## A.3. Quantidades medidas

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

## A.4. O critério

```text
escolher o maior beta tal que E(beta, b) >= piso, no budget b declarado
```

Implementado em `select_by_declared_criterion`. Propriedades:

- lê somente mecanismo, nunca acurácia;
- avaliado analiticamente sobre um vetor de importância fixo, então a grade
  inteira custa alguns sorts, não uma run por candidato;
- **falha em vez de relaxar** o piso quando nenhum candidato qualifica. Relaxar
  silenciosamente seria o post-hoc que o critério existe para evitar.

## A.5. Limitação importante do critério

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

Essas três formas são sintéticas. **A distribuição real do Qwen já foi medida** e
está na seção 10 acima: duas camadas anômalas com PR de 1-2% das unidades, e as
outras 22 entre 63% e 84%. Ou seja, o regime de cauda extrema previsto aqui
ocorre de fato, mas só em duas das 24 camadas.

## A.6. Verificação

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

## A.7. Como rodar o diagnóstico de capacidade

```sh
cd /mnt/B-SSD/fachel/dual-heater-mlp
CUDA_VISIBLE_DEVICES= HF_HOME=.hf-cache PYTHONPATH=. \
  .venv/bin/python experiments/qwen_capacity_diagnostic.py \
    --tasks 2 --budget 0.25 --min-effective-plasticity 0.6
```

O script grava `results/qwen_capacity_diagnostic/manifest.json` com critério,
protocolo, estatísticas por estágio, grade completa e o ponto selecionado.

O script não tem acesso a nenhum loop de avaliação: não computa acurácia,
retenção nem FAA. Isso é deliberado.
