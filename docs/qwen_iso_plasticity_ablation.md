# Ablação iso-plasticidade: como a proteção é distribuída importa?

Estado: aritmética implementada e verificada (55 testes). O diagnóstico no
checkpoint real ainda não foi executado.

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

Isto não resgata o endpoint do BERT. `docs/bert_slowheat_diagnostic_results.md`
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
| Jaccard do conjunto protegido | 0,890 |
| turnover do pool livre | 0,296 |

Leituras:

- **densidade 1,000** significa que nenhuma unidade fica sem utilidade. O budget
  é o único limitador da proteção, confirmando o que o smoke sugeriu;
- **PR agregado de 11,5% contra PR por camada de 63-84%** não é contradição: é a
  assinatura de magnitudes heterogêneas entre camadas. Foi esse contraste que
  revelou o bug de escopo;
- **duas camadas anômalas**: camada 3 (PR = 95) e camada 21 (PR = 40) de 4864
  concentram importância em 1-2% das unidades, enquanto as outras 22 usam 63-84%.
  Achado estrutural independente do bug, e ainda sem explicação;
- **Jaccard 0,890 após uma única fronteira** indica que o conjunto protegido
  fossiliza rápido. Com 10 tarefas, o pool livre tende a convergir para unidades
  nunca úteis. É o risco previsto na seção 3 do documento de calibração, agora
  medido.

Os `beta` das famílias iso-plasticidade precisam ser remedidos sob escopo local;
os valores da primeira execução estão descartados.
