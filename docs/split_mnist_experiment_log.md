# Registro experimental: Functional SlowHeat em Split-MNIST

Este documento consolida os testes realizados no repositório até o momento,
as correções introduzidas durante a investigação e as conclusões que podem ou
não ser sustentadas pelos resultados. Ele deve ser lido como um registro de
pesquisa em andamento, não como uma alegação de estado da arte.

Os números abaixo foram obtidos de CSVs e JSONs produzidos pelo runner e
reportados durante o desenvolvimento. Os artefatos brutos não são versionados
no Git. Para uma publicação, os arquivos por seed, configurações congeladas e
matrizes de acurácia devem ser arquivados junto com a revisão exata do código.

## 1. O que significa `slowheat_replay_hidden_beta_30_budget_0.25`

O nome é uma especificação executável do método:

```text
slowheat_replay_hidden_beta_30_budget_0.25
│        │      │      │       │
│        │      │      │       └── pelo menos 25% da capacidade fica plástica
│        │      │      └────────── força de proteção beta = 30
│        │      └───────────────── protege somente camadas ocultas
│        └──────────────────────── usa memória de replay
└───────────────────────────────── usa Functional SlowHeat
```

### `slowheat`

Functional SlowHeat estima a importância funcional de cada neurônio pela
estatística de primeira ordem:

```text
u_i = |z_i * dL/dz_i|
```

onde `z_i` é a pré-ativação do neurônio e `L` é a loss. O sinal é normalizado
por camada, acumulado durante a tarefa e consolidado na fronteira entre
tarefas. Essa importância é invariável à reescala recíproca das conexões de um
neurônio ReLU positivamente homogêneo.

A importância consolidada gera `slow_heat` em `[0, 1]`. Na proteção suave, o
fator plástico por neurônio é:

```text
m_i = 1 / (1 + beta * slow_heat_i)
```

O otimizador aplica o fator ao delta final produzido por AdamW, depois de
momentum, precondicionamento e weight decay. Com
`optimizer_state_policy="follow_update"`, o mesmo fator interpola os estados
`exp_avg` e `exp_avg_sq`.

A proteção é fatorada e bidirecional entre camadas SlowHeat consecutivas:

```text
M_l[i,j] = min(m_output_l[i], m_output_l_minus_1[j])
```

Assim, a importância de um neurônio protege sua linha de entrada e as colunas
correspondentes na camada SlowHeat seguinte.

### `replay`

O método mantém uma memória episódica balanceada das classes anteriores. Na
configuração usada nos experimentos:

- `replay_per_class=20`;
- `replay_batch_size=64`;
- o minibatch de replay é concatenado ao minibatch da tarefa atual;
- replay e SlowHeat+replay usam a mesma memória e os mesmos índices
  determinísticos por seed.

Replay fornece exemplos antigos reais durante a aprendizagem da tarefa atual.
Os experimentos mostraram que essa informação é essencial: SlowHeat sozinho
não substitui replay neste protocolo.

### `hidden`

Somente as camadas ocultas são `SlowHeatLinear`. A cabeça de saída é uma
`nn.Linear` comum e permanece totalmente plástica.

Isso é importante em class-incremental learning. A cabeça global precisa
recalibrar a competição entre classes antigas e novas. Proteger a saída pode
preservar logits antigos, mas também pode impedir essa recalibração e aumentar
o `classifier_gap`.

No modo hidden-only, as conexões entre camadas ocultas continuam recebendo a
proteção fatorada. Os pesos da cabeça de classificação não recebem máscara
SlowHeat.

### `beta_30`

`beta` controla a intensidade da proteção suave:

- `beta=0`: nenhuma redução do update;
- beta pequeno: proteção leve e maior aquisição;
- beta grande: proteção forte e aprendizagem mais lenta nas unidades
  consolidadas.

Com `beta=30`, uma unidade com `slow_heat=1` recebe aproximadamente `1/31` do
delta nativo. Os testes de 1, 2, 5, 10 e 20 épocas mostraram que essa proteção
forte precisa de mais passos para adquirir novas tarefas. Dez épocas foi o
melhor compromisso observado.

### `budget_0.25`

O budget é a fração mínima de neurônios que deve continuar livre:

```text
protected_count <= floor((1 - plasticity_budget) * unit_count)
```

Portanto, `budget=0.25` não significa “proteger 25%”. Significa:

- manter pelo menos 25% dos neurônios plásticos;
- permitir que no máximo aproximadamente 75% sejam protegidos;
- selecionar os protegidos pelo ranking de importância funcional.

## 2. Protocolo Split-MNIST

O benchmark principal é class-incremental e possui cinco tarefas:

```text
T1: 0/1
T2: 2/3
T3: 4/5
T4: 6/7
T5: 8/9
```

Características do protocolo:

- um único MLP e uma única cabeça de dez classes;
- nenhum task ID na avaliação class-incremental;
- task ID usado apenas na avaliação diagnóstica task-aware;
- fronteiras de tarefa conhecidas para chamar `consolidate()`;
- inicialização de parâmetros byte a byte idêntica dentro de cada seed;
- splits e sequências de minibatches determinísticos e pareados;
- classes futuras removidas da loss até serem apresentadas;
- MLP com dimensões `784 -> 256 -> 128 -> 10`;
- `batch_size=128`;
- `train_per_class=1000` e `test_per_class=500`;
- AdamW com `lr=1e-3` e `weight_decay=1e-4`;
- política de estado SlowHeat `follow_update`.

## 3. Métricas

### Acurácia média final

Média da acurácia das cinco tarefas depois do último estágio. É a principal
métrica class-incremental. Maior é melhor.

### Average forgetting

Para cada tarefa antiga, mede a diferença entre seu melhor desempenho
histórico e o desempenho final. Menor é melhor.

### Backward transfer

Compara a acurácia final de uma tarefa com sua acurácia imediatamente após ser
aprendida. É frequentemente próximo do negativo do forgetting, mas as duas
métricas podem divergir quando há recuperação ou trajetórias não monotônicas.

### Forward transfer

Compara o desempenho em uma tarefa antes de treiná-la com o baseline aleatório
pareado. Os métodos começam pareados; por isso FWT foi semelhante entre as
variantes e não foi decisivo nesta investigação.

### Task-aware accuracy

Restringe a decisão às duas classes da tarefa avaliada. Ela não é a métrica
principal, pois fornece informação de tarefa, mas ajuda a localizar o problema:

- task-aware alta e class-incremental baixa: a representação ainda discrimina
  as classes, mas a cabeça global está enviesada;
- ambas baixas: houve degradação da própria representação ou aquisição fraca.

### Classifier gap

Definido por:

```text
classifier_gap = task_aware_final_accuracy - final_average_accuracy
```

Menor é melhor, desde que a task-aware accuracy não tenha sido sacrificada.

## 4. Validação de implementação

Antes dos experimentos longos, a suíte chegou a 84 testes automatizados. Entre
os comportamentos cobertos estão:

- invariância da importância funcional à reescala recíproca ReLU;
- importância zero para neurônio ReLU morto;
- orçamento mínimo de capacidade plástica;
- proteção fatorada de linhas e colunas;
- hard-freeze exato de parâmetros consolidados;
- aplicação da máscara ao delta final de SGD e AdamW;
- semântica `follow_update` e `native` para estado do otimizador;
- checkpoint que falha fechado sem registrar as mesmas máscaras;
- inicialização pareada entre métodos;
- replay e distillation em execução sintética pequena;
- matrizes class-incremental e task-aware;
- agregação multi-seed e diferenças pareadas;
- sweep de épocas com artefatos em formato longo.

Esses testes validam a implementação e o protocolo. Eles não demonstram
superioridade empírica.

## 5. Cronologia experimental

### 5.1 Execução inicial: proteção sem replay

Uma execução inicial comparou vanilla e variantes SlowHeat com poucas épocas.
As acurácias finais ficaram próximas de `0.18-0.19`, com forgetting próximo de
`0.97-0.98`.

Conclusão:

- ocorreu esquecimento catastrófico quase completo;
- SlowHeat isolado não apresentou vantagem útil;
- pequenas diferenças de milésimos não constituíam evidência de eficácia.

### 5.2 Comparação ampliada em cinco seeds e cinco épocas

Foram adicionados beta `10/30/100`, hard-freeze, replay, distillation,
SlowHeat+replay, SlowHeat+distillation e avaliação task-aware.

Resultados centrais:

| Método | Acurácia final | Forgetting | Task-aware | Classifier gap |
|---|---:|---:|---:|---:|
| replay | 0.77200 | 0.26500 | 0.98232 | 0.21032 |
| slowheat_replay | 0.69904 | 0.34225 | 0.97956 | 0.28052 |
| distillation | 0.19420 | 0.99040 | 0.85056 | 0.65636 |
| slowheat_beta_30 | 0.18960 | 0.98130 | 0.72564 | 0.53604 |
| hard_freeze | 0.17396 | 0.96365 | 0.67436 | 0.50040 |

Conclusões:

- replay foi o primeiro baseline realmente eficaz;
- SlowHeat protegendo também a saída prejudicou o replay;
- distillation preservou alguma discriminação task-aware, mas não resolveu a
  competição class-incremental;
- proteção extrema ou hard-freeze não compensou ausência de exemplos antigos;
- o grande classifier gap apontou a cabeça global como gargalo central.

### 5.3 Ablação hidden-only, beta e budget

O notebook passou a testar:

- `protect_output=False`, codificado como `hidden`;
- beta `1`, `3`, `10` e `30`;
- budgets `0.25`, `0.50` e `0.75`;
- 1, 5 e 10 épocas inicialmente;
- diferenças pareadas diretamente contra replay.

Com cinco seeds, a maior média em cinco épocas foi obtida por:

```text
slowheat_replay_hidden_beta_3_budget_0.50
```

com acurácia `0.78184` e forgetting `0.25075`. Contra replay, os deltas
pareados foram:

- acurácia: `+0.00984`;
- forgetting: `-0.01425`;
- classifier gap: `-0.01048`.

Entretanto, ao substituir o IC normal pelo IC Student-t adequado para `n=5`,
os intervalos cruzaram zero. A variante era promissora, mas não confirmada.

Na mesma exploração, `hidden_beta_30_budget_0.25` em dez épocas apresentou
uma direção mais consistente. Isso motivou fixar esse candidato.

### 5.4 Execuções de configuração incorreta que foram informativas

Duas execuções não testaram o candidato confirmatório pretendido:

1. `slowheat_beta_30` sem replay e protegendo a saída;
2. `slowheat_replay_hidden_beta_30_budget_0.25` com apenas duas épocas.

Na primeira, em 19 seeds, SlowHeat isolado perdeu aproximadamente `49.6`
pontos percentuais de acurácia para replay e aumentou forgetting em cerca de
`46.6` pontos. Essa execução é uma ablação negativa válida: SlowHeat não
substitui replay.

Na segunda, o arquivo `multi_seed_config.json` confirmou
`epochs_per_task=2`. O candidato perdeu aproximadamente `12.4` pontos de
acurácia. Isso não contradiz o resultado de dez épocas; mostrou que beta `30`
reduz a aquisição quando o orçamento de treino é curto.

Esses casos estabeleceram uma regra de reprodutibilidade: interpretar um
resultado somente depois de conferir o `multi_seed_config.json`.

### 5.5 Sweep de 1, 5, 10 e 20 épocas

O candidato fixado foi comparado a replay em diferentes orçamentos. Na
execução mais recente com 20 seeds:

| Épocas | Método | Acurácia | Forgetting | Task-aware | Gap |
|---:|---|---:|---:|---:|---:|
| 1 | replay | 0.45861 | 0.60656 | 0.96298 | 0.50437 |
| 1 | SlowHeat hidden | 0.27485 | 0.72021 | 0.90166 | 0.62681 |
| 5 | replay | 0.77275 | 0.26568 | 0.98219 | 0.20944 |
| 5 | SlowHeat hidden | 0.76744 | 0.26239 | 0.98214 | 0.21470 |
| 10 | replay | 0.76288 | 0.28274 | 0.98336 | 0.22048 |
| 10 | SlowHeat hidden | **0.77549** | **0.26258** | 0.98354 | **0.20805** |
| 20 | replay | 0.76048 | 0.28631 | 0.98187 | 0.22139 |
| 20 | SlowHeat hidden | 0.76614 | 0.27761 | **0.98461** | 0.21847 |

Conclusões:

- uma época é insuficiente para beta `30`;
- em cinco épocas, replay e SlowHeat hidden têm desempenho semelhante, com
  pequena troca entre acurácia e retenção;
- dez épocas é o melhor ponto observado para o candidato SlowHeat;
- vinte épocas melhora a discriminação task-aware, mas piora a competição
  class-incremental em relação a dez épocas;
- a queda de dez para vinte épocas reforça que o gargalo residual está na
  cabeça global, não apenas na representação.

### 5.6 Agregações intermediárias de 8 e 19 seeds

Antes da execução final de 20 seeds, duas agregações ampliaram a avaliação do
candidato correto.

Na execução pareada de oito seeds e dez épocas, os deltas contra replay foram:

- acurácia: `+1.295` ponto percentual;
- forgetting: `-2.050` pontos;
- classifier gap: `-1.308` pontos;
- task-aware forgetting: `-0.222` ponto.

Com Student-t e sete graus de liberdade, a acurácia teve `p ~= 0.049`, o
forgetting `p ~= 0.024` e o classifier gap ficou limítrofe em `p ~= 0.052`.
Essa execução confirmou a direção, mas ainda tinha baixa potência e forte
sensibilidade a uma seed.

Uma agregação posterior de 19 seeds, ainda sem o JSON pareado correspondente,
produziu em dez épocas:

| Método | Acurácia | Forgetting | Classifier gap |
|---|---:|---:|---:|
| replay | 0.76693 | 0.27772 | 0.21705 |
| SlowHeat hidden + replay | 0.77619 | 0.26103 | 0.20796 |

Como só as marginais foram disponibilizadas naquele momento, essa execução
reforçou consistência, mas não foi usada para um teste pareado formal. A
execução seguinte de 20 seeds incluiu as diferenças pareadas e é o resultado
principal documentado abaixo.

## 6. Resultado pareado principal em 20 seeds

O resultado principal atual usa:

```text
epochs_per_task = 10
replay_per_class = 20
method = slowheat_replay_hidden_beta_30_budget_0.25
reference = replay
```

### Médias

| Métrica | Replay | SlowHeat hidden + replay |
|---|---:|---:|
| Acurácia final | 0.76288 | **0.77549** |
| Average forgetting | 0.28274 | **0.26258** |
| Task-aware accuracy | 0.98336 | 0.98354 |
| Task-aware forgetting | 0.01145 | **0.00828** |
| Classifier gap | 0.22048 | **0.20805** |

### Diferenças pareadas

Os JSONs armazenam IC normal com fator `1.96`. Como `n=20`, este documento
recalcula IC95% com Student-t e 19 graus de liberdade.

| Métrica | Delta SlowHeat - replay | IC95% Student-t | Interpretação |
|---|---:|---:|---|
| Acurácia final | **+1.261 pp** | +0.669 a +1.853 pp | melhora |
| Average forgetting | **-2.016 pp** | -2.781 a -1.252 pp | melhora |
| Classifier gap | **-1.243 pp** | -1.813 a -0.673 pp | melhora |
| Task-aware accuracy | +0.018 pp | inclui zero | equivalente |
| Task-aware forgetting | **-0.318 pp** | -0.448 a -0.187 pp | melhora |

Testes t pareados exploratórios produziram aproximadamente:

- acurácia: `p ~= 0.00027`;
- forgetting: `p < 0.0001`;
- classifier gap: `p ~= 0.0002`;
- task-aware forgetting: `p < 0.0001`.

O padrão é coerente com a hipótese do método: a discriminação task-aware final
é essencialmente igual, enquanto retenção e competição global melhoram.

## 7. Custo computacional e comparação compute-aware

Em dez épocas, na execução pareada:

- replay: aproximadamente `1.32 s`;
- SlowHeat hidden + replay: aproximadamente `3.50 s`;
- multiplicador de tempo: aproximadamente `2.66x`.

Os tempos absolutos servem apenas para comparação dentro da mesma execução e
máquina.

Quando o orçamento computacional é livre, replay com cinco épocas é um baseline
muito forte:

| Configuração | Acurácia | Forgetting | Tempo aproximado |
|---|---:|---:|---:|
| replay, 5 épocas | 0.77275 | 0.26568 | 0.71 s |
| SlowHeat hidden, 10 épocas | 0.77549 | 0.26258 | 3.50 s |

SlowHeat ganha aproximadamente `0.274` ponto percentual de acurácia e `0.310`
ponto de forgetting, mas custa cerca de cinco vezes mais que replay em cinco
épocas. Assim:

- sob o mesmo número de épocas, SlowHeat hidden vence;
- sob comparação de custo-benefício, replay em cinco épocas permanece mais
  eficiente.

## 8. Conclusões sustentadas

Os experimentos sustentam, neste protocolo específico:

1. SlowHeat isolado não é competitivo com replay.
2. Replay é essencial para retenção no Split-MNIST class-incremental usado.
3. Proteger a cabeça de saída prejudica sua recalibração, principalmente em
   orçamentos curtos e médios.
4. Hidden-only resolve a principal vulnerabilidade da combinação original.
5. Beta `30` exige mais passos de aquisição; dez épocas foi o melhor ponto.
6. Em dez épocas e 20 seeds pareadas, o candidato melhora acurácia, forgetting
   e classifier gap contra replay com o mesmo orçamento de treino.
7. A task-aware accuracy equivalente indica que o ganho principal ocorre em
   retenção e competição global, não por aumento amplo da capacidade
   representacional final.
8. O benefício tem custo de tempo relevante.

## 9. Conclusões que ainda não são sustentadas

Os resultados não demonstram que:

- SlowHeat seja SOTA em continual learning;
- SlowHeat supere replay sob orçamento computacional igual;
- os resultados generalizem além de Split-MNIST;
- o método seja robusto a ordens de tarefas diferentes;
- o ganho permaneça com memória de replay maior ou menor;
- a vantagem supere baselines especializados como DER++, ER-ACE, AGEM, EWC e
  SI;
- os p-valores exploratórios permaneçam após seleção de hiperparâmetros em um
  conjunto completamente independente.

As 20 seeds mais recentes incluem seeds usadas durante a exploração. Elas
aumentam a precisão da estimativa, mas não formam um conjunto confirmatório
inteiramente independente. O valor `1010` presente na lista é uma seed válida,
mas deve ser confirmado como intencional para evitar erro de transcrição de
`101`.

## 10. Próximos testes recomendados

### Confirmação independente

- congelar `hidden`, beta `30`, budget `0.25`, dez épocas e memória `20/classe`;
- escolher antes da execução 20 seeds novas, sem sobreposição;
- declarar acurácia final como endpoint primário;
- reportar diferenças pareadas com Student-t, bootstrap e contagem de sinais;
- não reajustar hiperparâmetros nas seeds confirmatórias.

### Fairness e custo

- comparar por mesmo número de épocas;
- comparar por mesmo número total de exemplos processados;
- comparar por tempo e FLOPs aproximados;
- incluir replay com mais épocas e early stopping por validação;
- contabilizar o custo adicional dos hooks, consolidação e máscaras.

### Baselines

- DER++;
- ER-ACE;
- AGEM ou GEM;
- EWC;
- SI;
- LwF/distillation calibrada;
- replay com correção de logits ou loss balanceada.

### Generalização

- múltiplas ordens de classes;
- Permuted-MNIST como diagnóstico diferente;
- Split CIFAR-100;
- TinyImageNet ou outro stream visual mais difícil;
- arquiteturas maiores e diferentes budgets de memória.

### Ablações restantes

- hidden-only sem replay;
- beta `30` com budget adaptativo;
- proteção da saída parcial em vez de binária;
- calibração da cabeça global;
- diferentes tamanhos de replay;
- comparação com redução global de learning rate para separar seletividade de
  simples desaceleração.

## 11. Configuração candidata congelada

Até que um teste independente indique o contrário, o candidato experimental é:

```python
SplitMNISTConfig(
    hidden_dims=(256, 128),
    batch_size=128,
    epochs_per_task=10,
    train_per_class=1_000,
    validation_per_class=200,
    test_per_class=500,
    learning_rate=1e-3,
    weight_decay=1e-4,
    replay_per_class=20,
    replay_batch_size=64,
    optimizer_state_policy="follow_update",
    methods=(
        "replay",
        "slowheat_replay_hidden_beta_30_budget_0.25",
    ),
)
```

Essa configuração é a melhor hipótese experimental atual. Ela não deve ser
alterada usando os resultados das próximas seeds confirmatórias.
