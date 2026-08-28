# Replay com seleção de imagens

O benchmark oferece quatro políticas de memória episódica:

- `first`: controle histórico; mantém os primeiros exemplos materializados de cada classe;
- `loss`: mantém os exemplos com maior cross-entropy individual;
- `representative`: usa herding sobre embeddings normalizados da penúltima camada;
- `hybrid`: combina ranks de loss (`0,50`), entropia (`0,30`) e cobertura (`0,20`).

Os rankings usam exclusivamente o split de treino e são calculados pelo próprio
learner ao final de cada tarefa. Por isso, dois learners podem escolher memórias
diferentes. Comparações entre eles medem os algoritmos completos e não isolam o
efeito de SlowHeat mantendo o replay fixo.

## Com e sem memória

O uso de memória é definido pelo método, sem alterar a arquitetura básica:

```python
from dataclasses import replace

from experiments.split_mnist_suite import baseline_config

without_memory = replace(
    baseline_config(),
    methods=("vanilla", "slowheat_hidden_beta_30_budget_0.25"),
)

with_ranked_memory = replace(
    baseline_config(),
    methods=("replay", "slowheat_replay_hidden_beta_30_budget_0.25"),
    replay_selection="hybrid",
)
```

Métodos sem replay não criam o buffer, não executam o seletor e reportam zero
bytes e zero exemplos de ranking. `replay_per_class` controla quantas imagens
são retidas por classe em cada experiência.

## Sweep

O sweep opt-in compara os quatro seletores e inclui as ablações sem memória:

```bash
python run_all_tests.py \
  --num-seeds 10 \
  --sections replay-selection-sweep \
  --device cpu
```

Com os cinco streams e dez seeds, são 500 execuções de learner: as duas
referências sem memória rodam uma vez por dataset/seed, enquanto Replay e
SlowHeat+Replay rodam uma vez para cada um dos quatro seletores.

Use `--dry-run` para inspecionar a matriz, `--no-download` quando os datasets já
estiverem disponíveis e `--fresh` para exigir uma nova árvore de resultados.

## Consultar acurácia e forgetting

O utilitário `show_cache_results.py` mostra apenas Replay e SlowHeat+Replay. A
saída usa porcentagens e IC95% normal sobre as seeds concluídas.

Todos os benchmarks e caches encontrados:

```bash
python show_cache_results.py \
  results/cache_all_datasets_10seeds/replay_selection_sweep
```

Um cache em todos os benchmarks:

```bash
python show_cache_results.py \
  results/cache_all_datasets_10seeds/replay_selection_sweep \
  --cache hybrid
```

Um benchmark e cache específicos:

```bash
python show_cache_results.py \
  results/cache_all_datasets_10seeds/replay_selection_sweep \
  --benchmark split_cifar10_cnn \
  --cache loss
```

Para mostrar somente o cache com maior acurácia em cada benchmark e learner:

```bash
python show_cache_results.py \
  results/cache_all_datasets_10seeds/replay_selection_sweep \
  --accuracy --high
```

Também estão disponíveis `--accuracy --low`, `--forget --high` e
`--forget --low`. A métrica e a direção devem ser informadas juntas. A seleção
compara as médias entre os caches disponíveis para cada combinação de benchmark
e learner; por isso, datasets e learners diferentes não são comparados entre si.
O IC95% continua aparecendo na linha selecionada, mas não participa do ranking.

Também é possível passar diretamente a pasta de um benchmark ou cache. Se a
execução ainda estiver incompleta, o utilitário agrega os `seed_*/results.json`
já disponíveis e mostra um aviso no terminal.

## Checkpoints e dados sensíveis

Replay e SlowHeat+Replay salvam um checkpoint móvel ao final de cada tarefa. Uma
retomada reinicia apenas a tarefa que estava incompleta. O checkpoint contém os
tensores das imagens selecionadas, rótulos, scores e estado do modelo/otimizador;
portanto, deve receber a mesma proteção aplicada ao dataset de treinamento.
