# Protocolo Split-CIFAR-10 e Split-CIFAR-100

Status: implementado e coberto por testes. Há uma triagem CNN local com dez
seeds em `results/paired_differences.csv`; o arquivo contém diferenças
pareadas, mas não as matrizes de acurácia e o manifesto de ambiente completos.
Há também quatro seeds parciais do protocolo MLP de Split-CIFAR-10 versionadas
como diagnóstico de uma execução interrompida. Não há resultado confirmatório
CNN, agregado MLP completo nem resultado Split-CIFAR-100.

Os adapters em `experiments/visual_generalization.py` expõem os benchmarks
visuais com uma única cabeça compartilhada:

- Split-CIFAR-10: 10 classes em 5 tarefas de 2 classes;
- Split-CIFAR-100: 100 classes em 10 tarefas de 10 classes.

Há duas seções opt-in que mantêm o mesmo stream Split-CIFAR-10 e preservam as
imagens como tensores NCHW: `split-cifar10-cnn` usa uma CNN `3→32→64`, com
pooling adaptativo `2×2`; `split-cifar10-resnet18` usa uma ResNet-18 com stem
CIFAR `3×3`, blocos `2–2–2–2`, canais `64–128–256–512`, GroupNorm e pooling
global. Ambas têm cabeça compartilhada de 10 classes.

Ambos usam Class-IL. A avaliação principal não recebe o identificador da
tarefa e considera todas as classes vistas até o estágio. Classes futuras são
mascaradas no treino e na avaliação. SlowHeat continua recebendo as fronteiras
de tarefa para chamar `consolidate()`; portanto, o método é boundary-aware.
A avaliação task-aware é registrada apenas como diagnóstico.

## Dados e partições

O torchvision baixa CIFAR-10 e CIFAR-100 para `data/` por padrão. Use
`--no-download` somente quando os datasets já estiverem disponíveis. As
imagens RGB de 32×32 são convertidas para `[0,1]`, normalizadas canal a canal
com média `(0.4914, 0.4822, 0.4465)` e desvio-padrão
`(0.2470, 0.2435, 0.2616)`, e achatadas para 3.072 entradas. O código usa as
mesmas constantes nos dois datasets.

Dentro de cada seed, os índices de cada classe são embaralhados de forma
determinística. A validação é retirada primeiro do conjunto de treino; a
amostra de treino vem do restante. O conjunto de teste é selecionado
separadamente. Não há sobreposição entre treino e validação.

| Configuração | Split-CIFAR-10 | Split-CIFAR-100 |
|---|---:|---:|
| Ordem de classes | `0..9` | `0..99` |
| Tarefas × classes | `5 × 2` | `10 × 10` |
| Treino por classe | 4.000 | 400 |
| Validação por classe | 500 | 50 |
| Teste por classe | 1.000 | 100 |
| MLP | `3072 → 1024 → 512 → 10` | `3072 → 1024 → 512 → 100` |
| Épocas por tarefa | 10 | 10 |
| Learning rate / weight decay | `1e-3` / `1e-4` | `1e-3` / `1e-4` |
| Replay | 20 por classe; lote 64 | 20 por classe; lote 64 |
| SlowHeat padrão | `beta=30`, budget `0.25` | `beta=30`, budget `0.25` |

As imagens achatadas permitem reutilizar o engine MLP pareado. Esses
experimentos testam streams visuais mais difíceis, mas não medem desempenho de
SlowHeat em CNNs e não devem ser apresentados como tal.

### Benchmark CNN

O benchmark CNN usa 5 épocas por tarefa e os controles `vanilla`,
`slowheat_none`, `slowheat_unidirectional`, `slowheat` e `hard_freeze`. Ele
também executa os pares:

- `lpr` e `slowheat_lpr`;
- `classifier_expander` e `slowheat_classifier_expander`;
- `scroll` e `slowheat_scroll`.

A inicialização dos parâmetros treináveis é idêntica dentro de cada par e de
cada seed. Os contrastes multi-seed são calculados contra `vanilla` e contra
cada método normal, permitindo ler diretamente `método+SlowHeat - método`.

LPR calcula covariâncias não centradas das ativações da memória e aplica o
precondicionador proximal camada a camada antes do passo do otimizador. O
Classifier Expander usa classificação inner-task no fluxo atual, replay com
distilação do modelo anterior e uma segunda fase que treina somente a cabeça
na memória balanceada. SCROLL acumula estatísticas suficientes, resolve a
cabeça por ridge regression e adapta a representação apenas com replay.

O SCROLL original pressupõe uma representação pré-treinada. Como este projeto
não distribui um checkpoint externo, task 0 é usado explicitamente como
bootstrap pareado da representação. Portanto, `scroll` neste runner é uma
adaptação autocontida do protocolo, não uma reprodução numérica do resultado
pré-treinado do artigo.

### Benchmark ResNet-18

A ResNet-18 possui versões nativa e SlowHeat com os mesmos nomes e shapes de
parâmetros treináveis. A versão SlowHeat registra explicitamente:

- a branch principal de cada BasicBlock;
- a projeção `1×1` quando dimensões ou stride mudam;
- o fan-out do tensor de entrada para a branch principal e o downsample;
- um rastreador de importância sem parâmetros após cada soma residual;
- os parâmetros affine de GroupNorm associados ao canal produtor.

O controle `slowheat_none` deve reproduzir o update nativo antes de qualquer
consolidação. A seção executa os mesmos cinco controles e os pares LPR,
Classifier Expander e SCROLL do benchmark CNN pequeno. Ainda não há resultado
multi-seed da ResNet; implementado significa somente que o grafo, o runner e os
smoke tests estão disponíveis. Como a inversão de covariância do LPR cresce
cubicamente com a dimensão dos patches, sua frequência é reduzida de 30 para
300 passos nesta arquitetura; `lpr` e `slowheat_lpr` usam exatamente o mesmo
valor.

### Triagem CNN exploratória com dez seeds

O export local `results/paired_differences.csv`, analisado em 25 de agosto de
2026, contém dez seeds pareadas. A direção abaixo é sempre
`método+SlowHeat - método`; portanto, diferenças negativas de forgetting e
classifier gap são favoráveis. Valores de desempenho estão em pontos
percentuais (p.p.).

| Contraste pareado | Acurácia final | Forgetting | BWT | Acurácia task-aware | Classifier gap | Sinais da acurácia |
|---|---:|---:|---:|---:|---:|---:|
| SlowHeat+LPR − LPR | +0,78 p.p. | −4,66 p.p. | +4,66 p.p. | −1,29 p.p. | −2,07 p.p. | 8/10 positivos |
| SlowHeat+Classifier Expander − Classifier Expander | −0,88 p.p. | −5,13 p.p. | +10,28 p.p. | −2,05 p.p. | −1,17 p.p. | 2/10 positivos |
| SlowHeat+SCROLL − SCROLL | +5,78 p.p. | −1,25 p.p. | −4,07 p.p. | +3,83 p.p. | −1,95 p.p. | 10/10 positivos |

O padrão sugere três regimes distintos. SlowHeat+LPR foi o par mais
equilibrado: ganhou acurácia média e reduziu forgetting nas dez seeds.
SlowHeat+Classifier Expander reduziu forgetting, mas não melhorou a acurácia
final média. SlowHeat+SCROLL obteve o maior ganho de acurácia e de acurácia
task-aware; seus efeitos sobre forgetting e BWT permanecem inconclusivos.
Comparações desses métodos contra `vanilla` não isolam o efeito do SlowHeat e
não substituem os contrastes pareados acima.

Com `n=10`, o IC 95% t pareado aproximado para a mudança de acurácia fica acima
de zero em LPR (+0,19 a +1,38 p.p.) e SCROLL (+4,14 a +7,42 p.p.). Para
Classifier Expander ele cruza zero por margem pequena (−1,80 a +0,03 p.p.).
Esses intervalos não corrigem a multiplicidade da exploração e ainda não são
uma confirmação independente. O arquivo também não basta para auditar
trajetórias por tarefa, configuração de ambiente ou escores absolutos por seed.
Os três IDs presentes no export anterior foram reexecutados e seus valores
mudaram; o agregado de dez seeds deve ser tratado como uma nova execução, não
como simples extensão do arquivo anterior.

O custo adicional médio observado foi 3,72 s para LPR, 3,85 s para Classifier
Expander e 0,71 s para SCROLL. Os pares preservaram o mesmo número de exemplos
processados e o mesmo uso de memória de replay; o overhead estimado foi de
138.925.280 FLOPs para LPR e Classifier Expander e 21.594.080 FLOPs para
SCROLL.

O próximo passo arquitetural é repetir a comparação com a ResNet-18 e arquivar
os artefatos completos antes de congelar qualquer contraste confirmatório:

```bash
PYTHONPATH=src:. python3 run_all_tests.py \
  --num-seeds 10 \
  --sections split-cifar10-resnet18 \
  --device cuda
```

## Métodos dos protocolos MLP gerais

As seções `split-cifar10` e `split-cifar100` executam os 32 itens de
`ALL_VISUAL_METHODS`, na ordem abaixo. A seção CNN usa apenas os cinco controles
e os três pares listados em “Benchmark CNN”.

1. `vanilla`
2. `slowheat`
3. `slowheat_beta_10`
4. `slowheat_beta_30`
5. `slowheat_beta_100`
6. `slowheat_adaptive`
7. `slowheat_native_state`
8. `slowheat_unidirectional`
9. `slowheat_unbudgeted`
10. `slowheat_none`
11. `hard_freeze`
12. `replay`
13. `distillation`
14. `slowheat_replay`
15. `slowheat_distillation`
16. `derpp`
17. `slowheat_derpp_hidden_beta_30_budget_0.25`
18. `er_ace`
19. `slowheat_er_ace_hidden_beta_30_budget_0.25`
20. `agem`
21. `ewc`
22. `si`
23. `lwf_calibrated`
24. `replay_balanced`
25. `replay_more_epochs`
26. `replay_early_stopping`
27. `replay_global_lr_reduction`
28. `slowheat_replay_hidden_beta_30_budget_0.25`
29. `slowheat_hidden_beta_30_budget_0.25`
30. `slowheat_replay_hidden_adaptive_beta_30_budget_0.25`
31. `slowheat_replay_partial_output_beta_30_budget_0.25`
32. `slowheat_replay_hidden_beta_30_budget_0.25_calibrated`

Os métodos compartilham, dentro de cada seed, inicialização treinável,
partições, agenda dos exemplos atuais e índices de memória determinísticos.
Os contrastes pareados são calculados contra Replay e DER++ quando aplicável.
“Implementado” aqui significa que o método passa pelo mesmo engine e pelos
testes de integração; não implica reprodução exata dos artigos originais nem
ajuste específico para CIFAR.

## Execução

Instale primeiro as dependências de pesquisa:

```bash
python -m pip install -e '.[research]'
```

Visualize métodos, seeds e diretórios sem baixar dados ou treinar:

```bash
python run_all_tests.py \
  --num-seeds 10 \
  --sections split-cifar10 split-cifar100 \
  --dry-run
```

Execute os dois benchmarks:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python run_all_tests.py \
  --num-seeds 10 \
  --sections split-cifar10 split-cifar100 \
  --device cpu
```

Execute o benchmark CNN pequeno, preferencialmente em GPU:

```bash
PYTHONPATH=src:. python run_all_tests.py \
  --num-seeds 3 \
  --sections split-cifar10-cnn \
  --device cuda
```

Para validar apenas o plano:

```bash
PYTHONPATH=src:. python run_all_tests.py \
  --num-seeds 3 \
  --sections split-cifar10-cnn \
  --device cuda \
  --dry-run
```

### ResNet-18 com GroupNorm

Execute somente o benchmark residual com dez seeds pareadas:

```bash
PYTHONPATH=src:. python3 run_all_tests.py \
  --num-seeds 10 \
  --sections split-cifar10-resnet18 \
  --device cuda
```

Acrescente `--dry-run` para verificar métodos, seeds, arquitetura e diretório
sem carregar o CIFAR-10. A execução usa os mesmos cinco controles e os três
pares normal/+SlowHeat da CNN pequena. Ela é opt-in e não altera o protocolo
histórico nem o agregado anterior.

### Sweep CNN de estabilidade/plasticidade

O sweep seguinte preserva `vanilla` e `slowheat_none` como controles e cruza:

- proteção `unidirectional` e `hidden-only`;
- `beta ∈ {3, 10}`;
- budget plástico `{0.50, 0.75}`.

Execute as dez configurações com dez seeds pareadas:

```bash
PYTHONPATH=src:. python run_all_tests.py \
  --num-seeds 10 \
  --sections split-cifar10-cnn-sweep \
  --device cuda
```

Inspecione previamente os métodos e seeds com o mesmo comando acrescido de
`--dry-run`. Os resultados ficam em
`results/split_mnist_protocol/split_cifar10_cnn_sweep/`, isolados do piloto
original.

O argumento obrigatório `--num-seeds` gera essa quantidade de seeds
pseudoaleatórias distintas e reproduzíveis. Para definir valores específicos,
use `--baseline-seeds` e passe exatamente a quantidade declarada em
`--num-seeds`. Resultados são gravados em
`results/split_mnist_protocol/split_cifar10/`,
`results/split_mnist_protocol/split_cifar10_cnn/`,
`results/split_mnist_protocol/split_cifar10_resnet18/`,
ou `results/split_mnist_protocol/split_cifar100/`, conforme a seção, salvo uso de
`--output-dir`.
Execuções retomam seeds concluídas cuja configuração coincide. `--fresh`
desativa a retomada e exige diretórios de saída novos.

Para incluir também Split-MNIST, Permuted-MNIST e o sintético em uma única
chamada:

```bash
python run_all_tests.py --num-seeds 10 --all-datasets-all-methods --device cpu --no-download
```

Com 32 métodos, dez seeds e até dez tarefas por dataset, a suíte completa é
computacionalmente cara. O `--dry-run` valida somente o plano; ele não carrega
os datasets nem estima duração ou memória. Tempos de parede só devem ser
comparados dentro da mesma execução e máquina.
