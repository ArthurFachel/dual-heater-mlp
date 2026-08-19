# Split-CIFAR-10 e Split-CIFAR-100

Os adapters em `experiments/visual_generalization.py` expõem dois benchmarks
visuais de aprendizagem contínua com uma única cabeça compartilhada:

- Split-CIFAR-10: 10 classes, divididas em 5 tarefas de 2 classes;
- Split-CIFAR-100: 100 classes, divididas em 10 tarefas de 10 classes.

Ambos usam o cenário Class-IL. A avaliação não recebe o identificador da tarefa
e considera todas as classes vistas até aquele estágio. As fronteiras das
tarefas continuam disponíveis ao SlowHeat para consolidação.

## Preparação dos dados

O torchvision baixa CIFAR-10 e CIFAR-100 automaticamente para `data/`, salvo
quando `--no-download` é usado. As imagens RGB de 32×32 são normalizadas com as
estatísticas usuais do CIFAR e achatadas para 3.072 entradas, permitindo usar o
mesmo engine MLP e as mesmas comparações pareadas dos demais protocolos.

Essa escolha isola a mudança de stream e de complexidade visual, mas não deve
ser interpretada como evidência de desempenho com CNNs.

## Execução

Visualize o plano sem baixar dados ou treinar:

```bash
python run_all_tests.py --sections split-cifar10 split-cifar100 --dry-run
```

Execute os dois benchmarks:

```bash
python run_all_tests.py --sections split-cifar10 split-cifar100 --device cpu
```

Cada método recebe inicialização, partições, minibatches e memória de replay
pareados dentro de cada seed. Cada seção CIFAR executa os 31 métodos visuais
implementados ou configurados explicitamente pelo projeto:

- vanilla e os controles/variantes diretas do SlowHeat;
- `beta = 10, 30, 100`, hard freeze e controles de budget/estado;
- Replay, distillation, DER++ e suas combinações SlowHeat;
- ER-ACE, A-GEM, EWC, SI e LwF calibrado;
- replay balanceado, mais épocas, early stopping e learning rate reduzido;
- ablações hidden-only, budget adaptativo, saída parcial e calibração.

A ordem e os nomes exatos estão em `ALL_VISUAL_METHODS`, em
`experiments/split_mnist_suite.py`, e aparecem no plano produzido por
`--dry-run`. Com 31 métodos e dez seeds por dataset, a execução completa é
substancialmente mais cara que o protocolo anterior de quatro métodos.
