# Protocolo Split-CIFAR-10 e Split-CIFAR-100

Status: implementado e coberto por testes; nenhum resultado CIFAR está
versionado no repositório.

Os adapters em `experiments/visual_generalization.py` expõem dois benchmarks
visuais com uma única cabeça compartilhada:

- Split-CIFAR-10: 10 classes em 5 tarefas de 2 classes;
- Split-CIFAR-100: 100 classes em 10 tarefas de 10 classes.

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

## Métodos

Cada seção executa os 31 itens de `ALL_VISUAL_METHODS`, na ordem abaixo:

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
19. `agem`
20. `ewc`
21. `si`
22. `lwf_calibrated`
23. `replay_balanced`
24. `replay_more_epochs`
25. `replay_early_stopping`
26. `replay_global_lr_reduction`
27. `slowheat_replay_hidden_beta_30_budget_0.25`
28. `slowheat_hidden_beta_30_budget_0.25`
29. `slowheat_replay_hidden_adaptive_beta_30_budget_0.25`
30. `slowheat_replay_partial_output_beta_30_budget_0.25`
31. `slowheat_replay_hidden_beta_30_budget_0.25_calibrated`

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
  --sections split-cifar10 split-cifar100 \
  --dry-run
```

Execute os dois benchmarks:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python run_all_tests.py \
  --sections split-cifar10 split-cifar100 \
  --device cpu
```

As seeds secundárias padrão são `311, 617, 919, 1223, 1523, 1823, 2129,
2423, 2729, 3037`. Substitua-as com `--baseline-seeds`. Resultados são gravados
em `results/split_mnist_protocol/split_cifar10/` e
`results/split_mnist_protocol/split_cifar100/`, salvo uso de `--output-dir`.
Execuções retomam seeds concluídas cuja configuração coincide. `--fresh`
desativa a retomada e exige diretórios de saída novos.

Com 31 métodos, dez seeds e até dez tarefas por dataset, a suíte completa é
computacionalmente cara. O `--dry-run` valida somente o plano; ele não carrega
os datasets nem estima duração ou memória. Tempos de parede só devem ser
comparados dentro da mesma execução e máquina.
