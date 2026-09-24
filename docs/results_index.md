# Índice de resultados versionados

Criado em 22 de setembro de 2026 pela revisão documental. Antes deste índice,
13 diretórios de `results/` não eram citados por documento algum, incluindo a
confirmação congelada e o maior artefato do repositório.

Regra de leitura: **nenhum diretório aqui é confirmatório, exceto os dois
marcados como tal**. Todos os demais são exploratórios e não têm pré-registro.

## Confirmatório

| Diretório | Conteúdo |
|---|---|
| `protocol_post_eval_fix/confirmation/` | 20 seeds, commit `d5b22ad`, 2026-09-03. `preregistration.lock.json` com `status = frozen_before_execution` |
| `protocol_post_eval_fix_d5b22ad/confirmation/` | 20 seeds, commit `f2f7616`, 2026-09-04. Mesmo `sha256` de pré-registro |

Resultado e ressalvas em [`confirmatory_protocol.md`](confirmatory_protocol.md).
As duas execuções concordam em todas as métricas científicas e divergem apenas
em campos de custo. Ambas registram árvore Git suja.

## Split-MNIST exploratório

| Diretório | Conteúdo | Por que existe |
|---|---|---|
| `protocol_post_eval_fix_d5b22ad/` (fora de `confirmation/`) | 9 agregados: baselines equal-epochs (20 seeds x 12 métodos), SlowHeat+DER++ (20 seeds), ablações de método e de memória 5/10/20/50/100 | suíte secundária do mesmo commit |
| `100seeds/` | 100 seeds x 12 métodos | maior *n* do repositório; nenhuma tabela publicada o usa |
| `secondary_post_eval_fix_d5b22ad/` | análises secundárias, incluindo a **única** análise `equal_examples` | o manuscrito §4.2 discute o critério mas não reporta o resultado |
| `protocol_post_eval_fix_d5b22/` | run parcial, só `slowheat_derpp_exploratory` | interrompida; nome quase idêntico ao diretório completo |

Os CSVs soltos na raiz de `results/` (`split_mniist_results.csv`,
`permutated_mnist_download.csv`, `pair_differences.csv`,
`paired_differences.csv`) **não registram seeds, commit nem configuração**. Não
devem ser citados como fonte. Ver o aviso de rastreabilidade na seção 7 de
[`project_methods_and_results.md`](project_methods_and_results.md).

## Visual

| Diretório | Conteúdo |
|---|---|
| `split_mnist_protocol/split_cifar10/`, `split_cifar100/` | 10 seeds x 19 métodos cada |
| `split_mnist_protocol/split_cifar10_{cnn,vgg11,resnet18}*` | sweeps por backbone |
| `dualheat_pairs/split_cifar10/`, `split_cifar100/` | 10 seeds x 8 métodos |
| `dualheat_pairs_portable/`, `_existing/`, `_smoke/` | relatórios derivados, sem dados brutos |

Contrato em [`split_cifar.md`](split_cifar.md).

## Replay seletivo

| Diretório | Conteúdo |
|---|---|
| `cache_derpp_10seeds/replay_selection_sweep/` | **~100 GB**: 5 datasets x 5 caches, 25 agregados de 50 seeds cada. Maior artefato do repositório |
| `replay_selection_full/replay_selection_sweep/` | 10 agregados de 10 seeds |

[`replay_selection.md`](replay_selection.md) referenciava
`results/cache_all_datasets_10seeds/`, que **nunca existiu**; os comandos foram
corrigidos para `cache_derpp_10seeds`.

## BERT / CLINC150

| Diretório | Conteúdo | Cuidado |
|---|---|---|
| `bert_slowheat_review/` | diagnóstico de 10 seeds | **fonte primária** de [`bert_slowheat_diagnostic_results.md`](bert_slowheat_diagnostic_results.md) |
| `bert_slowheat_diagnostic/` | versão de **3 seeds** do mesmo diagnóstico | **colisão de nomes**: o nome casa com o do documento, mas o documento usa o diretório `_review`. Números diferentes (vanilla 46,83% vs 48,47%; hard 71,61% vs 71,38%) |
| `bert_full_coverage/` | 10 métodos x 3 seeds, `evaluate_test=false` | ver [`bert_full_coverage_ablation.md`](bert_full_coverage_ablation.md) |
| `new_bert/` | execução independente do protocolo; tem `ANALISE.md` próprio | tabela não replicada em documento algum |
| `secrets/` | segunda execução independente do mesmo protocolo | **nome opaco e sem README**; renomear |

`new_bert/` e `secrets/` têm a mesma estrutura e todos os hashes distintos: são
execuções independentes, não cópias.

## Qwen2

| Diretório | Conteúdo | Estado |
|---|---|---|
| `qwen_capacity_diagnostic/` | manifesto de capacidade, escopo local, 2 tarefas, seed 0 | o critério `--min-effective-plasticity` **não foi aplicado** (`declared_before_run: false`) |
| `qwen_iso_plasticity/` | calibração (`seed0-2`, `hard_seed0-2`) **mais** a confirmação `confirm120_seed10-19` | as runs de calibração usam 30 passos por tarefa, valor revogado pela tabela K, e estão obsoletas. As 10 seeds `confirm120_*` usam 120 passos, `protocol_hash` idêntico, e são a fonte de [`goals/resultados_confirmacao.md`](../goals/resultados_confirmacao.md) |
| `qwen_layer_anomaly/` | 7 manifestos, 2 ordens x 3 seeds | executado para decidir se a anomalia de L3/L21 é artefato de seed/ordem. **A resposta nunca foi escrita**; `docs/qwen_layer_anomaly.md` não existe. O PR agregado varia 11.567–27.955 entre seeds |
| `run_logs/` | logs de smoke em GPU, sweep de passos, runs de calibração | `results_gpu_memory_smoke.log` mede 5,447 GiB de pico |

Protocolo congelado em `goals/protocol_iso_plasticity.md`. Gate 1, Gate 2 e
Gate 3 **não foram registrados**.

## Hard versus soft (MLP e CNN)

| Diretório | Conteúdo | Estado |
|---|---|---|
| `hard_vs_soft/split_mnist_mlp/` | 10 seeds x 6 métodos | completo |
| `hard_vs_soft/permuted_mnist_mlp/` | 10 seeds x 6 métodos | completo |
| `hard_vs_soft/split_cifar10_mlp/` | 10 seeds x 6 métodos | completo |
| `hard_vs_soft/split_cifar100_mlp/` | 10 seeds x 6 métodos | completo |
| `hard_vs_soft/split_cifar10_cnn/` | 10 seeds x 6 métodos | treino em 23/09; `pair_report` regenerado em 24/09 com `--report-only` (ver §6 dos resultados) |

Único conjunto do repositório com **protocolo congelado antes da execução e
árvore Git limpa** (commit `6f4d12d`) fora da confirmação Split-MNIST. Desenho
em [`hard_vs_soft_protection.md`](hard_vs_soft_protection.md), resultados em
[`hard_vs_soft_results.md`](hard_vs_soft_results.md).

## Pendências deste índice

- decidir entre renomear `results/secrets/` ou dar-lhe um README;
- resolver a colisão `bert_slowheat_diagnostic` vs `bert_slowheat_review`;
- escrever `docs/qwen_layer_anomaly.md` a partir de `results/qwen_layer_anomaly/`;
- publicar ou descartar os agregados de 100 seeds e o sweep de replay.