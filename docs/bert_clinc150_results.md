# BERT/CLINC150: protocolo, métodos e resultados históricos

Estado da auditoria: 14 de setembro de 2026.

Este documento registra todas as onze saídas BERT/CLINC150 encontradas nos três
diretórios históricos principais. Os números foram lidos de cada
`results.json` e conferidos contra suas matrizes 10 x 10. Eles são resultados
reais, porém exploratórios: cada método possui uma única seed, o teste foi
consultado após cada estágio e todas as execuções registram uma árvore Git
`dirty`.

## 1. Protocolo comum

| Item | Valor |
|---|---|
| Modelo | `google/bert_uncased_L-4_H-256_A-4` |
| Dataset | `clinc/clinc_oos`, configuração `plus` |
| Cenário | Class-IL, cabeça global de 150 intenções |
| Tarefas | 10 domínios, 15 intenções por domínio |
| OOS | excluído |
| Ordem | banking, credit_cards, kitchen_and_dining, home, auto_and_commute, travel, utility, work, small_talk, meta |
| Comprimento máximo | 128 tokens |
| Batch | 2 |
| Otimizador | AdamW, LR `5e-5`, weight decay `0.01` |
| Scheduler | linear global, warmup `0.1`, sem reinício por tarefa |
| SlowHeat | força `3.0`, budgets FFN/atenção `0.25`, decay `0.99`, combinação `max` |
| FastHeat | decay `0.9`, força `0.5`, limiar `0.5` |
| FastHeat top-k | fração `0.25`, somente em `dualheat_global_topk` |
| Dados | SHA-256 `d3a9e59d27027f7edd8544b1d525caef1bc811160f691e156c548bb09cd0f381` |

A seed efetiva aparece no `protocol.json` de cada `seed_<n>`. O `seed=42` do
`multi_seed_config.json` é apenas o valor da configuração-base.

As revisões do modelo, tokenizer e dataset estão `null` nos protocolos. Por
isso, embora os artefatos tenham sido executados com o mesmo conteúdo de dados,
a reprodução futura não deve presumir que os nomes remotos ainda resolvem para
o mesmo snapshot.

## 2. Métodos usados nas runs históricas

- `vanilla`: fine-tuning sequencial de todos os 11.209.110 parâmetros;
- `slowheat`: SlowHeat em FFN e atenção, sem congelar parâmetros não vinculados;
- `slowheat_bound`: SlowHeat com parâmetros não cobertos por máscaras
  congelados, restando 3.191.446 treináveis;
- `dualheat`: protocolo bound com FastHeat pós-GELU;
- `slowheat_global`: budget global separado por família FFN/atenção;
- `slowheat_hierarchical`: mínimos plásticos locais e redistribuição global;
- `dualheat_global_topk`: escopo global com FastHeat top-k preparado pelo host
  BERT.

`slowheat` e `dualheat` não formam uma ablação de custo pareado: o primeiro
possui 11.209.110 parâmetros treináveis e o segundo 3.191.446. A comparação
estrutural adequada é `slowheat_bound` contra `dualheat` ou
`slowheat_global` contra `dualheat_global_topk`.

## 3. Métricas

- FAA: acurácia média final Class-IL;
- Fgt: forgetting médio das tarefas antigas;
- TA: acurácia média final task-aware/oráculo;
- F1: macro-F1 final sobre as 150 classes;
- BWT: igual a `-Fgt` nestas runs;
- FWT: não medido e armazenado como `null`.

Todos os valores abaixo são proporções, não porcentagens.

## 4. Resultados de duas épocas, seed 0

Diretório: `results/bert_vanilla_slowheat_dualheat_v2/`.

| Método | Treináveis | FAA | Fgt | TA | F1 final |
|---|---:|---:|---:|---:|---:|
| `vanilla` | 11.209.110 | 0,128444 | 0,759012 | 0,371556 | 0,134137 |
| `slowheat` | 11.209.110 | 0,134889 | 0,707901 | 0,482667 | 0,142930 |
| `dualheat` | 3.191.446 | 0,110889 | 0,715062 | 0,356889 | 0,116258 |

A maior FAA deste grupo foi `slowheat`, mas `n=1` e o número de parâmetros
muda no contraste com `dualheat`.

Fontes:

- `results/bert_vanilla_slowheat_dualheat_v2/seed_0/vanilla/results.json`;
- `results/bert_vanilla_slowheat_dualheat_v2/seed_0/slowheat/results.json`;
- `results/bert_vanilla_slowheat_dualheat_v2/seed_0/dualheat/results.json`;
- `results/bert_vanilla_slowheat_dualheat_v2/seed_0/protocol.json`;
- `results/bert_vanilla_slowheat_dualheat_v2/environment.json`.

## 5. Variantes de Heat de duas épocas, seed 0

Diretório: `results/bert_heat_variants/`. Os quatro métodos têm o mesmo envelope
de 3.191.446 parâmetros treináveis.

| Método | FAA | Fgt | TA | F1 final |
|---|---:|---:|---:|---:|
| `slowheat_bound` | 0,103556 | 0,717037 | 0,309333 | 0,101709 |
| `slowheat_global` | 0,112667 | 0,709383 | 0,330667 | 0,113035 |
| `slowheat_hierarchical` | 0,106222 | 0,709877 | 0,323333 | 0,106089 |
| `dualheat_global_topk` | 0,090444 | 0,718765 | 0,310667 | 0,093645 |

Neste único pareamento, `slowheat_global` teve a maior FAA. Adicionar FastHeat
top-k ao envelope global produziu:

```text
Delta FAA = dualheat_global_topk - slowheat_global = -0,022222
Delta Fgt = dualheat_global_topk - slowheat_global = +0,009383
```

O `aggregate.json` histórico não inclui esse contraste, portanto os deltas acima
são diferenças diretas dos dois `results.json`, não um resultado inferencial.

Fontes: `results/bert_heat_variants/seed_0/<método>/results.json`,
`seed_0/protocol.json`, `multi_seed_config.json` e `environment.json`.

## 6. Resultados de dez épocas, seed 67

Diretório: `results/bert_10epoch/`.

| Método | Treináveis | FAA | Fgt | TA | F1 final | Classificação do artefato |
|---|---:|---:|---:|---:|---:|---|
| `vanilla` | 11.209.110 | 0,136889 | 0,875062 | 0,345556 | 0,142922 | histórico co-localizado |
| `slowheat` | 11.209.110 | 0,141111 | 0,863457 | 0,428889 | 0,144886 | histórico co-localizado |
| `dualheat` | 3.191.446 | 0,128222 | 0,843704 | 0,328667 | 0,134736 | histórico co-localizado |
| `slowheat_bound` | 3.191.446 | 0,136000 | 0,848395 | 0,306444 | 0,143531 | sessão atual do agregado |

Esse diretório mistura duas sessões. Vanilla, SlowHeat e DualHeat pertencem a
uma sessão anterior preservada em seus resultados/checkpoints. Os arquivos
`environment.json`, `multi_seed_config.json`, `protocol.json` e `aggregate.json`
foram depois sobrescritos por uma execução somente de `slowheat_bound`. Assim,
os quatro números são reais, mas o agregado atual não descreve os três primeiros.

Não se deve concluir que dez épocas são melhores ou piores que duas: as seeds
são diferentes e o scheduler global altera fortemente o LR recebido por tarefas
tardias.

## 7. Limitações e validade

1. Cada método possui apenas uma seed. IC95% de largura zero em agregados com
   `n=1` não significa incerteza zero.
2. `evaluate_test=true`; o teste foi consultado depois de cada estágio. As runs
   servem para diagnóstico e desenvolvimento, não confirmação.
3. Todas registram commit-base
   `f7047f8791ae3de6b412b1449641b749a9c76663` e `dirty=true`. O diff executado
   não foi fingerprintado, então não há reprodução bit a bit pela proveniência.
4. O scheduler é único para as dez tarefas. A queda do LR confunde posição da
   tarefa, aquisição e retenção.
5. O diretório `bert_10epoch` mistura sessões e metadados sobrescritos.
6. Os IDs remotos de modelo/dataset não foram fixados por revisão no protocolo.
7. A acurácia task-aware é muito maior que a Class-IL em todos os grupos. Parte
   importante do problema é competição/calibração entre as 150 classes globais,
   não apenas discriminação dentro de cada domínio.
8. As primeiras tarefas chegam a acurácia final zero em várias configurações.
   O aumento de épocas melhorou aquisição inicial, mas não resolveu retenção.
9. O runner atual ainda usa `evaluate_test=True` por padrão e não oferece flag
   CLI para desativá-lo na execução comum.
10. O resume persiste modelo, optimizer, scheduler e replay, mas a auditoria do
    código identificou ausência de RNG CPU/CUDA no checkpoint histórico.

## 8. O que pode ser afirmado

Pode-se afirmar que:

- as onze matrizes foram concluídas e são aritmeticamente consistentes;
- `slowheat_global` liderou o único grupo de variantes com envelope treinável
  pareado;
- FastHeat top-k não trouxe ganho nesse diagnóstico específico;
- mais épocas não evitaram forgetting alto;
- existe um grande gap entre avaliação task-aware e Class-IL.

Não se pode afirmar que:

- SlowHeat ou DualHeat é superior em geral;
- alguma diferença é estatisticamente confirmada;
- o resultado escala para BERT-base ou LLMs;
- os resultados são exatamente reproduzíveis a partir dos manifests atuais.

## 9. Protocolo recomendado para a próxima execução

- fixar revisões de modelo, tokenizer e dataset;
- usar seeds de calibração e confirmação disjuntas;
- manter `evaluate_test=false` durante calibração e desenvolvimento;
- congelar manifesto antes da avaliação final;
- usar múltiplas seeds e, idealmente, múltiplas ordens de tarefas;
- adicionar `dualheat_global_topk - slowheat_global` aos contrastes agregados;
- salvar fingerprint do código e rejeitar resume incompatível;
- salvar/restaurar RNG de CPU/CUDA;
- usar diretório novo por sessão para não sobrescrever metadados;
- considerar scheduler reiniciado por tarefa ou documentar a posição como fator
  experimental.

## 10. Implementação relacionada

- `src/dual_heater/bert.py`;
- `src/dual_heater/transformer.py`;
- `src/dual_heater/fast_heat.py`;
- `src/dual_heater/optim.py`;
- `experiments/split_clinc150.py`;
- [functional_slowheat_transformers.md](functional_slowheat_transformers.md);
- [live_dashboard.md](live_dashboard.md);
- [methods_catalog.md](methods_catalog.md).
