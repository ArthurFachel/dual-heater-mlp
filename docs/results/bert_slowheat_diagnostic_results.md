# BERT/CLINC150: diagnóstico mecanístico, replay, decisão e runs históricas

Estado da análise: 19 de setembro de 2026. Runs históricas auditadas em
14 de setembro de 2026.

Documento único de BERT/CLINC150. A **Parte I** traz os três estudos
exploratórios de duas tarefas que fundamentam a decisão do projeto. A
**Parte II** registra as onze saídas históricas de dez tarefas, com seed única,
preservadas como registro. As duas partes usam protocolos diferentes e **não
podem ser agregadas como uma amostra**.

---

# Parte I — Diagnóstico mecanístico, replay e decisão

## Escopo

Esta parte sintetiza três estudos exploratórios, somente em validação, sobre
os dois primeiros domínios oficiais do CLINC150 (`banking -> credit_cards`). Todos
usam uma cabeça Class-IL global, recebem a fronteira entre tarefas e avaliam
acurácia média final, retenção da primeira tarefa, aquisição da segunda tarefa,
drift de parâmetros e custo de execução.

Os estudos não formam uma única amostra. O diagnóstico mecanístico usa dez seeds;
os dois estudos de replay usam três seeds distintas. Os contrastes são
interpretados somente dentro de cada estudo pareado.

## Integridade e proveniência

| Estudo | Runs | Seeds | Data SHA-256 | Source SHA-256 |
|---|---:|---|---|---|
| Mecanismo | 60/60 | 2, 9, 10, 28, 30, 32, 57, 67, 2005, 2012 | `5e1024...3413` | `4bb938...60f9` |
| Replay, 20 exemplos/classe | 12/12 | 11, 22, 33 | `5e1024...3413` | `ef17c0...2d32` |
| Replay, orçamentos 1/5/10 | 18/18 | 11, 22, 33 | `5e1024...3413` | `4bb938...60f9` |

Todos os 90 `results.json` são finitos. Dentro de cada estudo, ordem de tarefas,
hash dos dados e hash do código são uniformes. Os três ambientes registram o
commit-base `5e96093cf20a3de1e77b5ea78c8650c25efb38c8` com árvore Git suja. O
estudo de replay com memória 20 usa outro fingerprint de fonte; por isso seus
valores não devem ser reagrupados com os outros estudos como uma amostra única.

## 1. SlowHeat contra o Transformer padrão

O baseline `vanilla` é o mesmo BERT treinado sequencialmente com AdamW, sem
SlowHeat e sem replay. No diagnóstico de dez seeds, todas as variantes SlowHeat
aumentaram a acurácia média final e reduziram o forgetting em relação a esse
Transformer padrão.

| Condição | Acurácia média final | Delta contra vanilla | Forgetting de T1 | Sinais do delta de acurácia |
|---|---:|---:|---:|---|
| BERT vanilla | 48,47 ± 2,41% | — | 90,90 ± 4,69 pp | — |
| SlowHeat beta 3 | 51,75 ± 2,58% | +3,28 pp | 83,63 ± 4,56 pp | 9 positivos / 1 negativo |
| SlowHeat beta 10 | 59,65 ± 3,21% | +11,18 pp | 67,20 ± 6,54 pp | 10 positivos / 0 negativos |
| SlowHeat beta 30 | 66,05 ± 2,95% | +17,58 pp | 53,47 ± 6,59 pp | 10 positivos / 0 negativos |
| SlowHeat hard aprendido | 71,38 ± 2,81% | +22,92 pp | 41,03 ± 6,33 pp | 10 positivos / 0 negativos |
| SlowHeat hard aleatório | 67,73 ± 3,21% | +19,27 pp | 50,40 ± 6,07 pp | 10 positivos / 0 negativos |

Portanto, a implementação do SlowHeat em BERT foi funcional e trouxe benefício
consistente sobre o fine-tuning sequencial padrão neste protocolo de duas
tarefas. Essa conclusão não implica superioridade sobre replay nem eficácia em
uma sequência completa de dez tarefas.

## 2. Diagnóstico mecanístico em dez seeds

| Contraste/condição | Acurácia média final | Retenção T1 | Aquisição T2 | Leitura |
|---|---:|---:|---:|---|
| Hard aprendido | 71,38 ± 2,81% | 55,27 ± 6,46% | 87,50 ± 2,06% | esquece 41,03 ± 6,33 pp de T1 |
| Hard aprendido - hard aleatório | +3,65 pp | +9,37 pp | -2,07 pp | FAA positiva em 7/10; retenção positiva em 10/10 |
| Hard aprendido - beta 3 | +19,63 pp | +42,60 pp | -3,33 pp | os três sinais se repetem em 10/10 seeds |

O drift dos parâmetros hard-protected é exatamente zero. Logo, a máscara final
está funcionando. Hard superar beta 3 mostra que a proteção soft com beta 3 foi
mais fraca que hard neste protocolo; hard aprendido superar hard aleatório em
retenção mostra que o ranking contém sinal.
Entretanto, três seeds revertem o contraste de acurácia média final contra hard
aleatório e a aquisição da tarefa nova cai. O resultado identifica um mecanismo
de estabilidade útil, mas também um limite de plasticidade.

## 3. Interação com replay em 20 exemplos por classe

| Condição | Acurácia média final | Retenção T1 | Aquisição T2 | Tempo |
|---|---:|---:|---:|---:|
| Replay | 91,22 ± 0,59% | 92,00 ± 0,67% | 90,44 ± 1,64% | 142,20 ± 0,69 s |
| Hard + replay | 90,39 ± 0,42% | 93,67 ± 0,88% | 87,11 ± 1,71% | 313,86 ± 2,11 s |
| Hard + replay - replay | -0,83 pp | +1,67 pp | -3,33 pp | aproximadamente 2,21x |

A diferença de acurácia média final é negativa nas três seeds. A proteção compra
pequeno ganho de retenção com perda maior de aquisição e mais que duplica o tempo.
O gate rejeita escalar hard+replay nesse orçamento.

## 4. Eficiência sob pouca memória

| Exemplos/classe | Delta FAA: hard+replay - replay | Delta retenção T1 | Delta aquisição T2 | Sinais FAA |
|---:|---:|---:|---:|---|
| 1, primário | +2,17 pp | +8,00 pp | -3,67 pp | 3 positivos / 0 negativos |
| 5, secundário | +0,06 pp | +3,67 pp | -3,56 pp | 1 positivo / 2 negativos |
| 10, secundário | -0,94 pp | +2,33 pp | -4,22 pp | 0 positivos / 3 negativos |

O gate primário de memória 1 exigia cumulativamente:

1. delta de acurácia média final positivo em todas as seeds;
2. perda média de aquisição de T2 não pior que 2 pontos percentuais;
3. drift RMS e máximo exatamente zero nos parâmetros protegidos;
4. memória de replay idêntica dentro de cada par.

Os itens 1, 3 e 4 passaram. O item 2 falhou: a perda foi 3,67 pontos. Orçamentos
5 e 10 eram secundários e não podem ser promovidos post hoc; além disso, eles não
produziram um ganho consistente.

## Decisão

A evidência sustenta duas conclusões simultâneas:

1. SlowHeat foi implementado e validado mecanicamente em BERT, e superou o BERT
   sequencial padrão no diagnóstico de duas tarefas.
2. A topologia atual não demonstrou vantagem sobre replay: aumentar proteção
   reduz forgetting, mas também reduz a plasticidade da tarefa nova.

Portanto:

- não executar SlowHeat+replay na sequência completa de dez tarefas;
- não abrir outra grade de beta, cobertura, memória ou budget para resgatar o
  mesmo endpoint após observar estes resultados;
- não afirmar superioridade do SlowHeat sobre replay ou sobre baselines de CL;
- registrar o resultado como evidência positiva contra fine-tuning sequencial e
  como diagnóstico negativo frente ao replay;
- priorizar, no trabalho seguinte, baselines padrão e/ou uma hipótese mecanística
  nova, com protocolo e seeds de calibração/confirmação congelados antes da run.

Uma hipótese nova pode investigar plasticidade adaptativa, outra topologia de
proteção ou representação compartilhada, mas deve começar em outro protocolo.
Ela não é continuação confirmatória destes diagnósticos.

## Fontes primárias

- `results/bert_slowheat_review/diagnostic_summary.json`;
- `results/bert_slowheat_review/diagnostic_table.md`;
- `results/bert_slowheat_replay_diagnostic/diagnostic_summary.json`;
- `results/bert_slowheat_replay_diagnostic/diagnostic_table.md`;
- `results/bert_slowheat_replay_budget_diagnostic/diagnostic_summary.json`;
- `results/bert_slowheat_replay_budget_diagnostic/diagnostic_table.md`;
- `results/<estudo>/seed_<seed>/<condição>/protocol.json`;
- `results/<estudo>/seed_<seed>/<condição>/<método>/results.json`.

Atenção: existe também `results/bert_slowheat_diagnostic/`, uma versão de
**três seeds** do mesmo diagnóstico, com números diferentes (vanilla 46,83%
contra 48,47%; hard 71,61% contra 71,38%). Apesar do nome parecido com o deste
documento, ele **não** é fonte das tabelas acima. Ver
[results_index.md](results_index.md).

---

# Parte II — Runs históricas de dez tarefas

Auditoria de 14 de setembro de 2026.

Esta parte registra todas as onze saídas BERT/CLINC150 encontradas nos três
diretórios históricos principais. Os números foram lidos de cada
`results.json` e conferidos contra suas matrizes 10 x 10. Eles são resultados
reais, porém exploratórios: cada método possui uma única seed, o teste foi
consultado após cada estágio e todas as execuções registram uma árvore Git
`dirty`.

## II.1. Protocolo comum

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

## II.2. Métodos usados nas runs históricas

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

## II.3. Métricas

- FAA: acurácia média final Class-IL;
- Fgt: forgetting médio das tarefas antigas;
- TA: acurácia média final task-aware/oráculo;
- F1: macro-F1 final sobre as 150 classes;
- BWT: igual a `-Fgt` nestas runs;
- FWT: não medido e armazenado como `null`.

Todos os valores abaixo são proporções, não porcentagens.

## II.4. Resultados de duas épocas, seed 0

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

## II.5. Variantes de Heat de duas épocas, seed 0

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

## II.6. Resultados de dez épocas, seed 67

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

## II.7. Limitações e validade das runs históricas

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

## II.8. O que pode ser afirmado a partir das runs históricas

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

## II.9. Protocolo recomendado para a próxima execução

Nenhum destes dez itens foi executado até 22 de setembro de 2026.

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

## II.10. Implementação relacionada

- `src/dual_heater/bert.py`;
- `src/dual_heater/transformer.py`;
- `src/dual_heater/fast_heat.py`;
- `src/dual_heater/optim.py`;
- `experiments/split_clinc150.py`;
- [functional_slowheat_transformers.md](../mechanisms/functional_slowheat_transformers.md);
- [live_dashboard.md](../tooling/live_dashboard.md);
- [methods_catalog.md](../audits/methods_catalog.md).
