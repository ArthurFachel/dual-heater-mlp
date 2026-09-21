# BERT SlowHeat: diagnóstico mecanístico, replay e decisão

Estado da análise: 19 de setembro de 2026.

## Escopo

Este documento sintetiza três estudos exploratórios, somente em validação, sobre
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
