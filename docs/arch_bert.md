# BERT — evidência, protocolo e limites

Documento-guia do host BERT. É a fonte para a seção de limites do artigo. O
contrato genérico de Transformers está em
[functional_slowheat_transformers.md](functional_slowheat_transformers.md).

Escopo: `SlowHeatBertForSequenceClassification` sobre
`google/bert_uncased_L-4_H-256_A-4`, benchmark CLINC150 class-incremental com
cabeça global de 150 intenções.

---

## Claim

Em um diagnóstico de duas tarefas do CLINC150, Functional SlowHeat melhora a
acurácia média final sobre fine-tuning sequencial em **+22,92 pontos
percentuais** (10 de 10 seeds), e o ranking de importância aprendido supera
máscaras aleatórias pareadas em retenção (10 de 10 seeds). **Entretanto, contra
replay o método não se sustenta**: a acurácia média final cai 0,83 p.p. nas três
seeds e o tempo mais que duplica.

Este é um **resultado negativo deliberadamente reportado**. Ele delimita onde o
mecanismo funciona (contra fine-tuning ingênuo) e onde não funciona (contra uma
baseline forte de continual learning).

**Atualização de 24/09/2026:** a suíte `hard_vs_soft` removeu o confundimento
entre regime de proteção e arquitetura. Hard não vence soft em nenhum dos cinco
alvos MLP/CNN, e perde onde a capacidade aperta. Logo, o +22,92 p.p. acima
**não pode ser atribuído ao regime hard**; ele é específico desta arquitetura
ou do seu regime de capacidade. Ver
[hard_vs_soft_results.md](hard_vs_soft_results.md).

---

## Evidência citável

### B1 — SlowHeat contra fine-tuning sequencial (10 seeds)

Dois primeiros domínios do CLINC150 (`banking -> credit_cards`), somente
validação.

| Condição | Acurácia média final | Δ vs vanilla | Forgetting de T1 | Sinais |
|---|---:|---:|---:|---|
| BERT vanilla | 48,47 ± 2,41% | — | 90,90 ± 4,69 pp | — |
| SlowHeat β=3 | 51,75 ± 2,58% | +3,28 pp | 83,63 ± 4,56 pp | 9+/1− |
| SlowHeat β=10 | 59,65 ± 3,21% | +11,18 pp | 67,20 ± 6,54 pp | 10+/0− |
| SlowHeat β=30 | 66,05 ± 2,95% | +17,58 pp | 53,47 ± 6,59 pp | 10+/0− |
| SlowHeat hard aprendido | 71,38 ± 2,81% | **+22,92 pp** | 41,03 ± 6,33 pp | 10+/0− |
| SlowHeat hard aleatório | 67,73 ± 3,21% | +19,27 pp | 50,40 ± 6,07 pp | 10+/0− |

Monotonicidade em `beta` é consistente: mais proteção, mais retenção.

### B2 — O ranking aprendido carrega sinal

| Contraste | Acurácia final | Retenção T1 | Aquisição T2 | Sinais |
|---|---:|---:|---:|---|
| Hard aprendido − hard aleatório | +3,65 pp | +9,37 pp | −2,07 pp | acurácia 7/10; **retenção 10/10** |
| Hard aprendido − β=3 | +19,63 pp | +42,60 pp | −3,33 pp | 10/10 nos três |

**O drift dos parâmetros protegidos é exatamente zero**, o que valida a máscara
mecanicamente. A retenção superar a máscara aleatória em 10 de 10 seeds mostra
que a importância funcional aprendida não é ruído. O custo aparece na aquisição
da tarefa nova, que cai.

### B3 — Contra replay o método perde (3 seeds)

| Condição | Acurácia final | Retenção T1 | Aquisição T2 | Tempo |
|---|---:|---:|---:|---:|
| Replay (20 ex/classe) | 91,22 ± 0,59% | 92,00 ± 0,67% | 90,44 ± 1,64% | 142,20 s |
| Hard + replay | 90,39 ± 0,42% | 93,67 ± 0,88% | 87,11 ± 1,71% | 313,86 s |
| **Diferença** | **−0,83 pp** | +1,67 pp | −3,33 pp | **≈2,21×** |

Negativo nas três seeds. A proteção compra retenção e paga com aquisição e
tempo.

### B4 — Gate de eficiência sob pouca memória (reprovado)

| Exemplos/classe | Δ acurácia | Δ retenção T1 | Δ aquisição T2 | Sinais |
|---:|---:|---:|---:|---|
| 1 (primário) | +2,17 pp | +8,00 pp | −3,67 pp | 3+/0− |
| 5 (secundário) | +0,06 pp | +3,67 pp | −3,56 pp | 1+/2− |
| 10 (secundário) | −0,94 pp | +2,33 pp | −4,22 pp | 0+/3− |

O gate primário exigia, cumulativamente: acurácia positiva em todas as seeds;
perda de aquisição não pior que 2 p.p.; drift zero; memória idêntica por par.
**Os itens 1, 3 e 4 passaram; o item 2 falhou** (perda de 3,67 p.p.). Orçamentos
5 e 10 eram secundários e não podem ser promovidos post hoc.

**Artefatos:** `results/bert_slowheat_review/`,
`results/bert_slowheat_replay_diagnostic/`,
`results/bert_slowheat_replay_budget_diagnostic/`. Commit-base
`5e96093`. Detalhe completo, incluindo as 11 runs históricas de dez tarefas, em
[bert_slowheat_diagnostic_results.md](bert_slowheat_diagnostic_results.md).

---

## Proveniência

| Item | B1/B2 | B3 | B4 |
|---|---|---|---|
| Seeds | 10 | 3 | 3 |
| Gate declarado antes | — | — | **sim, e reprovou** |
| Árvore Git limpa | **não** | **não** | **não** |
| Somente validação | sim | sim | sim |
| Fingerprint de fonte uniforme | sim | **outro** | sim |

**Atenção:** o estudo de replay com memória 20 usa um fingerprint de fonte
diferente dos outros dois (`ef17c0…` contra `4bb938…`). Seus valores **não podem
ser reagrupados** com os demais como amostra única.

O gate reprovado de B4 é um ativo metodológico: mostra um critério declarado
antes da run que de fato barrou a promoção de um resultado.

---

## O que NÃO se pode afirmar

- **Que SlowHeat funciona em NLP.** O diagnóstico é de duas tarefas; a sequência
  completa de dez domínios não foi executada com o mecanismo.
- **Que SlowHeat supera replay.** Não supera, e isso está medido.
- **Que o resultado escala para BERT-base ou LLMs.** O backbone é um BERT
  reduzido, com 4 camadas e hidden 256.
- **Que as 11 runs históricas de dez tarefas suportam qualquer conclusão.** Cada
  método tem uma seed, o teste foi consultado a cada estágio, e o diretório
  `bert_10epoch` mistura duas sessões com metadados sobrescritos.
- **Que o ganho de B1 venha do regime de proteção hard.** A suíte
  `hard_vs_soft` testou os dois regimes em MLP e CNN sob protocolo congelado, e
  hard **não** vence soft em nenhum dos cinco alvos, perdendo em
  Split-CIFAR-100/MLP (−1,41 p.p., 10/10 seeds). O ganho do BERT é específico
  desta arquitetura ou do seu regime de capacidade — não é uma propriedade
  transferível do congelamento binário. Ver
  [hard_vs_soft_results.md](hard_vs_soft_results.md).
- **Que o achado negativo de B3/B4 seja uma regra geral.** Hard+replay contra
  replay replica em 2 dos 5 alvos não-Transformer (CIFAR-100/MLP −3,19 p.p.,
  CIFAR-10/CNN −6,22 p.p.), **inverte** em 1 (CIFAR-10/MLP, +2,55 p.p.) e é
  nulo em 2.
- **O contraste `dualheat_global_topk − slowheat_global`.** Continua fora da
  lista de pares agregados (achado #10 da auditoria), então FastHeat top-k não
  tem avaliação inferencial.

---

## Ameaças à validade

1. **Duas tarefas não são continual learning.** É um diagnóstico de mecanismo.
2. **Backbone pequeno.** 11,2M de parâmetros; a dinâmica de importância pode
   diferir em escala.
3. **Árvore Git suja em todas as runs.**
4. **Trade-off retenção/aquisição não foi resolvido**, apenas medido. É a
   limitação central do método nesta arquitetura.
5. **A decisão de não escalar foi tomada após ver os resultados** — o que está
   correto do ponto de vista de protocolo (evita grade post-hoc), mas significa
   que não há confirmação independente do achado negativo.

---

## Decisão registrada

O projeto decidiu **não** executar SlowHeat+replay na sequência completa de dez
tarefas e **não** abrir nova grade de beta, cobertura, memória ou budget para
resgatar o mesmo endpoint. Uma hipótese nova deve começar em outro protocolo,
com seeds e critérios congelados antes; ela não é continuação confirmatória
destes diagnósticos.

---

## Protocolo

Escopo, integridade, proveniência e as 11 runs históricas em
[bert_slowheat_diagnostic_results.md](bert_slowheat_diagnostic_results.md).
Ablação de cobertura completa e ownership do grafo de parâmetros em
[bert_full_coverage_ablation.md](bert_full_coverage_ablation.md).
Runner: `experiments/split_clinc150.py`. Host: `src/dual_heater/bert.py`.