# MLP — evidência, protocolo e limites

Documento-guia da arquitetura MLP. É a fonte para a seção de resultados
principais do artigo. Contratos de mecanismo estão em
[functional_slowheat.md](functional_slowheat.md); o texto do paper está em
`article/manuscript.md`.

Escopo: `SlowHeatMLP` + `SlowHeatAdamW` em Split-MNIST e Permuted-MNIST
class-incremental, cabeça global compartilhada.

---

## Claim

Em Split-MNIST class-incremental, acrescentar Functional SlowHeat ao Replay
melhora a acurácia média final em **+0,87 pontos percentuais** (IC95%
[+0,29; +1,45], `p = 0,0052`, 20 seeds pré-registradas) e reduz o forgetting
médio em 1,46 p.p. (`p = 0,00059`), ao custo de 1,8x a 1,9x o tempo de parede
conforme a execução.

Este é o **único resultado confirmatório pré-registrado do projeto**. Todo o
restante deste documento é exploratório.

---

## Evidência citável

### E1 — Confirmação congelada (primária)

Contraste pré-registrado: `slowheat_replay_hidden_beta_30_budget_0.25` menos
`replay`. Endpoint primário: acurácia média final class-incremental após a
quinta tarefa.

| Métrica | Replay | SlowHeat+Replay | Diferença pareada | IC95% t (gl=19) | t | p |
|---|---:|---:|---:|---:|---:|---:|
| **Acurácia média final** | 0,77040 | 0,77909 | **+0,00869** | [+0,00293; +0,01445] | 3,157 | **0,0052** |
| Average forgetting | 0,27244 | 0,25781 | −0,01462 | [−0,02207; −0,00718] | −4,112 | 0,00059 |
| Classifier gap | 0,21293 | 0,20558 | −0,00735 | [−0,01304; −0,00166] | −2,704 | 0,0141 |
| Task-aware final | 0,98333 | 0,98467 | +0,00134 | [+0,00005; +0,00263] | 2,168 | 0,0431 |

Valores de acurácia e forgetting são **idênticos nas duas execuções**. O custo
difere, porque é a única dimensão sensível à máquina:

| Execução | Replay | SlowHeat+Replay | Diferença | t | Overhead |
|---|---:|---:|---:|---:|---:|
| `protocol_post_eval_fix` | 3,003 s | 5,496 s | +2,493 s | 35,95 | 1,83× |
| `protocol_post_eval_fix_d5b22ad` | 3,036 s | 5,828 s | +2,792 s | 135,06 | 1,92× |

Sinais por seed: 17+/3− na acurácia, 18−/2+ no forgetting, 20+/0− no tempo.
Bootstrap pareado (10.000 reamostragens, seed 20260815) concorda com o t:
[+0,00316; +0,01381].

Somente o endpoint primário foi pré-registrado. As demais linhas são
secundárias e descritivas; o ganho task-aware **não deve ser reportado como
positivo** — o t pareado dá `p = 0,043`, mas o teste exato de sinais dá
`p = 0,115` com 14+/6−.

**Artefatos:** `results/protocol_post_eval_fix/confirmation/` (commit `d5b22ad`)
e `results/protocol_post_eval_fix_d5b22ad/confirmation/` (commit `f2f7616`).
Pré-registro: `preregistration.lock.json`, `status = frozen_before_execution`,
`preregistered_at = 2026-08-15`, `sha256 = 015b3162…a9b2` idêntico nos dois.

### E2 — Replicação independente

As duas execuções foram comparadas par a par nos 20 arquivos de resultado por
seed: dos 100 campos escalares, **21 diferem e todos são de custo**
(`elapsed_seconds`, `optimizer_step_seconds`, `peak_memory_*`,
`selection_seconds`). Nenhuma métrica científica difere. Os arquivos têm hashes
SHA-256 distintos, ou seja, são execuções independentes em commits diferentes
que reproduziram os mesmos números.

### E3 — Contrastes pareados de 10 seeds (exploratório)

Suíte `dualheat_pairs`, endpoint primário, diferença sempre candidato menos
referência, com Holm aplicado dentro de cada dataset (4 contrastes):

**Split-MNIST**

| Contraste | Referência | Candidato | Diferença | p Holm | Sinais |
|---|---:|---:|---:|---:|---|
| vs DER++ | 82,40% | 85,58% | **+3,182 pp** | <0,0001 | 10+/0− |
| vs Convencional | 19,64% | 19,52% | −0,122 pp | 0,0005 | 0+/10− |
| vs Replay | 76,24% | 76,38% | +0,136 pp | 1,000 | 5+/5− |
| vs ER-ACE | 71,21% | 71,13% | −0,080 pp | 1,000 | 3+/7− |

**Permuted-MNIST** — os quatro contrastes sobrevivem a Holm:

| Contraste | Referência | Candidato | Diferença | p Holm | Sinais |
|---|---:|---:|---:|---:|---|
| vs Convencional | 72,34% | 80,01% | **+7,673 pp** | 0,0003 | 10+/0− |
| vs ER-ACE | 93,54% | 94,08% | +0,534 pp | 0,0004 | 9+/1− |
| vs Replay | 93,55% | 93,97% | +0,421 pp | 0,0011 | 10+/0− |
| vs DER++ | 95,63% | 95,73% | +0,101 pp | 0,0282 | 8+/2− |

**Artefatos:** `results/dualheat_pairs/{split_mnist,permuted_mnist}/pair_report.json`,
commit `be05068`, `status = exploratory_paired_suite`.

---

## Proveniência

| Item | E1 (confirmação) | E3 (pareados) |
|---|---|---|
| Pré-registro congelado | **sim** | não |
| Seeds declaradas antes | sim, 20 | não |
| Árvore Git limpa | **não** | **não** |
| Replicado independentemente | **sim** | não |
| Endpoint único declarado | sim | não (4 contrastes, Holm aplicado) |

**Risco para submissão:** ambas as execuções da confirmação registram
`git dirty = true` e o diff executado não foi fingerprintado, portanto não há
reprodução bit a bit a partir da proveniência. Ver
[results_provenance_status.md](results_provenance_status.md), item P1.

---

## O que NÃO se pode afirmar

- **Superioridade geral sobre continual learning.** O contraste confirmado é
  contra Replay simples, em Split-MNIST, com uma configuração
  (`beta=30`, `budget=0.25`).
- **Vantagem sobre DER++ ou ER-ACE.** Estes não fizeram parte da confirmação
  congelada. Em Split-MNIST, SlowHeat+DER++ mostra +3,18 p.p. sobre DER++, mas
  isso é exploratório e sem pré-registro próprio.
- **Vantagem de custo.** O overhead é sistemático: +83% a +92% de tempo,
  positivo em 20 de 20 seeds nas duas execuções.
- **Que o ganho task-aware seja real.** `p = 0,043` no t pareado, mas
  `p = 0,115` no teste de sinais.
- **Os números das seções 7–10 de `project_methods_and_results.md`.** Não são
  reproduzíveis a partir de nenhum artefato em disco.

---

## Ameaças à validade

1. **Árvore Git suja em 100% dos agregados MLP.** Impede reprodução exata.
2. **Benchmark único e fácil.** Split-MNIST com 5 tarefas de 2 classes tem teto
   alto; ganhos de ~1 p.p. podem não transferir. Permuted-MNIST reforça, mas é
   da mesma família.
3. **Sem tuning por método.** Todos os agregados usam `lr=1e-3` fixo. Um
   revisor pode alegar que os baselines não foram ajustados de forma justa.
4. **`beta` e `budget` não foram varridos** com fronteira de Pareto; o par
   (30; 0,25) vem de exploração anterior à confirmação.
5. **Efeito pequeno.** +0,87 p.p. com desvio-padrão de diferenças de ~1,2 p.p.
   depende do pareamento por seed para ser detectável.

---

## Protocolo

Cenário, controles pareados, métricas, correção de eval-mode e o resultado
completo estão em [confirmatory_protocol.md](confirmatory_protocol.md).
O log cronológico de todos os testes Split-MNIST, incluindo hipóteses
descartadas e a semântica dos nomes estruturados de método, está em
[split_mnist_experiment_log.md](split_mnist_experiment_log.md).
O protocolo pareado de 4 pares está em
[dualheat_paired_protocol.md](dualheat_paired_protocol.md).
O piloto sintético de 3 seeds, que **precede a implementação atual** e não é
evidência, está em [synthetic_ablation_pilot.md](synthetic_ablation_pilot.md).

## Execução

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=src:. python -m experiments.confirmatory_split_mnist --device cpu
```

Runners: `experiments/split_mnist.py`, `experiments/confirmatory_split_mnist.py`,
`experiments/confirmatory_statistics.py`, `experiments/split_mnist_suite.py`.