# Síntese e encerramento do diagnóstico BERT SlowHeat

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Consolidar os três diagnósticos BERT/CLINC150 em um registro científico auditável, documentar que o SlowHeat melhora retenção à custa de plasticidade e encerrar esta linha experimental sem iniciar uma sequência de dez tarefas.

**Architecture:** Não criar outro runner nem executar novo treinamento: os artefatos atuais já respondem ao gate predefinido. Fazer uma auditoria somente leitura dos `results.json`, `protocol.json` e summaries existentes; depois registrar a síntese em um documento canônico e atualizar README/manuscrito para que protocolo, resultado e decisão não se contradigam. Manter cada estudo estatisticamente separado, porque eles usam conjuntos de seeds diferentes e o diagnóstico de replay com memória 20 possui outro `source_sha256`.

**Tech stack:** Markdown, JSON, Python 3.12 somente para validação offline, pytest e Ruff já configurados no repositório.

---

## Current context / assumptions

- Há três estudos completos e validation-only:
  - `results/bert_slowheat_review/`: 60/60 runs, seis condições, dez seeds (`2, 9, 10, 28, 30, 32, 57, 67, 2005, 2012`);
  - `results/bert_slowheat_replay_diagnostic/`: 12/12 runs, quatro condições, três seeds (`11, 22, 33`);
  - `results/bert_slowheat_replay_budget_diagnostic/`: 18/18 runs, seis condições, três seeds (`11, 22, 33`).
- Todos os 90 `results.json` analisados são JSON finito; dentro de cada estudo, `task_order`, `data_sha256` e `source_sha256` são uniformes.
- Os três estudos usam o mesmo `data_sha256` (`5e1024bc9d7aceb03c18ae339e6d1264cb51ddafe3d5febd07b4515f77183413`) e a mesma ordem `banking -> credit_cards`.
- `bert_slowheat_review` e `bert_slowheat_replay_budget_diagnostic` usam `source_sha256=4bb93859f517e48de11726474a36ec9f4260c4faa8b42ec86d78f4bd2ca660f9`; `bert_slowheat_replay_diagnostic` usa `ef17c06040a39c31b6997afe8ac92e729637cf6842b7b592ef2aaeb6f0352d32`. Não combinar números entre estudos como se fossem uma única amostra pareada.
- Os três `environment.json` registram commit `5e96093cf20a3de1e77b5ea78c8650c25efb38c8` com `dirty=true`. O documento deve preservar essa limitação de proveniência.
- Resultado mecanístico de dez seeds:
  - hard aprendido contra hard aleatório: `+3,65 pp` de acurácia média final, mas com sinais `7 positivos / 3 negativos`; retenção de T1 `+9,37 pp` em `10/10` seeds; aquisição de T2 `-2,07 pp` em média;
  - hard contra beta 3: `+19,63 pp` de acurácia média final em `10/10` seeds e `+42,60 pp` de retenção de T1, mas `-3,33 pp` de aquisição de T2;
  - hard aprendido ainda esquece `41,03 ± 6,33 pp` de T1 e termina em `71,38 ± 2,81%` de acurácia média final;
  - drift protegido hard é exatamente zero, portanto o problema não é falha da máscara final.
- Resultado replay com 20 exemplos/classe:
  - `hard+replay - replay = -0,83 pp` de acurácia média final, negativo nas três seeds;
  - retenção de T1 melhora `+1,67 pp`, mas aquisição de T2 cai `-3,33 pp`;
  - hard+replay leva `313,86 s` contra `142,20 s` de replay, aproximadamente `2,21x` o tempo.
- Resultado de baixo orçamento:
  - memória 1, contraste primário: acurácia média final `+2,17 pp` em `3/3` seeds e retenção `+8,00 pp`, mas aquisição de T2 `-3,67 pp`;
  - memória 5: acurácia média final `+0,06 pp`, com sinais mistos, e aquisição de T2 `-3,56 pp`;
  - memória 10: acurácia média final `-0,94 pp`, negativa em `3/3` seeds, e aquisição de T2 `-4,22 pp`.
- O gate predefinido de memória 1 exigia simultaneamente diferença positiva de acurácia média final em todas as seeds, perda média de aquisição de T2 não pior que `2 pp`, drift protegido zero e memória pareada. O segundo requisito falhou (`-3,67 pp`). Um ganho no endpoint principal não permite ignorar um gate conjunto declarado antes da execução.
- Conclusão operacional: há sinal de ranking para retenção, mas não há melhoria Pareto aceitável. Não executar sequência de dez tarefas, não promover orçamento 5/10 post hoc e não abrir outra grade de beta/cobertura/plasticidade dentro desta mesma hipótese.
- Não alterar nenhum `results.json`, `protocol.json`, checkpoint, stream de telemetria, snapshot de Heat ou summary existente. A síntese será documentação nova, não reescrita da evidência.
- A árvore Git já contém arquivos modificados e não rastreados relacionados aos diagnósticos. Em eventual commit, adicionar somente os arquivos de documentação nomeados neste plano; nunca usar `git add .`.

## Files likely to change

- Create: `docs/results/bert_slowheat_diagnostic_results.md`
- Modify: `README.md:244-307`
- Modify: `article/manuscript.md:206-299`

No production code, tests, result artifacts or experiment runners need to change.

---

## Task 1: Revalidar a integridade dos três estudos sem mutação

**Objective:** Confirmar que a síntese parte de todos os artefatos esperados e que não há JSON não finito ou protocolo internamente misturado.

**Files:** No changes.

**Step 1: Confirm run counts, JSON finitude and within-study protocol identity**

Run from the repository root:

```bash
python3 - <<'PY'
import json
import math
from pathlib import Path

EXPECTED = {
    "bert_slowheat_review": (60, 60),
    "bert_slowheat_replay_diagnostic": (12, 12),
    "bert_slowheat_replay_budget_diagnostic": (18, 18),
}


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


for name, (expected_results, expected_protocols) in EXPECTED.items():
    root = Path("results") / name
    results = sorted(root.glob("seed_*/*/*/results.json"))
    protocols = sorted(root.glob("seed_*/*/protocol.json"))
    assert len(results) == expected_results, (name, len(results))
    assert len(protocols) == expected_protocols, (name, len(protocols))
    assert all(finite(json.loads(path.read_text())) for path in results)
    loaded = [json.loads(path.read_text()) for path in protocols]
    for key in ("task_order", "data_sha256", "source_sha256"):
        assert len({json.dumps(item[key], sort_keys=True) for item in loaded}) == 1
    print(f"{name}: {len(results)} results, {len(protocols)} protocols, finite, uniform")
PY
```

Expected output:

```text
bert_slowheat_review: 60 results, 60 protocols, finite, uniform
bert_slowheat_replay_diagnostic: 12 results, 12 protocols, finite, uniform
bert_slowheat_replay_budget_diagnostic: 18 results, 18 protocols, finite, uniform
```

**Step 2: Confirm cross-study data identity and disclose source differences**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

names = (
    "bert_slowheat_review",
    "bert_slowheat_replay_diagnostic",
    "bert_slowheat_replay_budget_diagnostic",
)
records = {}
for name in names:
    protocol_path = next((Path("results") / name).glob("seed_*/*/protocol.json"))
    protocol = json.loads(protocol_path.read_text())
    records[name] = {
        "data": protocol["data_sha256"],
        "source": protocol["source_sha256"],
        "tasks": protocol["task_order"],
    }

assert len({item["data"] for item in records.values()}) == 1
assert len({tuple(item["tasks"]) for item in records.values()}) == 1
assert len({item["source"] for item in records.values()}) == 2
for name, item in records.items():
    print(name, item)
PY
```

Expected: all three rows show the same data hash and task order; review/budget share one source hash, while the memory-20 replay study reports the second source hash.

**Step 3: Stop on any mismatch**

If either command fails, do not write the result narrative. Resolve missing/corrupt artifacts or protocol identity first; never reduce expected counts to fit what happens to be present.

---

## Task 2: Create the canonical result and decision record

**Objective:** Replace scattered interpretation across three generated tables with one conservative, cited record of what passed, what failed and why no full sequence follows.

**Files:**
- Create: `docs/results/bert_slowheat_diagnostic_results.md`

**Step 1: Write the complete document**

Create `docs/results/bert_slowheat_diagnostic_results.md` with exactly this content:

```markdown
# BERT SlowHeat: diagnóstico mecanístico, replay e decisão

Estado da análise: 19 de setembro de 2026.

## Escopo

Este documento sintetiza três estudos exploratórios validation-only sobre os
dois primeiros domínios oficiais do CLINC150 (`banking -> credit_cards`). Todos
usam uma cabeça Class-IL global, recebem a fronteira entre tarefas e avaliam
acurácia média final, retenção da primeira tarefa, aquisição da segunda tarefa,
drift de parâmetros e custo de execução.

Os estudos não formam uma única amostra. O diagnóstico mecanístico usa dez seeds;
os dois estudos de replay usam três seeds distintas. Contrastes são interpretados
somente dentro de cada estudo pareado.

## Integridade e proveniência

| Estudo | Runs | Seeds | Data SHA-256 | Source SHA-256 |
|---|---:|---|---|---|
| Mecanismo | 60/60 | 2, 9, 10, 28, 30, 32, 57, 67, 2005, 2012 | `5e1024...3413` | `4bb938...60f9` |
| Replay, 20 exemplos/classe | 12/12 | 11, 22, 33 | `5e1024...3413` | `ef17c0...2d32` |
| Replay, orçamentos 1/5/10 | 18/18 | 11, 22, 33 | `5e1024...3413` | `4bb938...60f9` |

Todos os 90 `results.json` são finitos. Dentro de cada estudo, ordem de tarefas,
hash dos dados e hash do código são uniformes. Os três ambientes registram o
commit `5e96093cf20a3de1e77b5ea78c8650c25efb38c8` com árvore Git suja. O estudo
de replay com memória 20 usa outro fingerprint de fonte; por isso seus valores
não devem ser reagrupados com os outros estudos como uma amostra única.

## 1. Diagnóstico mecanístico em dez seeds

| Contraste/condição | Acurácia média final | Retenção T1 | Aquisição T2 | Leitura |
|---|---:|---:|---:|---|
| Hard aprendido | 71,38 ± 2,81% | 55,27 ± 6,46% | 87,50 ± 2,06% | esquece 41,03 ± 6,33 pp de T1 |
| Hard aprendido - hard aleatório | +3,65 pp | +9,37 pp | -2,07 pp | FAA positiva em 7/10; retenção positiva em 10/10 |
| Hard aprendido - beta 3 | +19,63 pp | +42,60 pp | -3,33 pp | os três sinais se repetem em 10/10 seeds |

O drift dos parâmetros hard-protected é exatamente zero. Logo, a máscara final
está funcionando. Hard superar beta 3 mostra que a proteção soft é fraca; hard
aprendido superar hard aleatório em retenção mostra que o ranking contém sinal.
Entretanto, três seeds revertem o contraste de acurácia média final contra hard
aleatório e a aquisição da tarefa nova cai. O resultado identifica um mecanismo
de estabilidade, não uma melhoria geral.

## 2. Interação com replay em 20 exemplos por classe

| Condição | Acurácia média final | Retenção T1 | Aquisição T2 | Tempo |
|---|---:|---:|---:|---:|
| Replay | 91,22 ± 0,59% | 92,00 ± 0,67% | 90,44 ± 1,64% | 142,20 ± 0,69 s |
| Hard + replay | 90,39 ± 0,42% | 93,67 ± 0,88% | 87,11 ± 1,71% | 313,86 ± 2,11 s |
| Hard + replay - replay | -0,83 pp | +1,67 pp | -3,33 pp | aproximadamente 2,21x |

A diferença de acurácia média final é negativa nas três seeds. A proteção compra
pequeno ganho de retenção com perda maior de aquisição e mais que duplica o tempo.
O gate rejeita escalar hard+replay nesse orçamento.

## 3. Eficiência sob pouca memória

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

A evidência converge para a mesma dinâmica: aumentar proteção reduz forgetting,
mas reduz plasticidade da tarefa nova. O ranking aprendido é informativo para
retenção e a implementação hard bloqueia drift corretamente, porém nenhuma das
comparações com replay produz uma melhoria Pareto que satisfaça o gate declarado.

Portanto:

- não executar SlowHeat+replay na sequência completa de dez tarefas;
- não abrir outra grade de beta, cobertura, memória ou budget para resgatar o
  mesmo endpoint após observar estes resultados;
- não afirmar superioridade do SlowHeat sobre replay ou sobre baselines de CL;
- registrar o resultado como diagnóstico negativo útil da topologia atual;
- priorizar, no trabalho seguinte, baselines padrão e/ou uma hipótese mecanística
  nova, com protocolo e seeds de calibração/confirmacão congelados antes da run.

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
```

**Step 2: Validate that every declared number remains tied to machine-readable evidence**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

mechanism = json.loads(Path("results/bert_slowheat_review/diagnostic_summary.json").read_text())
replay = json.loads(Path("results/bert_slowheat_replay_diagnostic/diagnostic_summary.json").read_text())
budget = json.loads(Path("results/bert_slowheat_replay_budget_diagnostic/diagnostic_summary.json").read_text())

m = mechanism["paired_contrasts"]["learned_hard_minus_random_hard"]
assert round(100 * m["final_average_accuracy"]["mean"], 2) == 3.65
assert sum(value > 0 for value in m["final_average_accuracy"]["by_seed"].values()) == 7
assert sum(value > 0 for value in m["task1_retention"]["by_seed"].values()) == 10

r = replay["paired_contrasts"]["hard_replay_minus_replay"]
assert round(100 * r["final_average_accuracy"]["mean"], 2) == -0.83
assert all(value < 0 for value in r["final_average_accuracy"]["by_seed"].values())

b = budget["paired_contrasts"]["hard_replay_minus_replay_memory_1"]
assert round(100 * b["final_average_accuracy"]["mean"], 2) == 2.17
assert round(100 * b["task2_acquisition"]["mean"], 2) == -3.67
assert all(value > 0 for value in b["final_average_accuracy"]["by_seed"].values())
print("BERT SlowHeat synthesis verified")
PY
```

Expected: `BERT SlowHeat synthesis verified`.

**Step 3: Commit the canonical result record only if commit authorization is active**

```bash
git add docs/results/bert_slowheat_diagnostic_results.md
git commit -m "docs: record BERT SlowHeat diagnostic outcome"
```

Do not stage anything under `results/`.

---

## Task 3: Replace future-tense README guidance with the completed decision

**Objective:** Prevent readers from following obsolete commands as if the replay diagnostics were still pending.

**Files:**
- Modify: `README.md:244-307`

**Step 1: Replace the three diagnostic subsections**

Replace the block from `### Two-task SlowHeat mechanism diagnostic` through the paragraph ending with “they must not be searched post hoc for a favorable result.” with:

```markdown
### BERT SlowHeat two-task diagnostics

Three validation-only diagnostics are complete on the first two CLINC150 domains.
The ten-seed mechanism study found that learned hard protection improved T1
retention over matched random hard protection, but its final-accuracy contrast
reversed in 3/10 seeds and T2 acquisition fell. Hard masks had exactly zero
protected drift, so this is a stability-plasticity limitation rather than a
masking failure.

Replay then exposed the same trade-off. At 20 stored examples per class,
hard SlowHeat+replay lost 0.83 percentage points of final average accuracy to
replay and took about 2.21x as long. At the predeclared one-example primary
budget, hard SlowHeat+replay gained 2.17 points of final average accuracy but
lost 3.67 points of T2 acquisition, violating the 2-point acquisition gate.
Budgets 5 and 10 did not provide a consistent alternative.

The current decision is not to run a ten-task SlowHeat+replay sequence or add
another post-hoc parameter grid for this hypothesis. Full protocol, paired
contrasts, provenance limitations and source artifact paths are in
[`docs/results/bert_slowheat_diagnostic_results.md`](../../docs/results/bert_slowheat_diagnostic_results.md).
The executable diagnostic runners remain available for exact reproduction:

- `experiments.bert_slowheat_diagnostic`;
- `experiments.bert_slowheat_replay_diagnostic`;
- `experiments.bert_slowheat_replay_budget_diagnostic`.
```

Do not retain the old GPU commands in the main README: they invite duplicate expensive runs and imply an open decision. Reproduction commands remain discoverable through each runner’s `--help` and Git history.

**Step 2: Validate links and stale language**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

readme = Path("README.md").read_text()
assert Path("docs/results/bert_slowheat_diagnostic_results.md").is_file()
assert "BERT SlowHeat two-task diagnostics" in readme
assert "2.17 points" in readme
assert "violating the 2-point acquisition gate" in readme
assert "#### Replay interaction follow-up" not in readme
assert "#### Low-memory replay efficiency" not in readme
print("README diagnostic status verified")
PY
```

Expected: `README diagnostic status verified`.

**Step 3: Commit only the README update if authorized**

```bash
git add README.md
git commit -m "docs: mark BERT SlowHeat diagnostics complete"
```

---

## Task 4: Bring the manuscript into agreement with the completed transformer evidence

**Objective:** Include the BERT result without inflating an exploratory two-task diagnostic into an efficacy claim.

**Files:**
- Modify: `article/manuscript.md:206-299`

**Step 1: Insert a new section before the current Related Work section**

After the paragraph ending “require their own tuning before scientific comparison.”, insert:

```markdown
## 6. BERT/CLINC150 Mechanism Diagnostics

We evaluated the FFN+attention SlowHeat topology on the first two domains of a
150-way Class-IL CLINC150 stream. These runs used validation only and oracle task
boundaries. They are mechanism diagnostics rather than a full benchmark.

Across ten paired seeds, learned hard protection improved final average accuracy
by 3.65 percentage points over matched random hard protection, but the sign
reversed in three seeds. T1 retention improved by 9.37 points in all ten seeds,
while T2 acquisition decreased by 2.07 points on average. Protected parameter
drift was exactly zero, yet the learned hard condition still forgot 41.03 points
of T1. This indicates that the importance ranking contains retention signal but
the protected topology and plasticity allocation do not solve the task trade-off.

A predeclared replay interaction test rejected scaling the mechanism. With 20
stored examples per class, hard SlowHeat+replay trailed replay by 0.83 points of
final average accuracy in all three seeds and required approximately 2.21 times
the elapsed time. Under the primary one-example-per-class budget,
hard SlowHeat+replay improved final average accuracy by 2.17 points in all three
seeds, but reduced T2 acquisition by 3.67 points. This exceeded the predeclared
maximum acceptable acquisition loss of two points. The secondary five- and
ten-example budgets did not produce a consistent benefit.

We therefore do not advance this SlowHeat+replay configuration to the ten-task
sequence. The evidence supports a stability-plasticity effect and functional
ranking signal, not superiority over replay. Complete artifacts and provenance
limitations are recorded in `docs/results/bert_slowheat_diagnostic_results.md`.
```

**Step 2: Renumber only subsequent top-level sections**

Apply these heading changes, without altering their bodies:

```text
## 6. Related Work and Positioning       -> ## 7. Related Work and Positioning
## 7. Required Experiments Before Submission -> ## 8. Required Experiments Before Submission
## 8. Safe Claims                       -> ## 9. Safe Claims
## 9. Reproducibility Artifacts         -> ## 10. Reproducibility Artifacts
```

**Step 3: Update Safe Claims conservatively**

Under `Currently supported:`, append:

```markdown
- In a two-task BERT/CLINC150 diagnostic, learned hard rankings improved old-task
  retention over matched random masks, while replay comparisons exposed an
  acquisition-retention trade-off and failed the predeclared scaling gate.
```

Keep “SlowHeat outperforms established continual-learning baselines” under `Not currently supported:`. Do not claim transformer generalization from two tasks.

**Step 4: Add the new document to Reproducibility Artifacts**

Append this bullet to the artifact list:

```markdown
- BERT/CLINC150 diagnostic decision: `docs/results/bert_slowheat_diagnostic_results.md`
```

**Step 5: Validate manuscript scope and wording**

Run:

```bash
python3 - <<'PY'
from pathlib import Path

text = Path("article/manuscript.md").read_text()
assert "## 6. BERT/CLINC150 Mechanism Diagnostics" in text
assert "failed the predeclared scaling gate" in text
assert "not advance this SlowHeat+replay configuration" in text
assert "SlowHeat outperforms established continual-learning baselines" in text
assert text.count("docs/results/bert_slowheat_diagnostic_results.md") >= 2
print("manuscript scope verified")
PY
```

Expected: `manuscript scope verified`.

**Step 6: Commit the manuscript update if authorized**

```bash
git add article/manuscript.md
git commit -m "docs: add BERT SlowHeat negative diagnostic"
```

---

## Task 5: Verify repository consistency without running experiments

**Objective:** Ensure documentation changes are valid and no production behavior or raw evidence changed.

**Files:** No additional files unless a validation failure requires a documentation-only fix.

**Step 1: Run Markdown/reference assertions**

```bash
python3 - <<'PY'
from pathlib import Path

required = {
    "README.md": [
        "docs/results/bert_slowheat_diagnostic_results.md",
        "not to run a ten-task SlowHeat+replay sequence",
    ],
    "article/manuscript.md": [
        "BERT/CLINC150 Mechanism Diagnostics",
        "stability-plasticity effect",
    ],
    "docs/results/bert_slowheat_diagnostic_results.md": [
        "60/60",
        "12/12",
        "18/18",
        "+2,17 pp",
        "-3,67 pp",
        "não executar SlowHeat+replay na sequência completa",
    ],
}
for path, fragments in required.items():
    text = Path(path).read_text()
    for fragment in fragments:
        assert fragment in text, (path, fragment)
print("documentation assertions passed")
PY
```

Expected: `documentation assertions passed`.

**Step 2: Run the relevant lightweight tests**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest \
  tests/test_bert_slowheat_diagnostic.py \
  tests/test_bert_slowheat_diagnostic_common.py \
  tests/test_bert_slowheat_replay_diagnostic.py \
  tests/test_bert_slowheat_replay_budget_diagnostic.py -q
```

Expected: all selected tests PASS. These tests should not download data or initialize CUDA.

**Step 3: Run the complete CPU test suite**

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 -m pytest -q
```

Expected: all tests PASS.

**Step 4: Run Ruff even though only documentation changed**

```bash
python3 -m ruff check .
```

Expected: `All checks passed!`.

**Step 5: Prove that no result artifact changed**

```bash
git diff --check
git status --short
git diff --name-only -- results/
```

Expected:

- `git diff --check` emits nothing;
- `git diff --name-only -- results/` emits nothing;
- only `README.md`, `article/manuscript.md` and `docs/results/bert_slowheat_diagnostic_results.md` are newly changed by this plan, while pre-existing dirty files remain untouched.

**Step 6: Optional final documentation commit if earlier commits were not made and authorization is active**

```bash
git add \
  README.md \
  article/manuscript.md \
  docs/results/bert_slowheat_diagnostic_results.md
git commit -m "docs: conclude BERT SlowHeat diagnostics"
```

Never use `git add .`; do not push.

---

## Tests / validation acceptance criteria

Implementation is complete only when:

1. The audit confirms exactly 60, 12 and 18 finite results in the three studies.
2. Within-study protocol identity is uniform, and the cross-study source-hash difference is disclosed instead of hidden.
3. The canonical document reports raw signs as well as means; it does not use a mean-only superiority claim.
4. The memory-1 gate is explicitly marked failed because `-3,67 pp` violates the declared `-2 pp` acquisition limit.
5. The README no longer presents pending GPU runs or an undecided ten-task follow-up.
6. The manuscript reports transformer evidence as an exploratory two-task stability-plasticity result, not general efficacy.
7. Existing summaries, raw results, protocols, checkpoints and telemetry remain byte-untouched.
8. Focused tests, full pytest, Ruff and `git diff --check` pass.
9. No dataset load, dependency installation, GPU training, new parameter sweep, commit or push occurs without the corresponding authorization.

## Risks, tradeoffs, and open questions

- **Stopping is scientifically intentional:** The primary low-memory FAA contrast is positive, but its predeclared joint gate failed. Relaxing the acquisition threshold now would be post-hoc outcome switching.
- **Ranking signal is narrower than efficacy:** `10/10` positive retention differences against random masks support informative ranking; `7/10` FAA signs and persistent acquisition loss do not support an overall win.
- **Different source fingerprints:** The memory-20 replay study cannot be pooled numerically with the other studies. It remains valid as its own internally paired experiment.
- **Dirty provenance:** All environments report `dirty=true`; source fingerprints mitigate but do not eliminate this limitation. Do not describe the commit hash alone as sufficient reproduction provenance.
- **Only two tasks:** These runs diagnose mechanism dynamics. They do not establish ten-task scalability, BERT-base transfer, or superiority over standard CL baselines.
- **Potential future mechanism:** Adaptive plasticity or another protected topology may still be worth studying, but it must be framed as a new hypothesis with a new predeclared protocol, independent calibration seeds and held-out confirmation. It must not be presented as confirmation of the failed gate here.
- **Baseline priority:** If compute is allocated next, prefer a clean, separately planned benchmark of established baselines under the corrected protocol rather than another SlowHeat rescue sweep.
