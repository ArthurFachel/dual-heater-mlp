# Roadmap: dois artigos (ICML + IJCNN) a partir da linha "magnitude de ativação vs. esquecimento catastrófico"

**Estado:** plano de trabalho, não pré-registro. Nada aqui congela hipótese ou
endpoint — cada experimento citado (Fases 2.4 e 3) exige seu próprio protocolo
congelado antes da primeira seed, como manda a regra do projeto. Este documento
pode ser editado a qualquer momento; os protocolos, não.

**Data:** 2026-09-29
**Repo:** `/mnt/B-SSD/fachel/dual-heater-mlp`
**Ambiente:** `.venv/bin/python` (o `python3` do sistema não tem numpy)
**Idioma de trabalho:** pt-BR. **Idioma dos artigos:** inglês.

---

## Goal

Produzir **dois artigos distintos e não-sobrepostos** a partir da evidência já
acumulada: um paper de **protocolo/instrumentação** para o ICML (o instrumento
de pareamento de plasticidade e o que ele rejeita) e um paper **empírico
focado** para o IJCNN (magnitude de ativação `|z|` como critério de proteção
suficiente em transformers, mais o resultado de seletor de replay), cada um
terminando num manuscrito submetido.

---

## Current context / assumptions

### O que o projeto já estabeleceu (tudo pré-registrado, 10–20 seeds)

| # | Resultado | Força | Arquivo |
|---|---|---|---|
| R-A | Proteção seletiva por ativação bate máscara aleatória em BERT: **+12,23 pp, 10/10 seeds**, p_Holm = 0,0039 | **Forte, positivo** | `docs/results/criterion_ablation_results.md` |
| R-B | O componente do gradiente não compra nada: `\|z·dL/dz\|` empata com `\|z\|` (−0,17 pp, 5+/5, p = 1,00), com falsificador de variância descartando ranking degenerado | **Forte, negativo** | idem |
| R-C | Em LoRA, todo o ganho do `exact` vem de congelar `A` (= LoRA-FA, 2023); a máscara SlowHeat não adiciona nada (5+/5, p = 1,00) | **Forte, negativo** | `docs/results/exact_decomposition_results.md` |
| R-D | Seletor de replay: perda alta **piora** (até −4,05 pp), representatividade **melhora** (+0,67 a +1,01 pp); unânime 20/20 em 4 learners | **Forte, independente do método da casa** | `docs/results/replay_selector_results.md` |
| R-E | `effective_plasticity()` media a máscara **só sobre parâmetros com binding**; o endpoint primário de 28/09 comparou E=0,532 contra E=0,850 sem saber | **Achado de instrumentação** | `docs/lora/lora_confirmation_results.md` §R3 |
| R-F | `lr_control < vanilla` (exploratório) **não replicou**: sinal inverte, p = 0,754 / 0,344 | Correção documental aplicada | 7 ocorrências corrigidas em `45de59e` |

### Estado da run em andamento (no momento da escrita deste plano)

`scripts/launch_plasticity_matched.sh` está rodando destacado (3 shards,
PPID=1), pré-registrado em `goals/protocol_plasticity_matched.md`:

- **Contraste primário:** `frozen_a_control − lr_control_062` em forgetting.
- **Pareamento:** ambos a E_superfície = 0,622461713 (verificado exato em smoke).
- 10 seeds na banda 6.000.003+, Holm sobre família de 2, sinal exato.
- Progresso observado: ~18 min/seed; shard 0 tem 4 seeds (caminho crítico,
  ~72 min total).
- **Este plano não depende do desfecho** — a Fase 0 abaixo tem um galho
  explícito para cada um dos dois resultados possíveis.

### Prazos (verificados em 29/09/2026, com ressalva)

| venue | deadline | páginas | fonte |
|---|---|---|---|
| **ICML 2027** | abstract **~16/01/2027**, paper **~22/01/2027** AoE | 8 + refs | agregadores (OpenCurious, ResearchTheta) — **CFP oficial ainda não publicado** |
| **IJCNN 2027** | paper **31/01/2027** (23:59 UTC-12) | **6 páginas, formato IEEE**, via CMT | `ijcnn.org/2027`, `mldeadlines.com` |

> ⚠️ **Os dois agregadores discordam sobre o ICML** (22/01 vs. 28/01) e o site
> oficial do ICML 2027 ainda não publicou o CFP. A Tarefa 0.1 exige confirmar na
> fonte oficial antes de qualquer planejamento de calendário. **Não trate as
> datas acima como fato.** IJCNN 2027: Cape Town, 14–18/06/2027,
> notificação 15/03, camera-ready 12/04.

Os deadlines ficam a **~16 semanas** da data deste plano, e são **9 dias
separados** entre si — perto o suficiente para que o trabalho experimental seja
compartilhado, longe o suficiente para que os manuscritos sejam escritos em
sequência, não em paralelo.

### Premissas assumidas (corrija-me se alguma estiver errada)

1. Há acesso possível a 4× A6000, mas **ainda não confirmado**. Todo o plano
   principal cabe nas 3 GPUs locais (2× 1080 Ti + Titan Xp); o uso das A6000
   aparece só como upside opcional na Fase 5.
2. Autoria e ordem de autores são decisão do Fachel, fora deste plano.
3. O `article/manuscript.md` atual (604 linhas, eixo "paper de método") será
   **canibalizado, não editado** — vira fonte de seções para os dois novos
   manuscritos e depois é arquivado.

---

## Architecture / proposed approach

**A divisão editorial é por tipo de contribuição, não por dataset.** O ICML
recebe o argumento metodológico — *pareamento de plasticidade é uma condição de
validade que a literatura de CL não verifica, aqui está o instrumento, e ele
rejeita os resultados dos próprios autores*. O IJCNN recebe o achado empírico
autocontido e positivo — *em transformers, `|z|` puro iguala critérios de
importância que custam um backward hook, e a escolha do seletor de replay importa
mais do que a escolha do método*.

Isso respeita a ideia original (magnitude de ativação contra esquecimento): ela
**é** o resultado R-A/R-B, que é forte e positivo, e vira o coração do IJCNN. O
ICML fica com o que a evidência realmente sustenta como contribuição de primeira
linha: o instrumento e os três resultados negativos pré-registrados.

**Regra de não-sobreposição, inegociável:** R-A/R-B/R-D são o *corpo* do IJCNN e
aparecem no ICML **apenas como uma linha de tabela citando o paper irmão**.
R-C/R-E/o resultado da run em andamento são o *corpo* do ICML e **não aparecem**
no IJCNN. As duas submissões devem citar uma à outra como "companion paper,
under review".

---

## Roteiro em fases

```
Fase 0  (semana 1)      Fechar a run pendente e decidir o galho
Fase 1  (semanas 1-2)   Baselines: o gap que reviewer vai cobrar
Fase 2  (semanas 2-5)   Generalizar o instrumento para EWC/SI/MAS  [ICML]
Fase 3  (semanas 4-7)   Replicar |z| em um segundo host           [IJCNN]
Fase 4  (semanas 6-9)   Escrita IJCNN (6 pgs) -> submissão 31/01
Fase 5  (semanas 8-14)  Escrita ICML (8 pgs) -> submissão ~22/01
Fase 6  (semana 15)     Artefato, checklist, revisão final
```

> Atenção à ordem contraintuitiva: **o ICML vence primeiro** (~22/01) mas é o
> paper maior. Por isso a escrita do ICML **começa antes** e termina depois; o
> IJCNN (6 páginas, deadline 31/01) é escrito no meio, aproveitando que R-A/R-B/R-D
> já estão prontos e não precisam de experimento novo.

---

# FASE 0 — Fechar a run pendente e decidir o galho

## Tarefa 0.1 — Confirmar os deadlines na fonte oficial

**Por quê:** os agregadores discordam em 6 dias no ICML, e o plano inteiro
pendura nessas datas.

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
curl -sL --max-time 25 -A "Mozilla/5.0" "https://icml.cc/Conferences/2027/CallForPapers" \
  | sed -e 's/<[^>]*>/|/g' | tr -s '|' '\n' \
  | grep -iE "deadline|abstract|notification|january|february" | head -20
```

**Saída esperada:** as datas oficiais, ou vazio se o CFP ainda não subiu.
**Se vazio:** registre `PENDENTE` e re-execute semanalmente. Não invente datas.

Grave o resultado em `goals/venue_deadlines.md` com data da consulta e URL.

---

## Tarefa 0.2 — Verificar as 8 checagens de integridade da run pendente

**Só execute quando os três shards tiverem terminado** (`ls
results/plasticity_matched/ | grep -c seed_` retorna `10`).

Crie `scripts/verify_plasticity_matched.py`:

```python
#!/usr/bin/env python3
"""Verificações H1-H8 de goals/protocol_plasticity_matched.md.

Roda ANTES de olhar qualquer endpoint. Se alguma falhar, o contraste não é
interpretável e a run precisa ser diagnosticada, não analisada.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("results/plasticity_matched")
TARGET_E = 0.6224617129892229
ARMS = ("vanilla", "lr_control", "frozen_a_control")


def main() -> int:
    seed_dirs = sorted(ROOT.glob("seed_*"))
    failures: list[str] = []

    if len(seed_dirs) != 10:
        failures.append(f"H1: esperava 10 seeds, achei {len(seed_dirs)}")

    for seed_dir in seed_dirs:
        manifest_path = seed_dir / "manifest.json"
        if not manifest_path.exists():
            failures.append(f"H2: {seed_dir.name} sem manifest.json")
            continue
        manifest = json.loads(manifest_path.read_text())
        results = {row["method"]: row for row in manifest["results"]}

        if set(results) != set(ARMS):
            failures.append(f"H3: {seed_dir.name} tem braços {sorted(results)}")
            continue

        # H4: os dois braços do contraste pareados na superfície
        e_frozen = results["frozen_a_control"]["surface_plasticity"]
        e_lr = results["lr_control"]["surface_plasticity"]
        if abs(e_frozen - e_lr) > 1e-9:
            failures.append(f"H4: {seed_dir.name} E não pareado: {e_frozen} vs {e_lr}")

        # H5: ambos no alvo declarado
        for arm in ("frozen_a_control", "lr_control"):
            got = results[arm]["surface_plasticity"]
            if abs(got - TARGET_E) > 1e-9:
                failures.append(f"H5: {seed_dir.name}/{arm} E={got}, alvo={TARGET_E}")

        # H6: vanilla é a referência, E = 1
        if results["vanilla"]["surface_plasticity"] != 1.0:
            failures.append(f"H6: {seed_dir.name} vanilla E != 1.0")

        # H7: tokens de treino pareados dentro da seed
        tokens = {arm: results[arm]["train_tokens"] for arm in ARMS}
        if len(set(tokens.values())) != 1:
            failures.append(f"H7: {seed_dir.name} tokens não pareados: {tokens}")

        # H8: o lr_control realmente recebeu o tratamento
        lr_scale = results["lr_control"]["learning_rate_scale"]
        if abs(lr_scale - TARGET_E) > 1e-12:
            failures.append(f"H8: {seed_dir.name} lr_control scale={lr_scale}")

    if failures:
        print("FALHOU:")
        for item in failures:
            print(" -", item)
        return 1

    print(f"OK: {len(seed_dirs)} seeds, 8/8 verificações passaram")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Verificação:**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp && PYTHONPATH=. .venv/bin/python scripts/verify_plasticity_matched.py
```

**Saída esperada:** `OK: 10 seeds, 8/8 verificações passaram`
**Se falhar:** pare. Diagnostique antes de olhar endpoint. Um contraste com
pareamento quebrado não é interpretável.

---

## Tarefa 0.3 — Analisar o contraste primário

Só depois da 0.2 passar. Reuse os helpers existentes — não reimplemente
estatística:

- `experiments/confirmatory_statistics.py:exact_two_sided_sign_test`
- `experiments/analyze_confirmation.py:holm`

Crie `scripts/analyze_plasticity_matched.py` no mesmo formato de
`experiments/analyze_confirmation.py`. Endpoint primário: `forgetting`.
Secundário: `final_average_accuracy`. Holm sobre família de 2.

**Escreva o resultado em `docs/results/plasticity_matched_results.md`**, seguindo
a estrutura dos outros três (Pergunta / Desenho / Resultado / Interpretação /
Limites). **Escreva o relatório antes de decidir o que ele significa para o
artigo** — nessa ordem, não a inversa.

---

## Tarefa 0.4 — O galho

Os dois desfechos são publicáveis e **ambos mantêm o plano de dois artigos**. O
que muda é a frase central do ICML.

**Galho A — `frozen_a_control − lr_control_062` significativo (LoRA-FA sobrevive
ao pareamento correto):**
> Claim do ICML: *"o pareamento por superfície é necessário e mudamos a
> conclusão de três documentos; o efeito de LoRA-FA sobrevive quando pareado
> corretamente, mas o efeito atribuído à proteção seletiva não."*
> A Fase 2 continua como planejada.

**Galho B — empate (o efeito morre sob pareamento honesto):**
> Claim do ICML: *"o instrumento derrubou o resultado confirmatório dos próprios
> autores; um resultado pré-registrado passou por Holm e cinco gates e ainda
> assim descrevia condições que não existiam."*
> Este é o **caso mais forte** para um paper de protocolo. A Fase 2 ganha
> prioridade máxima, porque generalizar o instrumento passa a ser a contribuição
> inteira.

**Commit ao fim da Fase 0:**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add scripts/verify_plasticity_matched.py scripts/analyze_plasticity_matched.py docs/results/plasticity_matched_results.md
git commit -m "resultado: pareamento de plasticidade sobre a superficie, 10 seeds"
```

---

# FASE 1 — Baselines (o gap que qualquer reviewer vai cobrar)

**O problema, dito sem rodeios:** a busca no repo mostra que EWC existe apenas em
`experiments/split_mnist.py:183`. Não há EWC/SI/MAS/LwF nos hosts BERT nem Qwen.
Um paper de CL submetido a ICML **sem baselines padrão no host principal** é
rejeitado por isso sozinho, independente da qualidade do argumento.

## Tarefa 1.1 — Auditar o que realmente existe

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
grep -rniE "\"ewc\"|'ewc'|synaptic|\bmas\b|lwf|distill" --include=*.py src/ experiments/ | grep -v test | head -30
```

**Saída esperada:** ocorrências só em `experiments/split_mnist.py` e
`experiments/split_mnist_suite.py`. Registre o inventário real em
`docs/audits/baseline_inventory.md`.

## Tarefa 1.2 — TDD: porta EWC para o host BERT

Este é o maior bloco de código do plano. Siga o ciclo por método, um commit cada.

**RED** — crie `tests/test_bert_baselines.py`:

```python
"""EWC no host BERT: o termo de penalidade precisa existir e morder.

Sem isto o paper não tem baseline padrão no host principal, que é o primeiro
item que um reviewer de CL cobra.
"""

from __future__ import annotations

import pytest


def test_ewc_is_a_registered_clinc_method() -> None:
    from experiments.split_clinc150 import SUPPORTED_METHODS

    assert "ewc" in SUPPORTED_METHODS


def test_ewc_penalty_is_zero_on_the_first_task() -> None:
    """Sem tarefa anterior não há o que penalizar."""
    from dual_heater.ewc import ewc_penalty

    assert ewc_penalty(params={}, anchors={}, fisher={}, strength=1.0) == pytest.approx(0.0)


def test_ewc_penalty_grows_with_drift() -> None:
    """Penalidade quadrática: dobrar o drift quadruplica a penalidade."""
    import torch

    from dual_heater.ewc import ewc_penalty

    anchors = {"w": torch.zeros(4)}
    fisher = {"w": torch.ones(4)}

    near = ewc_penalty(
        params={"w": torch.full((4,), 0.1)}, anchors=anchors, fisher=fisher, strength=1.0
    )
    far = ewc_penalty(
        params={"w": torch.full((4,), 0.2)}, anchors=anchors, fisher=fisher, strength=1.0
    )
    assert far == pytest.approx(4.0 * near, rel=1e-5)


def test_ewc_penalty_respects_fisher_weighting() -> None:
    """Um parâmetro com Fisher zero não contribui, por mais que tenha driftado."""
    import torch

    from dual_heater.ewc import ewc_penalty

    penalty = ewc_penalty(
        params={"w": torch.tensor([10.0, 10.0])},
        anchors={"w": torch.zeros(2)},
        fisher={"w": torch.tensor([1.0, 0.0])},
        strength=1.0,
    )
    # Só o primeiro elemento conta: 0.5 * 1.0 * 10^2 = 50
    assert penalty == pytest.approx(50.0)
```

**Rodar e verificar que falha:**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp && CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python -m pytest tests/test_bert_baselines.py -q
```

**Saída esperada:** `4 failed` com `ModuleNotFoundError: No module named 'dual_heater.ewc'`.

**GREEN** — crie `src/dual_heater/ewc.py`:

```python
"""Elastic Weight Consolidation, o baseline padrão de CL baseado em penalidade.

Existia apenas dentro de experiments/split_mnist.py, acoplado àquele runner.
Extraído para um módulo próprio porque o host BERT e o host Qwen precisam do
mesmo termo, e duplicá-lo seria a terceira cópia.

Referência: Kirkpatrick et al., PNAS 2017 (arXiv 1612.00796).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def ewc_penalty(
    *,
    params: Mapping[str, torch.Tensor],
    anchors: Mapping[str, torch.Tensor],
    fisher: Mapping[str, torch.Tensor],
    strength: float,
) -> torch.Tensor | float:
    """Penalidade quadrática ponderada por Fisher: (λ/2) Σ F_i (θ_i − θ*_i)².

    Retorna 0.0 (float) quando não há âncora, que é o caso da primeira tarefa.
    Parâmetros presentes em ``params`` mas ausentes de ``anchors`` são ignorados:
    uma cabeça que cresceu entre tarefas não tem âncora e não deve ser penalizada.
    """

    total: torch.Tensor | float = 0.0
    for name, anchor in anchors.items():
        if name not in params or name not in fisher:
            continue
        drift = params[name] - anchor
        total = total + (fisher[name] * drift.pow(2)).sum()
    if isinstance(total, float):
        return 0.0
    return 0.5 * strength * total
```

**Rodar e verificar que passa:**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp && CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python -m pytest tests/test_bert_baselines.py -q
```

**Saída esperada:** `1 failed, 3 passed` (o teste de registro em
`SUPPORTED_METHODS` ainda falha — é a próxima subtarefa).

**Commit:**

```bash
git add src/dual_heater/ewc.py tests/test_bert_baselines.py
git commit -m "feat: extrai penalidade EWC para modulo proprio, com testes"
```

## Tarefa 1.3 — Ligar EWC ao runner do CLINC150

Registre `"ewc"` em `experiments/split_clinc150.py` (o tuple `methods` da
`SplitCLINC150Config`, linha ~266, e o dispatch de método). Acumule o Fisher
diagonal ao fim de cada tarefa, guarde as âncoras, some `ewc_penalty` à loss.

**Teste de mordida** (acrescente a `tests/test_bert_baselines.py`):

```python
def test_ewc_reduces_drift_versus_vanilla_on_a_two_task_smoke() -> None:
    """Um EWC que não reduz drift não está ligado, por mais que rode."""
    # Smoke de CPU: 2 tarefas, 2 exemplos/classe, 1 época.
    # Asserção: drift RMS dos parâmetros do encoder com ewc < com vanilla.
    # (Implementar chamando run_split_clinc150 com task_limit=2 nos dois métodos.)
```

**Por que este teste importa:** o modo de falha silencioso aqui é EWC rodar com
`strength` que nunca chega na loss — o método aparece na tabela, produz números
plausíveis, e é vanilla com outro nome. Exatamente o bug que já mordeu este
projeto duas vezes (`run_diagnostic` não propagando `importance_criterion`, e o
`learning_rate_scale` não chegando no `lr_control`).

## Tarefa 1.4 — Repetir para SI e MAS

Mesmo ciclo. SI (Zenke et al., 2017) e MAS (Aljundi et al., 2018) compartilham a
forma `Σ Ω_i (θ_i − θ*_i)²`; só muda como `Ω` é estimado. **DRY:** reuse
`ewc_penalty` passando `fisher=omega` — não escreva três penalidades quadráticas.

**Verificação ao fim da fase:**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp && CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python -m pytest -q
```

**Saída esperada:** `>= 765 passed` (750 atuais + os novos), 0 failed.

---

# FASE 2 — Generalizar o instrumento para métodos de penalidade `[ICML]`

**Esta é a contribuição central do ICML.** Hoje `surface_plasticity()` só existe
para LoRA. Para dizer *"a literatura de CL não verifica pareamento de
plasticidade"*, o instrumento precisa funcionar em EWC/SI/MAS.

## Tarefa 2.1 — Pré-registrar a métrica ANTES de implementar

**Escreva `goals/protocol_plasticity_generalization.md` primeiro.** A escolha da
métrica primária precisa estar congelada antes da primeira seed — a regra
inviolável do projeto, e aqui ela é especialmente necessária porque há três
candidatas defensáveis.

Declare explicitamente:

- **Primária:** razão média por parâmetro entre `Δ_native` (o update que o
  otimizador de fato aplicou) e `Δ_unpenalized` (o update sem o termo de
  penalidade). É a única que **reduz exatamente ao `E` da máscara** quando a
  modificação é um escalonamento diagonal — ou seja, a única que torna LoRA e
  EWC comensuráveis.
- **Diagnósticas:** razão de norma (`‖Δ_native‖/‖Δ_unpenalized‖`) e projeção
  direcional (`cos(Δ_native, Δ_unpenalized)`).
- **A ressalva honesta, no pré-registro e no paper:** para métodos de penalidade
  o update **não** é escalonamento diagonal — ele pode mudar de direção. Logo a
  generalização **não é única**, e é por isso que a primária é declarada antes,
  não escolhida depois de ver os números.

## Tarefa 2.2 — TDD: `plasticity_ratio` para wrappers de otimizador

**RED** — `tests/test_plasticity_generalization.py`:

```python
"""Plasticidade efetiva para métodos de penalidade.

O lema que torna LoRA e EWC comensuráveis: quando a modificação é um
escalonamento diagonal do update, plasticity_ratio() reduz ao E da máscara.
Se este teste quebra, os dois hosts não estão na mesma escala e a tabela
comparativa do paper é inválida.
"""

from __future__ import annotations

import pytest
import torch


def test_ratio_is_one_without_penalty() -> None:
    from dual_heater.plasticity import plasticity_ratio

    delta = {"w": torch.tensor([1.0, -2.0, 3.0])}
    assert plasticity_ratio(native=delta, unpenalized=delta) == pytest.approx(1.0)


def test_ratio_reduces_to_the_mask_mean_for_diagonal_scaling() -> None:
    """O lema. Uma máscara [1, 0.5, 0] tem E = 0.5; o ratio deve dar 0.5."""
    from dual_heater.plasticity import plasticity_ratio

    unpenalized = {"w": torch.tensor([2.0, 2.0, 2.0])}
    mask = torch.tensor([1.0, 0.5, 0.0])
    native = {"w": unpenalized["w"] * mask}

    assert plasticity_ratio(native=native, unpenalized=unpenalized) == pytest.approx(0.5)


def test_ratio_ignores_parameters_with_zero_unpenalized_update() -> None:
    """Divisão por zero não é plasticidade zero: é ausência de evidência."""
    from dual_heater.plasticity import plasticity_ratio

    native = {"w": torch.tensor([1.0, 0.0])}
    unpenalized = {"w": torch.tensor([2.0, 0.0])}
    # Só o primeiro elemento é informativo: 1/2 = 0.5
    assert plasticity_ratio(native=native, unpenalized=unpenalized) == pytest.approx(0.5)


def test_frozen_parameters_count_as_zero_plasticity() -> None:
    """O erro que originou este trabalho: congelado é E=0, não 'fora da média'."""
    from dual_heater.plasticity import plasticity_ratio

    native = {"a": torch.zeros(4), "b": torch.ones(4)}
    unpenalized = {"a": torch.ones(4), "b": torch.ones(4)}
    assert plasticity_ratio(native=native, unpenalized=unpenalized) == pytest.approx(0.5)
```

**Rodar, confirmar 4 failed, implementar `src/dual_heater/plasticity.py`, rodar,
confirmar 4 passed, commitar.**

## Tarefa 2.3 — Teste de mutação da cadeia

O padrão que já pegou dois bugs reais neste repo. Para cada mutação, o baseline
verde primeiro, depois a mutação, e confirme que **algum** teste fica vermelho:

| # | Mutação | Deve matar |
|---|---|---|
| M1 | `plasticity_ratio` ignora parâmetros com update nativo zero | `test_frozen_parameters_count_as_zero_plasticity` |
| M2 | denominador vira `native` em vez de `unpenalized` | `test_ratio_reduces_to_the_mask_mean_...` |
| M3 | strength da penalidade não chega na loss | `test_ewc_reduces_drift_versus_vanilla...` |
| M4 | a média é sobre tensores em vez de por parâmetro | `test_ratio_reduces_to_the_mask_mean_...` |

**Se alguma mutação sobreviver, o teste que falta é mais importante que o
próximo experimento.** Foi assim que apareceram `build_run_config` e
`plasticity_record` na sessão anterior.

## Tarefa 2.4 — Piloto MLP/CNN pareado

Com o instrumento pronto, rode o piloto de re-avaliação: EWC/SI/MAS em
Split-MNIST e Split-CIFAR100, medindo `plasticity_ratio` de cada método e
comparando cada um contra um `lr_control` pareado naquele ratio.

**Pré-registre antes de rodar.** Custo estimado ~4h numa GPU; cabe numa noite
nas 3 locais. **Peça autorização explícita antes de disparar** (regra do
projeto). Lance destacado com `setsid nohup ... & disown`.

**A pergunta que o piloto responde, e que é o título do ICML:** quantos dos
métodos publicados de CL, quando comparados contra um controle pareado em
plasticidade efetiva em vez de contra vanilla, ainda mostram efeito?

---

# FASE 3 — Segundo host para `|z|` `[IJCNN]`

**O limite declarado de R-A/R-B:** um host (BERT-mini), um benchmark
(Split-CLINC150), duas tarefas por sequência. Para um paper de 6 páginas isso é
fino demais.

## Tarefa 3.1 — Replicar a ablação de critério num segundo host

Reuse a máquina que já existe — `scripts/run_criterion_ablation.py` e
`criterion_ablation_conditions()` em `experiments/bert_slowheat_diagnostic.py:80`
já implementam os 4 braços com capacidade pareada (2.364.672 entradas cada).

Duas opções, em ordem de preferência:

1. **Sequência mais longa no mesmo host:** `task_limit=2` está fixo em
   `experiments/bert_slowheat_diagnostic.py:347`. Subir para 5 ou 10 tarefas
   ataca o limite mais citável ("duas tarefas por sequência") com o menor risco
   de implementação.
2. **Host novo:** Qwen com trackers FFN/atenção (não LoRA — R-C mostrou que a
   geometria do LoRA vaza proteção por unidade).

**Pré-registro obrigatório**, banda de seeds nova, disjunta de 4.000.003+.

## Tarefa 3.2 — O falsificador de variância, de novo

Repita a medição de variância do ranking normalizado (§F do protocolo original).
No BERT deu 2,9124e-03 vs. 2,9070e-03 — diferença de 0,2%, que é o que prova que
o empate `|z|` vs. `|z·dL/dz|` é real e não artefato de ranking degenerado.

**Sem este número, o empate não é reportável.** Um reviewer vai perguntar
exatamente isso.

---

# FASE 4 — Escrita do IJCNN (6 páginas, IEEE) `[deadline 31/01]`

**Título de trabalho:** *"Activation Magnitude Is Enough: Gradient-Free Neuron
Selection for Continual Learning in Transformers"*

**Claim:** ranquear unidades por `|z|` puro iguala `|z·dL/dz|` sob capacidade
pareada, e ambos batem máscara aleatória por +12,23 pp (10/10). O componente do
gradiente — a parte cara, que exige backward hook — não compra nada.

## Estrutura (6 páginas é apertado; corte cedo)

| Seção | Páginas | Conteúdo | Fonte |
|---|---|---|---|
| 1. Introduction | 0,75 | Proteção seletiva funciona; qual critério? | — |
| 2. Related Work | 0,5 | AWARe (2608.11758) ranqueia por saliência de ativação — **nosso braço `magnitude` é o critério deles** | `criterion_ablation_results.md` §"Relação com o AWARe" |
| 3. Method | 1,0 | Os 4 braços, o pareamento de capacidade | `bert_slowheat_diagnostic.py:80` |
| 4. Experiments | 2,0 | R-A (+12,23 pp), R-B (empate), variância do ranking, Fase 3 | `criterion_ablation_results.md` |
| 5. Replay selection | 0,75 | R-D — resultado independente, unânime 20/20 | `replay_selector_results.md` |
| 6. Limitations | 0,5 | Escopo honesto | — |
| Refs | 0,5 | — | — |

## Tarefa 4.1 — Esqueleto

Crie `article/ijcnn/manuscript.md` com os títulos de seção acima e, **em cada
um, o caminho do arquivo de resultado que o sustenta**. Nenhuma seção pode
existir sem fonte apontada.

## Tarefa 4.2 — Tabelas antes da prosa

Gere as tabelas a partir dos JSON de `results/criterion_ablation/`, não copiando
números do markdown à mão. Script: `scripts/build_ijcnn_tables.py`, saída em
`article/ijcnn/tables/`.

**Verificação (o teste que impede o erro mais comum em escrita de paper):**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp && PYTHONPATH=. .venv/bin/python scripts/build_ijcnn_tables.py --check
```

**Saída esperada:** `OK: 4 tabelas, todos os valores batem com os manifests`.

## Tarefa 4.3 — Escrever, seção por seção

Uma seção por sessão de trabalho, commit a cada. **Ordem: 3 → 4 → 5 → 2 → 6 → 1.**
Introdução por último, quando já se sabe o que o paper diz.

## Tarefa 4.4 — Converter para IEEE LaTeX

6 páginas no template IEEE, via CMT. Deixe 3 dias de folga para o estouro de
páginas, que é certo.

---

# FASE 5 — Escrita do ICML (8 páginas) `[deadline ~22/01, confirmar]`

**Título de trabalho:** *"Plasticity-Matched Controls: Re-evaluating Continual
Learning Methods Against What They Actually Constrain"*

**Claim:** comparações em CL raramente controlam quanta plasticidade o método
removeu. Damos um instrumento que mede isso, mostramos que ele muda conclusões —
inclusive as nossas, pré-registradas e já confirmadas — e re-avaliamos métodos
publicados sob pareamento correto.

## Estrutura

| Seção | Páginas | Conteúdo |
|---|---|---|
| 1. Introduction | 1,0 | O confundimento de plasticidade |
| 2. Effective plasticity | 1,5 | Definição, o lema `E_superfície ≤ E_bindings`, generalização para penalidade (Fase 2) |
| 3. Case study: our own result | 1,5 | R-E + o resultado da Fase 0. **O caso central.** |
| 4. Re-evaluation | 2,0 | Piloto EWC/SI/MAS da Fase 2.4 |
| 5. Negative results | 1,0 | R-C (LoRA-FA), R-F (não-replicação). R-A/R-B só como citação ao companion |
| 6. Limitations | 0,5 | Não-unicidade da generalização, hosts, benchmarks |
| 7. Related Work | 0,5 | — |

## Tarefa 5.1 — A seção 3 é o coração; escreva-a primeiro

A narrativa: um resultado confirmatório pré-registrado, com Holm, sinal exato e
cinco gates, descreveu condições experimentais que não existiam — e nenhum dos
gates podia detectar isso, porque todos verificavam a métrica, e era a métrica
que estava errada.

**Escreva isso sem suavizar.** O valor do paper está em ser o próprio caso, e um
paper de protocolo que esconde o próprio erro não tem nada a dizer.

## Tarefa 5.2 — Canibalizar o manuscrito antigo

`article/manuscript.md` tem material reaproveitável. As seções 3 (*Why Raw
Gradient Scaling Is Not AdamW Update Scaling*, linhas 122–159) e 4.4
(*Evaluation-mode restoration correction*, 210–234) são argumentos de
instrumentação e encaixam direto no eixo novo.

**Ao terminar:** `git mv article/manuscript.md article/_archive/manuscript_method_axis_2026-09.md`
com um cabeçalho explicando por que foi substituído. **Não apague** — é registro
de como a âncora mudou por evidência.

## Tarefa 5.3 — Upside opcional: 4× A6000

**Só se o acesso confirmar e a Fase 2.4 já estiver fechada.** Aplicar o protocolo
corrigido a O-LoRA / InfLoRA / N-LoRA no benchmark de 15 tarefas que esses
próprios papers usam, com LLaMA-7B.

Isso eleva o ICML de "re-avaliação em MLP/CNN" para "re-avaliação no benchmark
padrão da área". **É upside, não caminho crítico** — não deixe a submissão
depender disso.

---

# FASE 6 — Artefato e checklist

## Tarefa 6.1 — Suíte completa e limpa

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp && CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python -m pytest -q
```

**Saída esperada:** `N passed, 1 skipped` com N ≥ 780, 0 failed. O skip é
`tests/test_capacity_calibration.py:800` (precisa de CUDA), pré-existente.

## Tarefa 6.2 — Reprodutibilidade

Para cada tabela dos dois papers, um comando que a regenera a partir dos
artefatos versionados. Escreva em `docs/reproducibility.md`, uma linha por
tabela.

## Tarefa 6.3 — Checklist de submissão

Preencha o formulário do ICML com honestidade sobre os resultados negativos.
**Eles são o conteúdo, não uma fraqueza a minimizar.**

---

## Tests / validation — a regra que vale para todas as fases

1. **TDD por tarefa de código:** escreva o teste, rode e **veja falhar**,
   implemente o mínimo, rode e veja passar, commit. Um teste que nunca foi
   vermelho não provou nada.
2. **Teste de mutação nos caminhos críticos.** Todo parâmetro que atravessa
   CLI → config → runner → artefato precisa de um teste que quebre se o salto
   for removido. Este repo já teve dois bugs desse tipo exato
   (`importance_criterion` não propagado; `learning_rate_scale` não chegando ao
   `lr_control`), e o segundo só morreu depois de extrair `build_run_config`
   como seam testável.
3. **Verificações de integridade antes de endpoint.** Sempre. As 8 checagens da
   Tarefa 0.2 são o modelo: se o pareamento não bate, não olhe o resultado.
4. **Nunca `pytest ... | tail`.** Use `> log 2>&1; echo EXIT=$?`. Um pytest que
   morre no parsing de argumento retorna 4 e um pipe esconde isso — aconteceu
   nesta sessão com `--timeout=600` sem `pytest-timeout` instalado.
5. **Runs longas destacadas:** `setsid nohup ... & disown`. Uma run foi morta por
   SIGTERM aos 34 min em 28/09 por não estar destacada.

---

## Riscos, tradeoffs e questões em aberto

### Riscos

| # | Risco | Mitigação |
|---|---|---|
| R1 | **Deadline do ICML incerto** (agregadores discordam, CFP não publicado) | Tarefa 0.1 semanal. Planejar para a data **mais cedo** (22/01) |
| R2 | **Baselines é o maior bloco de código** e nenhum existe nos hosts principais | Fase 1 começa na semana 1, em paralelo com a Fase 0 |
| R3 | **Dois papers, um autor.** 16 semanas é apertado | O IJCNN reusa evidência pronta; se apertar, **corte a Fase 5.3, nunca a Fase 1** |
| R4 | Sobreposição ICML/IJCNN vista como submissão dupla | Regra de não-sobreposição explícita; citar como companion em ambos |
| R5 | A generalização da plasticidade **não é única** para métodos de penalidade | Declarar a primária antes da primeira seed; reportar as diagnósticas juntas |
| R6 | A Fase 2.4 pode mostrar que **nenhum** método sobrevive ao pareamento — resultado extremo demais para acreditar | Verificar o instrumento contra um caso onde a resposta é conhecida (escalonamento diagonal reduz ao `E` da máscara — teste 2.2) antes de acreditar num varrido |

### Tradeoffs assumidos

- **ICML = protocolo, IJCNN = empírico.** A alternativa (ICML com o método,
  IJCNN com o protocolo) foi descartada: a evidência não sustenta um paper de
  método — três resultados independentes mostram que a camada da casa não
  adiciona nada.
- **Baselines antes de experimento novo.** Um experimento a mais não salva uma
  submissão sem EWC/SI/MAS; a falta deles rejeita sozinha.
- **6 páginas do IJCNN forçam cortes.** R-D (replay) pode virar meia página ou
  sair. Se sair, vira um terceiro paper ou um workshop.

### Questões em aberto — precisam de decisão do Fachel

1. **As 4× A6000 estão confirmadas?** Muda a Fase 5.3 de opcional para caminho
   crítico, e muda o alcance do ICML.
2. **Co-autores?** Afeta quanto da Fase 1 pode ser paralelizado.
3. **Se a Fase 2.4 mostrar que a maioria dos métodos publicados não sobrevive ao
   pareamento — publicar o varrido completo ou restringir a 2-3 métodos com
   análise mais profunda?** Tem consequência de tom: o primeiro é uma acusação
   ampla, o segundo é um argumento mais defensável.
4. **Existe plano B se o ICML for rejeitado?** (ICLR 2028, NeurIPS 2027.)
   Vale decidir agora, não em abril.

---

## Apêndice: mapa de arquivos

### Já existem, verificados

```
docs/results/criterion_ablation_results.md      R-A, R-B  [IJCNN]
docs/results/replay_selector_results.md         R-D       [IJCNN]
docs/results/exact_decomposition_results.md     R-C       [ICML]
docs/lora/lora_confirmation_results.md §R3      R-E       [ICML]
goals/protocol_plasticity_matched.md            pré-registro da run em curso
src/dual_heater/lora_slowheat.py                surface_plasticity(), mask_coverage()
src/dual_heater/slow_heat.py:98                 IMPORTANCE_CRITERIA = ("functional", "magnitude")
experiments/bert_slowheat_diagnostic.py:80      criterion_ablation_conditions()
experiments/confirmatory_statistics.py          exact_two_sided_sign_test()
experiments/analyze_confirmation.py:107         holm()
scripts/run_criterion_ablation.py               runner reaproveitável na Fase 3
article/manuscript.md                           a canibalizar (Tarefa 5.2)
```

### A criar

```
goals/venue_deadlines.md                        Tarefa 0.1
scripts/verify_plasticity_matched.py            Tarefa 0.2
scripts/analyze_plasticity_matched.py           Tarefa 0.3
docs/results/plasticity_matched_results.md      Tarefa 0.3
docs/audits/baseline_inventory.md               Tarefa 1.1
src/dual_heater/ewc.py                          Tarefa 1.2
tests/test_bert_baselines.py                    Tarefa 1.2
goals/protocol_plasticity_generalization.md     Tarefa 2.1
src/dual_heater/plasticity.py                   Tarefa 2.2
tests/test_plasticity_generalization.py         Tarefa 2.2
article/ijcnn/manuscript.md                     Fase 4
scripts/build_ijcnn_tables.py                   Tarefa 4.2
article/icml/manuscript.md                      Fase 5
docs/reproducibility.md                         Tarefa 6.2
```
