# Cinco tarefas do caminho crítico do artigo — plano de implementação

> **Para o Hermes:** use a skill `subagent-driven-development` para executar tarefa por tarefa.

**Goal:** Executar os cinco itens que destravam a submissão — levantamento de literatura (L), o experimento de capacidade QB-2, o registro de integridade QA-6, a análise C-3 das 50 seeds de CNN, e o pré-registro M1 do DER++ — sem quebrar nenhum pré-registro congelado.

**Architecture:** Quatro das cinco tarefas são texto, análise de dados já em disco, ou pré-registro; só QB-2 consome GPU. A ordem é forçada por uma dependência real: **L bloqueia tudo** (se o protocolo de pareamento já existir na literatura, a âncora do artigo cai e QB-2 muda de justificativa), e dentro de QB-2 existe um bug de escala que precisa ser corrigido por TDD **antes** de qualquer GPU.

**Tech Stack:** Python 3.12 no venv do projeto (`.venv/bin/python`, numpy 2.5.2, torch 2.7.1+cu126), pytest, 3 GPUs Pascal (2× GTX 1080 Ti 11 GB + 1× Titan Xp 12 GB), Qwen2.5-0.5B via HuggingFace.

---

## Contexto e pressupostos

**Diretório de trabalho:** `/mnt/B-SSD/fachel/dual-heater-mlp`. Todos os caminhos abaixo são relativos a ele.

**Interpretador:** use **sempre** `.venv/bin/python`. O `python3` do sistema **não** tem numpy e vai falhar com `ModuleNotFoundError: No module named 'numpy'`.

**Estado do Git:** branch `main`, commit `475d3c6`, árvore **suja** com 6 arquivos modificados e 7 não rastreados (documentos das sessões anteriores). A Tarefa 0 commita isso antes de qualquer trabalho novo.

**Estado dos dados:**
- `results/qwen_lora_confirmation/` — 10/10 seeds, confirmação **fechada e confirmada** (`p = 0,02148`). **Não tocar, não re-rodar, não acrescentar seeds.**
- `results/cache_derpp_10seeds/` — 50 seeds × 5 caches × 5 datasets. Os checkpoints `.pt` foram removidos (100 GB → 959 MB); os agregados JSON/CSV estão intactos e verificados.
- `results/qwen_iso_plasticity/confirm120_seed{10..19}/` — 10 seeds, 120 passos, `protocol_hash = 2cd785d8…`.

**GPUs:** livres (`nvidia-smi` mostra ~7 MiB usados, 0% de utilização).

**Regra inviolável do projeto:** nenhum pré-registro congelado é editado depois de ver acurácia. Alterações exigem linha nova na tabela K **antes** da run correspondente. Um registro *a posteriori* é permitido apenas se declarar explicitamente que é posterior.

---

## Ordem de execução e dependências

```
Tarefa 0 (commit do estado atual)
    │
    ├── Tarefa 1-4    L: levantamento de literatura        [BLOQUEIA 5-9]
    │        │
    │        └── Tarefa 5-9   QB-2: rank pareado           [GPU, ~2h]
    │
    ├── Tarefa 10-11  QA-6: registro de integridade        [independente]
    ├── Tarefa 12-14  C-3: análise das 50 seeds            [independente]
    └── Tarefa 15-17  M1: pré-registro do DER++            [independente]
```

Tarefas 10 a 17 **não dependem de nada** e podem rodar em paralelo com 1 a 9.

---

## Tarefa 0: Commitar o estado atual antes de começar

**Objective:** Partir de uma árvore limpa, para que cada tarefa abaixo produza um diff legível.

**Files:** nenhum arquivo novo; apenas commit do que já existe.

**Step 1: Ver o que está pendente**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp && git status --short
```

Expected: 6 arquivos `M` e 7 `??`, entre eles `goals/proximo_passo_{artigo,mlp,cnn,bert,qwen}.md` e `docs/lora/lora_confirmation_results.md`.

**Step 2: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add goals/ docs/
git commit -m "docs: separa proximos passos por arquitetura e registra a confirmacao do LoRA"
```

Expected: `13 files changed` (aproximado).

**Step 3: Verificar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp && git status --short
```

Expected: saída vazia.

---

# L — Levantamento de literatura (BLOQUEADOR, sem GPU)

> **Por que bloqueia tudo:** a âncora recomendada do artigo é *"propomos um protocolo que separa quantidade de distribuição de plasticidade"*. Se alguém já publicou pareamento por plasticidade efetiva medida, a contribuição principal deixa de existir e QB-2 muda de justificativa (de "defender o resultado central" para "saneamento de um resultado secundário"). Nenhuma redação começa antes disto.

## Tarefa 1: Criar o documento de levantamento com as perguntas declaradas antes da busca

**Objective:** Declarar o que conta como "já existe" **antes** de buscar, para não racionalizar o resultado depois.

**Files:**
- Create: `docs/related_work/protocol_prior_art.md`

**Step 1: Criar o diretório e o arquivo**

Crie `docs/related_work/protocol_prior_art.md` com exatamente este conteúdo:

```markdown
# Levantamento de prioridade — protocolo de pareamento por plasticidade medida

> Documento de busca. As perguntas e os critérios de decisão abaixo foram
> escritos **antes** da primeira busca, pela mesma razão que um pré-registro
> existe: impedir que o resultado da busca seja reinterpretado para preservar a
> contribuição desejada.

**Estado:** em execução.
**Bloqueia:** toda a redação do artigo, e a justificativa de QB-2.

---

## A pergunta que decide a âncora

**Existe trabalho publicado que pareia braços experimentais por plasticidade
efetiva medida (não nominal) ao comparar métodos de proteção seletiva?**

"Plasticidade efetiva medida" significa: uma quantidade calculada a partir do
passo de otimização realmente aplicado, usada para igualar braços antes de
comparar acurácia — e não um hiperparâmetro declarado (`beta`, `lambda`,
`budget`) assumido como equivalente entre braços.

## Critérios de decisão, declarados antes da busca

| Achado | Consequência para o artigo |
|---|---|
| **Nada parecido** | Âncora de protocolo se mantém. Redação segue o plano. |
| **Existe pareamento por capacidade/parâmetros, mas não por plasticidade do passo** | Âncora se mantém, com posicionamento explícito contra esse trabalho na §9. |
| **Existe pareamento por plasticidade medida, em outro domínio** (ex.: poda, quantização) | Âncora enfraquece. A contribuição vira "transporte + validação em CL", e a §1 precisa dizer isso. |
| **Existe pareamento por plasticidade medida em continual learning** | **Âncora cai.** `goals/proximo_passo_artigo.md` §3 precisa ser reescrito antes de qualquer redação. |

## Alvos mínimos de busca

### Prioridade 1 — o protocolo (existencial)

Termos, porque a literatura de CL raramente usa a palavra "plasticidade":

- `iso-plasticity` / `isoplasticity`
- `matched plasticity` / `plasticity-matched`
- `effective step size matching` / `effective learning rate control`
- `capacity-controlled ablation` continual learning
- `gradient mask budget` continual learning
- `equal-capacity control` catastrophic forgetting
- `learning rate control` ablation "selective protection"
- `effective plasticity` neural network continual

### Prioridade 2 — a família LoRA-CL (afeta só os resultados negativos)

- **O-LoRA** — ortogonalização de subespaço por tarefa; colide com o mecanismo `rank`
- **InfLoRA** — subespaço livre de interferência; colide com `rank` e `slice`
- **CorDA**, **MoRAL**, **SAPT** e o que a busca revelar

### Prioridade 3 — vizinhos conceituais já citados na §9 do manuscrito

EWC, SI, SLNID, HAT, UCB, Neuron Activation Importance. Verificar se **algum
deles** reporta um controle de learning rate pareado. Se reportarem, isso muda
a força da afirmação "o controle é rotineiramente omitido".

---

## Resultados

_(preencher durante a busca — uma linha por trabalho relevante)_

| Trabalho | Ano | Venue | O que faz | Parea por plasticidade medida? | Colide com |
|---|---|---|---|---|---|

## Veredito

_(preencher ao fim, escolhendo uma das quatro linhas da tabela de critérios)_
```

**Step 2: Verificar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp && test -f docs/related_work/protocol_prior_art.md && wc -l docs/related_work/protocol_prior_art.md
```

Expected: um número em torno de 80.

**Step 3: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add docs/related_work/protocol_prior_art.md
git commit -m "docs: declara criterios do levantamento de prioridade antes da busca"
```

## Tarefa 2: Executar a busca de prioridade 1 (o protocolo)

**Objective:** Responder à pergunta existencial.

**Files:**
- Modify: `docs/related_work/protocol_prior_art.md` (seção "Resultados")

**Step 1: Buscar**

Use a ferramenta `web_search` com **cada** um dos 8 termos da seção "Prioridade 1", um por chamada, `limit=10`. Para cada resultado promissor, use `web_extract` no abstract.

Fontes a priorizar: arXiv, OpenReview, Proceedings of NeurIPS/ICML/ICLR/CVPR, TMLR.

**Step 2: Registrar cada achado relevante**

Para cada trabalho que passe do título, acrescente uma linha na tabela "Resultados" de `docs/related_work/protocol_prior_art.md`. Um trabalho é relevante se comparar métodos de regularização/proteção **e** mencionar qualquer forma de controle de capacidade ou de learning rate.

**Regra honesta:** registre também os trabalhos que **quase** batem. "Ninguém faz isso" é uma afirmação forte e precisa mostrar o que existe de mais próximo.

**Step 3: Verificar cobertura**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp && grep -c '^|' docs/related_work/protocol_prior_art.md
```

Expected: pelo menos 15 linhas de tabela no total do arquivo (as 4 de critérios + cabeçalhos + os achados).

**Step 4: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add docs/related_work/protocol_prior_art.md
git commit -m "docs: resultados da busca de prioridade sobre pareamento por plasticidade"
```

## Tarefa 3: Executar a busca de prioridade 2 (família LoRA-CL)

**Objective:** Fechar a related work dos três mecanismos negativos.

**Files:**
- Modify: `docs/related_work/protocol_prior_art.md`

**Step 1: Buscar** O-LoRA, InfLoRA, CorDA, MoRAL, SAPT — um `web_search` por nome, mais um genérico: `"LoRA" continual learning catastrophic forgetting subspace 2024 2025`.

**Step 2: Para cada um, registrar** na tabela: o que faz, e **qual dos três mecanismos falhados ele toca** (`rank`, `leak`, `slice`).

Isto importa porque os três mecanismos são reportados como resultado negativo. Se O-LoRA já fez `rank` e funcionou, a leitura muda de "o mecanismo não funciona" para "nossa implementação do mecanismo não funciona", que é uma afirmação diferente e mais fraca.

**Step 3: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add docs/related_work/protocol_prior_art.md
git commit -m "docs: levantamento da familia LoRA-CL"
```

## Tarefa 4: Escrever o veredito e propagar a consequência

**Objective:** Fechar o bloqueador com uma decisão explícita.

**Files:**
- Modify: `docs/related_work/protocol_prior_art.md` (seção "Veredito")
- Modify: `goals/proximo_passo_artigo.md` (§4) — **apenas** se o veredito for a quarta linha da tabela

**Step 1: Escrever o veredito**

Na seção "Veredito", escolha **uma** das quatro linhas da tabela de critérios e justifique em 3-5 frases, citando os trabalhos encontrados.

**Step 2: Propagar**

- Vereditos 1 ou 2: em `goals/proximo_passo_artigo.md`, na §4, substitua o texto do bloqueador por `**RESOLVIDO em <data>:** ver docs/related_work/protocol_prior_art.md`.
- Veredito 3: idem, mais uma nota de que a §3 precisa de ajuste de posicionamento.
- **Veredito 4: PARE.** Não siga para QB-2. Reporte ao Fachel que a âncora caiu e que a §3 precisa ser reescrita. Esta é uma decisão humana.

**Step 3: Verificar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp && grep -A3 '## Veredito' docs/related_work/protocol_prior_art.md
```

Expected: texto preenchido, não o placeholder `_(preencher ao fim...)_`.

**Step 4: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add docs/related_work/protocol_prior_art.md goals/proximo_passo_artigo.md
git commit -m "docs: veredito do levantamento de prioridade"
```

---

# QB-2 — `exact` com rank pareado (GPU, ~2h)

> **O que este experimento decide:** o braço `exact` treina 4.214.016 parâmetros contra 6.769.920 do `vanilla`, porque congela a matriz `A`. A leitura *"ele esquece menos porque aprendeu menos"* não está descartada, e o pareamento em `E` não a resolve — `E` mede plasticidade retida da máscara, não capacidade do adaptador. Se o efeito sobreviver com parâmetros pareados, a explicação por capacidade cai. Se desaparecer, o resultado confirmado era artefato **e isso precisa ser reportado**.

## ⚠ Bug de escala descoberto na inspeção — leia antes de tocar em GPU

Em `src/dual_heater/lora.py:102`:

```python
self.scaling = lora_alpha / r
```

E em `experiments/qwen_lora_slowheat.py:74`, `alpha` é um campo de `RunConfig` fixado em `16.0`, **sem flag de CLI**. O `--rank` existe; `--alpha` **não**.

Consequência: rodar `--rank 32` com o código atual muda `scaling` de `16/16 = 1,0` para `16/32 = 0,5`. O braço `r=32` treinaria com metade da escala efetiva do adaptador, introduzindo **um segundo confundimento** exatamente no experimento que existe para eliminar um confundimento. O resultado seria ininterpretável.

**Correção obrigatória:** expor `--alpha` e passar `alpha = 32.0` quando `rank = 32`, mantendo `scaling = 1,0` nos dois braços. As Tarefas 5 a 7 fazem isso por TDD.

## Tarefa 5: Escrever o teste que falha para a flag `--alpha`

**Objective:** Provar que `alpha` não é configurável hoje.

**Files:**
- Modify: `tests/test_lora_slowheat.py`

**Step 1: Escrever o teste**

Acrescente ao fim de `tests/test_lora_slowheat.py`:

```python
def test_alpha_is_configurable_and_defaults_to_sixteen() -> None:
    """QB-2 needs alpha=32 at rank=32 so that scaling=alpha/r stays 1.0.

    Without this, comparing r=32 against r=16 silently halves the adapter's
    effective scale and introduces the very confound the experiment removes.
    """

    from experiments.qwen_lora_slowheat import build_parser

    parser = build_parser()

    default = parser.parse_args(["--output", "/tmp/x"])
    assert default.alpha == 16.0

    scaled = parser.parse_args(["--output", "/tmp/x", "--rank", "32", "--alpha", "32"])
    assert scaled.alpha == 32.0
    assert scaled.rank == 32
```

**Step 2: Rodar e verificar a falha**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -m pytest tests/test_lora_slowheat.py::test_alpha_is_configurable_and_defaults_to_sixteen -q
```

Expected: **FAIL** com `ImportError: cannot import name 'build_parser'` ou `AttributeError`. Se passar, pare: a premissa está errada e o plano precisa ser revisto.

## Tarefa 6: Extrair `build_parser` e acrescentar `--alpha`

**Objective:** Implementação mínima para o teste passar.

**Files:**
- Modify: `experiments/qwen_lora_slowheat.py` (função `main`, linha ~374)

**Step 1: Refatorar**

Em `experiments/qwen_lora_slowheat.py`, localize `def main()` (por volta da linha 374). Extraia a construção do parser para uma função nova **acima** de `main`, e acrescente o argumento `--alpha` logo após `--rank`:

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--eval-per-class", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs-per-task", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--slow-strength", type=float, default=3.0)
    parser.add_argument("--plasticity-budget", type=float, default=0.5)
    parser.add_argument("--hard", action="store_true")
    return parser
```

**Importante:** copie os argumentos restantes de `main` (`--target-plasticity`, `--leak-combination`, `--arms`, e quaisquer outros presentes nas linhas 388-395) para dentro de `build_parser`, **na mesma ordem**, antes do `return`. Em seguida, substitua no corpo de `main` a construção do parser por:

```python
    parser = build_parser()
```

**Step 2: Propagar `alpha` para o `RunConfig`**

Em `main`, na construção do `RunConfig`, acrescente `alpha=arguments.alpha` junto de `rank=arguments.rank`.

**Step 3: Rodar o teste**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -m pytest tests/test_lora_slowheat.py::test_alpha_is_configurable_and_defaults_to_sixteen -q
```

Expected: `1 passed`.

**Step 4: Rodar a suíte de LoRA inteira, para garantir que nada quebrou**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -m pytest tests/test_lora.py tests/test_lora_slowheat.py tests/test_lora_sweep_aggregate.py tests/test_lora_confirmation_guard.py -q
```

Expected: todos passam. `test_lora_confirmation_guard.py` sozinho dá `6 passed in ~21s`.

**Step 5: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add experiments/qwen_lora_slowheat.py tests/test_lora_slowheat.py
git commit -m "experiments: expoe --alpha para manter scaling=alpha/r constante entre ranks"
```

## Tarefa 7: Escrever o teste que garante escala constante entre os dois braços

**Objective:** Travar a invariante que o experimento inteiro depende.

**Files:**
- Modify: `tests/test_lora.py`

**Step 1: Escrever o teste**

Acrescente ao fim de `tests/test_lora.py`:

```python
def test_scaling_is_constant_when_alpha_tracks_rank() -> None:
    """r=16/alpha=16 and r=32/alpha=32 must produce the same adapter scale.

    QB-2 compares those two configurations. If scaling differs between them,
    the comparison measures scale, not capacity.
    """

    import torch.nn as nn

    from dual_heater.lora import LoRALinear

    base_small = nn.Linear(64, 64, bias=False)
    base_large = nn.Linear(64, 64, bias=False)

    small = LoRALinear(base_small, r=16, lora_alpha=16.0)
    large = LoRALinear(base_large, r=32, lora_alpha=32.0)

    assert small.scaling == 1.0
    assert large.scaling == 1.0
    assert small.scaling == large.scaling
```

**Step 2: Rodar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -m pytest tests/test_lora.py::test_scaling_is_constant_when_alpha_tracks_rank -q
```

Expected: `1 passed`. Este teste deve passar **sem** mudar o código de produção — ele documenta uma propriedade já verdadeira de `scaling = lora_alpha / r`.

Se falhar, leia a assinatura real de `LoRALinear` em `src/dual_heater/lora.py:66` e ajuste os nomes dos argumentos. Não mude o código de produção para fazer o teste passar.

**Step 3: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add tests/test_lora.py
git commit -m "tests: trava a invariante de escala entre rank 16 e 32"
```

## Tarefa 8: Escrever o pré-registro de QB-2

**Objective:** Congelar seeds, braços, endpoint e critério **antes** de ligar a GPU.

**Files:**
- Create: `goals/protocol_lora_rank_matched.md`

**Step 1: Criar o arquivo**

Crie `goals/protocol_lora_rank_matched.md` com este conteúdo:

```markdown
# Protocolo congelado — capacidade pareada do braço `exact`

> Pré-registro. Escrito **antes** de qualquer seed deste experimento. Não é
> editado depois de ver acurácia; alterações exigem commit anterior à run e
> linha nova na tabela K.

**Estado: CONGELADO em <DATA>**, antes da primeira seed.

---

## A. O que motiva

A confirmação de
[protocol_lora_confirmation.md](protocol_lora_confirmation.md) fechou em 28/09
com o endpoint primário **confirmado**: `exact − lr_control` em forgetting,
diferença −0,0921, `p = 0,02148 < 0,025`, mediana −0,0998, 9/10 seeds.

A seção J daquele protocolo declarou, antes da run, um confundimento que ele
não resolve:

> `exact` treina 4.214.016 parâmetros contra 6.769.920 (r=16), porque congela
> `A`. `E` mede plasticidade retida da máscara, não capacidade do adaptador,
> então a leitura *esquece menos porque aprendeu menos* permanece possível.

Este protocolo existe para decidir essa leitura.

## B. Pergunta

Com o **número de parâmetros treináveis pareado**, o braço `exact` ainda reduz
o esquecimento em relação ao `vanilla`?

## C. Hipótese declarada antes da run

*`exact` com `r = 32` reduz o forgetting médio em relação a `vanilla` com
`r = 16`, com a diferença pareada por seed negativa na maioria das seeds.*

Hipótese nula: as diferenças pareadas se distribuem simetricamente em torno de
zero. **Resultado nulo encerra a linha e deve ser reportado**: significaria que
o efeito confirmado em 28/09 é explicável por capacidade do adaptador.

## D. Decisões congeladas

| # | Decisão | Valor congelado | Justificativa |
|---|---|---|---|
| B1 | Braços | `exact_r32` (r=32, alpha=32, `A` congelada) e `vanilla_r16` (r=16, alpha=16) — exatamente dois | qualquer braço a mais infla a correção múltipla sem responder à pergunta |
| B2 | Seeds | banda nova `1_000_003, 1_025_011, 1_050_017, 1_075_037, 1_100_043, 1_125_059, 1_150_061, 1_175_087, 1_200_089, 1_225_097` | disjuntas das confirmatórias do LoRA (700001+), das do Split-MNIST (104729+), das exploratórias (múltiplos de 11) e das do iso-plasticidade |
| B3 | Endpoint primário | forgetting médio, contraste `exact_r32 − vanilla_r16` | mesmo endpoint da confirmação que este protocolo defende |
| B4 | Endpoints secundários | FAA (mesmo contraste); parâmetros treináveis por braço (verificação, não resultado) | reportados sempre, nunca promovidos |
| B5 | Teste | sinal exato bicaudal sobre as 10 diferenças pareadas por seed | mesmo teste da confirmação; sem suposição de normalidade em n=10 |
| B6 | Correção múltipla | Holm sobre a família de 2 (forgetting e FAA no contraste primário), α=0,05 | família declarada **antes** da run |
| B7 | Plasticidade alvo | `E = 0,85` para `exact_r32`, re-resolvida por bisseção em cada fronteira | mesmo valor da confirmação, para que os dois experimentos sejam comparáveis |
| B8 | Critério de sucesso | `p < 0,025` no endpoint primário **e** mediana negativa | idêntico ao A8 da confirmação |

## E. Cenário

Idêntico ao executado na confirmação de 28/09 — que **não** é idêntico ao
descrito na seção E daquele documento; ver a tabela K dele, linha de 28/09.

| Item | Valor |
|---|---|
| Modelo | `Qwen/Qwen2.5-0.5B`, `Qwen2ForSequenceClassification`, 150 rótulos |
| Precisão | fp32 (Pascal CC 6.x não tem bf16 nativo) |
| Alvos LoRA | `gate_proj`, `up_proj`, `down_proj` |
| **`exact_r32`** | **`r = 32`, `alpha = 32,0`** → `scaling = 1,0` |
| **`vanilla_r16`** | **`r = 16`, `alpha = 16,0`** → `scaling = 1,0` |
| Dataset | `clinc/clinc_oos:plus`, Class-IL por domínio, 10 tarefas |
| Dados por tarefa | 750 treino / 300 validação |
| Épocas | 3 | `batch_size` | 8 | `max_length` | 48 | `lr` | 1e-4 |
| Hardware | uma seed inteira por GPU, sem DDP |

## F. Por que `alpha` acompanha o rank

`src/dual_heater/lora.py:102` define `scaling = lora_alpha / r`. Manter
`alpha = 16` em `r = 32` daria `scaling = 0,5` contra `1,0` do braço de
referência, introduzindo uma diferença de escala efetiva **dentro do
experimento que existe para remover um confundimento**. Com `alpha = 32` em
`r = 32`, os dois braços têm `scaling = 1,0`.

Coberto por `tests/test_lora.py::test_scaling_is_constant_when_alpha_tracks_rank`.

## G. Verificação obrigatória antes da run

1. `tests/test_lora.py::test_scaling_is_constant_when_alpha_tracks_rank` passa;
2. `tests/test_lora_slowheat.py::test_alpha_is_configurable_and_defaults_to_sixteen` passa;
3. suíte de LoRA completa verde;
4. os parâmetros treináveis dos dois braços diferem em **menos de 5%** — esta é
   a premissa inteira do experimento e é verificada nos manifestos da run de
   fumaça (Tarefa 9), **antes** das 10 seeds;
5. `E_eff = 0,8500` (tol. 1e-3) para `exact_r32` em todas as fronteiras;
6. as seeds de B2 nunca foram executadas com nenhum braço de LoRA.

## H. Análise declarada

```text
para cada seed s: d_s = forgetting(exact_r32, s) − forgetting(vanilla_r16, s)
teste: sinal exato bicaudal sobre {d_s}
confirma se: p < 0,025 E mediana(d_s) < 0
```

Sem análise intermediária, sem parada antecipada, sem inspeção seed a seed
antes das 10 terminarem.

## I. O que este protocolo não resolve

- **Expressividade do subespaço.** Igualar parâmetros treináveis **não** iguala
  expressividade: `exact_r32` tem 32 direções aleatórias fixas em `A`,
  `vanilla_r16` tem 16 direções aprendíveis. Se `exact_r32` vencer, a
  explicação "mais direções, ainda que fixas, ajudam" permanece aberta.
  Declarado aqui, antes da run, para não ser descoberto depois.
- **Generalização.** Um modelo, um benchmark, um ponto de plasticidade.
- **Prioridade.** Depende do levantamento em `docs/related_work/protocol_prior_art.md`.

## K. Registro de alterações

| Data | Alteração | Antes da run? |
|---|---|---|
| <DATA> | criação e congelamento. B1 a B8 fechados. | sim — nenhuma seed executada |
```

**Step 2: Substituir `<DATA>`**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
sed -i "s/<DATA>/$(date +%d\\/%m\\/%Y)/g" goals/protocol_lora_rank_matched.md
grep -c '<DATA>' goals/protocol_lora_rank_matched.md
```

Expected: `0`.

**Step 3: Commitar — este commit precisa preceder a run**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add goals/protocol_lora_rank_matched.md
git commit -m "goals: pre-registro do experimento de capacidade pareada (QB-2)"
```

## Tarefa 9: Run de fumaça de uma seed, para verificar a premissa

**Objective:** Confirmar que os parâmetros treináveis ficam pareados **antes** de gastar 2h de GPU.

**Files:**
- Create (saída): `results/qwen_lora_rank_matched_smoke/`

**Step 1: Rodar 2 tarefas em uma seed fora da banda pré-registrada**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
HF_HOME=.hf-cache PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
.venv/bin/python -m experiments.qwen_lora_slowheat \
  --output results/qwen_lora_rank_matched_smoke/exact_r32 \
  --tasks 2 --seed 7 --device cuda:0 \
  --train-per-class 50 --eval-per-class 20 \
  --batch-size 8 --epochs-per-task 3 --max-length 48 \
  --rank 32 --alpha 32 --target-plasticity 0.85 \
  --arms exact
```

Expected: termina sem erro, escreve `manifest.json`. Alguns minutos.

**Step 2: Rodar o braço de referência**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
HF_HOME=.hf-cache PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
.venv/bin/python -m experiments.qwen_lora_slowheat \
  --output results/qwen_lora_rank_matched_smoke/vanilla_r16 \
  --tasks 2 --seed 7 --device cuda:0 \
  --train-per-class 50 --eval-per-class 20 \
  --batch-size 8 --epochs-per-task 3 --max-length 48 \
  --rank 16 --alpha 16 \
  --arms vanilla
```

**Step 3: Verificar o pareamento de parâmetros — o gate da seção G item 4**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -c "
import json
a=json.load(open('results/qwen_lora_rank_matched_smoke/exact_r32/manifest.json'))
b=json.load(open('results/qwen_lora_rank_matched_smoke/vanilla_r16/manifest.json'))
pa=[r['trainable_parameters'] for r in a['results'] if r['method']=='exact'][0]
pb=[r['trainable_parameters'] for r in b['results'] if r['method']=='vanilla'][0]
print(f'exact_r32  : {pa:,}')
print(f'vanilla_r16: {pb:,}')
print(f'razao      : {pa/pb:.4f}')
print('GATE:', 'PASSA' if abs(pa/pb-1) < 0.05 else 'FALHA — NAO RODAR AS 10 SEEDS')
"
```

Expected: `GATE: PASSA`.

**Se o gate falhar:** pare e reporte. A premissa do experimento está errada e o pré-registro precisa de uma linha na tabela K **antes** de qualquer ajuste. Não ajuste o rank para "fazer bater" sem registrar.

**Step 4: Commitar a verificação**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add goals/protocol_lora_rank_matched.md
git commit -m "goals: registra a verificacao de pareamento de parametros (gate G4)" --allow-empty
```

> **Nota para o implementador:** as 10 seeds em si exigem **autorização explícita do Fachel** antes de rodar (regra permanente do projeto para GPU). Este plano vai até o gate. Depois do `GATE: PASSA`, pergunte antes de lançar.

---

# QA-6 — Registro de integridade do braço `hard` (texto, sem GPU)

> **O que a inspeção encontrou, e por que é mais sério do que o documento atual diz.**
>
> `goals/protocol_iso_plasticity.md` **já tem** um desvio registrado (linhas 265-276): o commit `095b1c8` acrescentou `hard_b0.75` e `hard_b0.5` ao runner sem linha prévia na tabela K. O documento encerra dizendo:
>
> > "As runs afetadas usam 30 passos e já estão obsoletas por outro motivo, o que limita o dano a este desvio."
>
> **Isso é falso.** Verifiquei os manifestos `confirm120_seed{10..19}` (120 passos, `protocol_hash = 2cd785d8…`): as duas famílias contêm **8 braços**, e `hard_b0.75` e `hard_b0.5` estão entre eles. E em `goals/resultados_confirmacao.md`, as **duas únicas comparações que sobrevivem a Holm** são `hard − vanilla`.
>
> Ou seja: o único resultado significativo da parte A do Qwen vem de um braço não pré-registrado.

## Tarefa 10: Corrigir a nota de desvio no protocolo iso-plasticidade

**Objective:** Registrar que o desvio alcança a run de 120 passos.

**Files:**
- Modify: `goals/protocol_iso_plasticity.md` (linhas 265-276)

**Step 1: Verificar o fato antes de escrever**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -c "
import json,glob
fs=sorted(glob.glob('results/qwen_iso_plasticity/confirm120_seed*/manifest.json'))
m=json.load(open(fs[0]))
print('manifestos 120 passos:',len(fs))
print('steps_per_task:',m['protocol']['steps_per_task'])
for fam in ('0.75','0.5'):
    names=[a['arm']['name'] for a in m['families'][fam]['arms']]
    print(f'familia {fam}: n={len(names)}', names)
"
```

Expected: 10 manifestos, `steps_per_task: 120`, e `hard_b0.75` / `hard_b0.5` presentes nas listas de 8 nomes.

**Step 2: Substituir o parágrafo final**

Em `goals/protocol_iso_plasticity.md`, substitua o parágrafo que começa com `Consequência: o braço `hard` não é um braço pré-registrado.` por:

```markdown
Consequência: o braço `hard` não é um braço pré-registrado. Ele pode ser
reportado como exploratório, nunca como parte do contraste congelado.

### Correção desta nota (28/09/2026)

A frase anterior desta seção afirmava que "as runs afetadas usam 30 passos e já
estão obsoletas por outro motivo, o que limita o dano a este desvio".
**Isso está errado e a correção importa.**

Verificação nos manifestos `results/qwen_iso_plasticity/confirm120_seed{10..19}/`
(`protocol_hash = 2cd785d8…`, `steps_per_task = 120`): as duas famílias contêm
**8 braços**, incluindo `hard_b0.75` e `hard_b0.5`. O desvio alcança a run de
confirmação, não só as runs obsoletas de 30 passos.

Consequência real, e ela é séria: em
[resultados_confirmacao.md](resultados_confirmacao.md), as **duas únicas
comparações que sobrevivem a Holm** são `hard − vanilla` em retenção t0
(+0,075 em `E*=0,75` e +0,094 em `E*=0,50`, ambas 10/10 seeds,
`p_Holm = 0,047`). Ambas vêm de um braço não pré-registrado.

**O que isto obriga:**

1. As duas comparações `hard − vanilla` são **exploratórias**, não
   confirmatórias, e o artigo precisa dizer isso onde as reportar.
2. A família de Holm de 24 comparações **inclui** braços não declarados. O
   limiar corrigido foi calculado sobre uma família que o protocolo não
   declarou, o que enfraquece — não fortalece — o `p_Holm = 0,047`.
3. **Não** recalcular Holm sobre uma família reduzida agora. Escolher a família
   depois de ver o resultado é exatamente o que a correção existe para impedir.
   A saída honesta é reportar o que foi feito e a sua limitação.
4. A parte A do Qwen, na prática, **não tem resultado confirmatório**. A
   hipótese central (`iso − permutado`) já era nula; o que restava vinha de um
   braço fora do pré-registro.
```

**Step 3: Verificar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
grep -c 'Correção desta nota' goals/protocol_iso_plasticity.md
```

Expected: `1`.

**Step 4: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add goals/protocol_iso_plasticity.md
git commit -m "goals: corrige o alcance do desvio do braco hard, que atinge a run de 120 passos"
```

## Tarefa 11: Propagar a ressalva aos documentos que citam o resultado

**Objective:** Nenhum documento pode citar `hard − vanilla` como confirmatório.

**Files:**
- Modify: `goals/resultados_confirmacao.md`
- Modify: `docs/architectures/arch_qwen.md`

**Step 1: Em `goals/resultados_confirmacao.md`**, logo após a tabela "O que sobrevive à correção para múltiplos testes", insira:

```markdown
> **Ressalva de pré-registro (28/09/2026).** As duas comparações acima vêm do
> braço `hard`, que **não consta da seção E** de
> [protocol_iso_plasticity.md](protocol_iso_plasticity.md) e não recebeu linha
> na tabela K antes da run. São resultados **exploratórios**, não
> confirmatórios, e não devem ser reportados como parte do contraste congelado.
> Além disso, a família de Holm de 24 comparações inclui braços não declarados,
> o que enfraquece o `p_Holm = 0,047`. Ver a seção "Correção desta nota" do
> protocolo.
```

**Step 2: Em `docs/architectures/arch_qwen.md`**, na tabela "Por que nada disto é citável hoje", substitua a linha do desvio de pré-registro por:

```markdown
| **Desvio de pré-registro** | Braço `hard` presente nos manifestos sem constar da seção E nem ter linha na tabela K antes da run — **incluindo os manifestos `confirm120_*` de 120 passos**. As duas únicas comparações que sobrevivem a Holm vêm desse braço, logo são exploratórias |
```

**Step 3: Verificar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
grep -l 'Ressalva de pré-registro' goals/resultados_confirmacao.md && grep -c 'confirm120' docs/architectures/arch_qwen.md
```

Expected: o caminho do arquivo e um número ≥ 1.

**Step 4: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add goals/resultados_confirmacao.md docs/architectures/arch_qwen.md
git commit -m "docs: marca as comparacoes do braco hard como exploratorias"
```

---

# C-3 — Análise das 50 seeds de CNN (CPU)

> **Descoberta que muda o tamanho da tarefa:** o `proximo_passo_cnn.md` descreve C-3 como "analisar os agregados", sugerindo trabalho estatístico. Inspecionei `results/cache_derpp_10seeds/replay_selection_sweep/sweep_report.json` e **a análise já está computada**: 50 seeds, 6 learners, 4 seletores, 5 datasets, com `slowheat_vs_derpp`, `slowheat_vs_replay`, `paired_differences_vs_first`, `memory_vs_no_memory` e Holm já aplicado (`"Holm over loss, representative and hybrid final-accuracy contrasts within each dataset/backbone/learner family"`).
>
> C-3 é, portanto, uma tarefa de **leitura e redação**, não de computação. `status: exploratory_not_independent_confirmation` já está declarado no próprio artefato.

## Tarefa 12: Extrair os contrastes principais para uma tabela legível

**Objective:** Transformar o JSON em números citáveis.

**Files:**
- Create: `docs/results/replay_selection_50seed_analysis.md`

**Step 1: Rodar o extrator**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -c "
import json
d=json.load(open('results/cache_derpp_10seeds/replay_selection_sweep/sweep_report.json'))
print('seeds:',len(d['seeds']),'| learners:',len(d['learners']),'| selectors:',d['selectors'])
print('status:',d['status'])
print('primary_endpoint:',d['primary_endpoint'])
print()
for family in ('slowheat_vs_derpp','slowheat_vs_replay'):
    print('='*70); print(family)
    for ds in sorted(d[family]):
        for sel in sorted(d[family][ds]):
            e=d[family][ds][sel].get(d['primary_endpoint'])
            if not e: continue
            pv=e.get('paired_values',[])
            pos=sum(1 for v in pv if v['difference']>0)
            holm=e.get('holm_adjusted_p')
            mean=e.get('mean')
            print(f'  {ds:18s} {sel:15s} mean={mean:+.4f}  n={len(pv)}  {pos}+/{len(pv)-pos}-  holm={holm}')
" 2>&1 | tee /tmp/c3_extract.txt
```

Expected: 50 seeds, `status: exploratory_not_independent_confirmation`, e uma grade de linhas por dataset × seletor.

**Step 2: Criar o documento**

Crie `docs/results/replay_selection_50seed_analysis.md` com cabeçalho, e cole as tabelas geradas:

```markdown
# Sweep de seleção de replay — 50 seeds

**Fonte:** `results/cache_derpp_10seeds/replay_selection_sweep/sweep_report.json`
**Status declarado no artefato:** `exploratory_not_independent_confirmation`
**Escopo:** 50 seeds × 6 learners × 4 seletores × 5 datasets (4.500 runs de learner)
**Endpoint primário:** `final_average_accuracy`
**Multiplicidade:** Holm sobre os contrastes de loss/representative/hybrid dentro de cada dataset/backbone/família de learner

> **Este é o maior `n` do repositório** e nenhuma tabela publicada o usava.
>
> **Análise exploratória por construção:** os dados já existiam em disco antes
> desta pergunta ser feita. Não é confirmação e não deve ser apresentada como
> tal.
>
> **Nota de artefato (28/09/2026):** os 4.000 checkpoints `.pt` deste diretório
> foram removidos (98,78 GB). Os agregados JSON/CSV estão intactos e foram
> verificados após a remoção. Nenhuma análise de endpoint depende dos pesos.

## Contrastes

_(colar aqui a saída de /tmp/c3_extract.txt)_

## Leitura

_(preencher)_
```

**Step 3: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add docs/results/replay_selection_50seed_analysis.md
git commit -m "docs: extrai os contrastes do sweep de 50 seeds"
```

## Tarefa 13: Escrever a leitura, checando a inversão ER-ACE/DER++

**Objective:** Responder se as 50 seeds replicam a inversão de sinal das 10 seeds.

**Files:**
- Modify: `docs/results/replay_selection_50seed_analysis.md` (seção "Leitura")

**Step 1: A pergunta específica**

`docs/architectures/arch_cnn.md` (C1) reporta, com 10 seeds: SlowHeat acoplado a ER-ACE melhora (+4,15 e +1,27 p.p.), acoplado a DER++ piora (−1,13 e −0,53). As 50 seeds cobrem `slowheat_vs_derpp` e `slowheat_vs_replay`.

Escreva 3-5 parágrafos respondendo:

1. O sinal de `slowheat_vs_derpp` nas 50 seeds concorda com o das 10 seeds?
2. A magnitude cresce, encolhe, ou vira ruído com `n` maior?
3. Algum contraste sobrevive a Holm com 50 seeds que não sobrevivia com 10?

**Step 2: A regra editorial obrigatória**

Sempre que citar forgetting, cite a acurácia final ao lado. Na CNN, proteção hard reduz forgetting em 8,34 p.p. **e ainda perde** 2,97 p.p. de acurácia — citar só forgetting inverteria a conclusão.

**Step 3: Verificar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
grep -c 'preencher' docs/results/replay_selection_50seed_analysis.md
```

Expected: `0`.

**Step 4: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add docs/results/replay_selection_50seed_analysis.md
git commit -m "docs: leitura do sweep de 50 seeds e a inversao ER-ACE/DER++"
```

## Tarefa 14: Ligar o documento novo ao índice e ao host

**Objective:** Um artefato que ninguém encontra é um artefato órfão — o problema que a auditoria de 22/09 levantou.

**Files:**
- Modify: `docs/results/results_index.md`
- Modify: `docs/architectures/arch_cnn.md` (seção C4)
- Modify: `goals/proximo_passo_cnn.md` (item C-3)

**Step 1:** Em `docs/results/results_index.md`, na seção "Replay seletivo", acrescente após a tabela:

```markdown
Análise dos contrastes em
[`replay_selection_50seed_analysis.md`](replay_selection_50seed_analysis.md)
(28/09/2026). Até então, o maior `n` do repositório não era citado por nenhum
documento.
```

**Step 2:** Em `docs/architectures/arch_cnn.md`, na tabela C4, mude o estado da linha de `cache_derpp_10seeds` de `não analisado` para:

```markdown
| `cache_derpp_10seeds/replay_selection_sweep/split_cifar*` | 50 seeds × 5 caches | **analisado em 28/09**, ver [replay_selection_50seed_analysis.md](../results/replay_selection_50seed_analysis.md); checkpoints `.pt` removidos, agregados intactos |
```

**Step 3:** Em `goals/proximo_passo_cnn.md`, marque C-3 como concluído na tabela do §6.

**Step 4: Verificar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -c "
import re,os
for p in ['docs/results/results_index.md','docs/architectures/arch_cnn.md','goals/proximo_passo_cnn.md']:
    t=open(p,encoding='utf-8').read()
    for l in re.findall(r'\]\(([^)#]+)',t):
        if l.startswith('http'): continue
        tgt=os.path.normpath(os.path.join(os.path.dirname(p),l))
        if not os.path.exists(tgt): print('QUEBRADO:',p,'->',l)
print('checagem de links concluida')
"
```

Expected: `checagem de links concluida`, sem linhas `QUEBRADO`.

**Step 5: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add docs/results/results_index.md docs/architectures/arch_cnn.md goals/proximo_passo_cnn.md
git commit -m "docs: liga a analise de 50 seeds ao indice e ao host CNN"
```

---

# M1 — Pré-registro do contraste DER++ (CPU)

> **O alvo:** `+3,18 p.p.` de SlowHeat+DER++ sobre DER++ em Split-MNIST, 10/10 seeds, sob Holm — o resultado exploratório mais forte do host MLP, e sem confirmação própria. É o único caminho para uma alegação que não seja "melhor que replay simples".

## Tarefa 15: Verificar quais seeds estão queimadas

**Objective:** Escolher uma banda de seeds que nenhum guard-rail recuse.

**Files:** nenhum; só leitura.

**Step 1: Listar as bandas já reservadas**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -c "
from experiments.confirmatory_split_mnist import CONFIRMATORY_SEEDS, DECLARED_EXPLORATORY_SEEDS
from experiments.qwen_lora_sweep import LORA_CONFIRMATORY_SEEDS
print('Split-MNIST confirmatorias :',len(CONFIRMATORY_SEEDS),'->',CONFIRMATORY_SEEDS[:3],'...',CONFIRMATORY_SEEDS[-1])
print('exploratorias declaradas   :',DECLARED_EXPLORATORY_SEEDS)
print('LoRA confirmatorias        :',LORA_CONFIRMATORY_SEEDS)
"
```

Expected: 20 seeds a partir de 104729; exploratórias como múltiplos de 11; LoRA na banda 700001+.

**Step 2: Confirmar que `dualheat_pairs` tem guard-rail**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
grep -n 'CONFIRMATORY_SEEDS' experiments/dualheat_pairs.py
```

Expected: linha 105-106, com `raise ValueError("seeds reservadas à confirmação não podem entrar nesta suíte")`.

Isto significa que a suíte exploratória **já** recusa as seeds confirmatórias do Split-MNIST. A banda nova de M1 precisa ser disjunta de tudo.

## Tarefa 16: Escrever o pré-registro de M1

**Objective:** Congelar o desenho antes de qualquer execução.

**Files:**
- Create: `goals/protocol_derpp_confirmation.md`

**Step 1: Criar o arquivo**

```markdown
# Protocolo congelado — confirmação de SlowHeat+DER++ em Split-MNIST

> Pré-registro, escrito **antes** de qualquer seed desta confirmação.

**Estado: CONGELADO em <DATA>.**

## A. O que motiva

`docs/architectures/arch_mlp.md`, evidência E3 (suíte `dualheat_pairs`,
exploratória, 10 seeds, Holm por dataset):

| Contraste | Referência | Candidato | Diferença | p Holm | Sinais |
|---|---:|---:|---:|---:|---|
| vs DER++ | 82,40% | 85,58% | **+3,182 pp** | <0,0001 | 10+/0− |

É o resultado exploratório mais forte do host MLP e **não tem pré-registro
próprio**. A confirmação congelada existente cobre apenas
`SlowHeat+Replay − Replay`.

## B. Pergunta confirmatória

Acrescentar Functional SlowHeat ao DER++ melhora a acurácia média final
class-incremental em Split-MNIST, em relação ao DER++ sozinho?

## C. Hipótese declarada antes da run

*`slowheat_derpp_hidden_beta_30_budget_0.25` supera `derpp` na acurácia média
final, com a diferença pareada por seed positiva na maioria das seeds.*

Hipótese nula: diferenças pareadas simétricas em torno de zero. **Resultado
nulo é publicável** e encerra a alegação; não haverá tentativa de resgate com
grade nova.

## D. Decisões congeladas

| # | Decisão | Valor | Justificativa |
|---|---|---|---|
| D1 | Braços | `derpp` e `slowheat_derpp_hidden_beta_30_budget_0.25` — exatamente dois | a pergunta é um contraste único |
| D2 | Seeds | `2_000_003, 2_025_011, 2_050_017, 2_075_037, 2_100_043, 2_125_059, 2_150_061, 2_175_087, 2_200_089, 2_225_097, 2_250_101, 2_275_103, 2_300_119, 2_325_127, 2_350_133, 2_375_149, 2_400_151, 2_425_163, 2_450_169, 2_475_179` (20) | banda própria, disjunta das confirmatórias do Split-MNIST (104729+), das exploratórias (múltiplos de 11), das do LoRA (700001+) e das de QB-2 (1000003+) |
| D3 | Endpoint primário | acurácia média final class-incremental, contraste `slowheat_derpp − derpp` | mesmo endpoint primário do resto do projeto |
| D4 | Endpoints secundários | average forgetting, BWT, tempo de parede | reportados, nunca promovidos |
| D5 | Teste | t pareado **e** sinal exato bicaudal, ambos reportados | a confirmação existente mostrou que eles podem discordar (task-aware: t dá 0,043, sinal dá 0,115); reportar os dois evita escolher o favorável |
| D6 | Correção múltipla | nenhuma — família de **uma** comparação primária | declarado antes; os secundários são descritivos |
| D7 | Critério | `p < 0,05` **nos dois testes** de D5 e diferença média positiva | mais estrito que o padrão, por causa da discordância conhecida |
| D8 | Hiperparâmetros | `beta = 30`, `budget = 0,25`, `lr = 1e-3`, idênticos ao exploratório | **não** re-tunar; re-tunar depois de ver o exploratório é escolher o hiperparâmetro pela acurácia |

## E. Cenário

Split-MNIST class-incremental, 5 tarefas de 2 classes, cabeça global
compartilhada, MLP `[256, 128]`, pareamento por seed de inicialização,
partições, minibatches e índices de replay. CPU.

## F. Verificação antes da run

1. as 20 seeds de D2 não pertencem a nenhuma banda reservada (guard-rail de
   `experiments/dualheat_pairs.py:105` deve aceitar);
2. suíte de testes verde;
3. este arquivo commitado antes da primeira seed.

## G. Análise declarada

```text
para cada seed s: d_s = FAA(slowheat_derpp, s) − FAA(derpp, s)
testes: t pareado bicaudal E sinal exato bicaudal
confirma se: ambos p < 0,05 E media(d_s) > 0
```

Sem parada antecipada, sem inspeção seed a seed.

## H. O que não resolve

- **Um benchmark.** Split-MNIST tem teto alto; `+3 p.p.` ali pode não
  transferir. A inversão de sinal na CNN (`arch_cnn.md`, C1) mostra que o
  efeito do SlowHeat depende do método base.
- **Sem tuning por método.** `lr = 1e-3` fixo para os dois braços. Um revisor
  pode alegar que o DER++ está sub-ajustado — limitação declarada, não
  resolvida.

## K. Registro de alterações

| Data | Alteração | Antes da run? |
|---|---|---|
| <DATA> | criação e congelamento. D1 a D8 fechados. | sim — nenhuma seed executada |
```

**Step 2: Substituir `<DATA>` e verificar as seeds**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
sed -i "s/<DATA>/$(date +%d\\/%m\\/%Y)/g" goals/protocol_derpp_confirmation.md
.venv/bin/python -c "
from experiments.confirmatory_split_mnist import CONFIRMATORY_SEEDS, DECLARED_EXPLORATORY_SEEDS
from experiments.qwen_lora_sweep import LORA_CONFIRMATORY_SEEDS
m1=[2000003,2025011,2050017,2075037,2100043,2125059,2150061,2175087,2200089,2225097,
    2250101,2275103,2300119,2325127,2350133,2375149,2400151,2425163,2450169,2475179]
reserved=set(CONFIRMATORY_SEEDS)|set(DECLARED_EXPLORATORY_SEEDS)|set(LORA_CONFIRMATORY_SEEDS)
print('n M1:',len(m1),'| unicas:',len(set(m1)))
print('colisao:',set(m1)&reserved or 'nenhuma')
"
```

Expected: `n M1: 20 | unicas: 20` e `colisao: nenhuma`.

**Step 3: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add goals/protocol_derpp_confirmation.md
git commit -m "goals: pre-registro da confirmacao de SlowHeat+DER++ (M1)"
```

## Tarefa 17: Registrar as seeds de M1 em código, com guard-rail

**Objective:** Um pré-registro que só existe em markdown pode ser burlado por engano. As seeds do LoRA já aprenderam essa lição (tabela K, 25/09).

**Files:**
- Modify: `experiments/confirmatory_split_mnist.py`
- Modify: `tests/test_split_mnist.py`

**Step 1: Escrever o teste que falha**

Acrescente ao fim de `tests/test_split_mnist.py`:

```python
def test_derpp_confirmatory_seeds_are_registered_and_disjoint() -> None:
    """M1's seeds must live in code, not only in a markdown pre-registration.

    A band that exists only in a document can be burned by an exploratory run
    by accident; the LoRA protocol already hit this (table K, 25/09).
    """

    from experiments.confirmatory_split_mnist import (
        CONFIRMATORY_SEEDS,
        DECLARED_EXPLORATORY_SEEDS,
        DERPP_CONFIRMATORY_SEEDS,
    )

    assert len(DERPP_CONFIRMATORY_SEEDS) == 20
    assert len(set(DERPP_CONFIRMATORY_SEEDS)) == 20
    assert not set(DERPP_CONFIRMATORY_SEEDS) & set(CONFIRMATORY_SEEDS)
    assert not set(DERPP_CONFIRMATORY_SEEDS) & set(DECLARED_EXPLORATORY_SEEDS)
```

**Step 2: Rodar e verificar a falha**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -m pytest tests/test_split_mnist.py::test_derpp_confirmatory_seeds_are_registered_and_disjoint -q
```

Expected: **FAIL** com `ImportError: cannot import name 'DERPP_CONFIRMATORY_SEEDS'`.

**Step 3: Implementar**

Em `experiments/confirmatory_split_mnist.py`, logo após a definição de `DECLARED_EXPLORATORY_SEEDS` (linha ~52), acrescente:

```python
#: Seeds frozen for the SlowHeat+DER++ confirmation (goals/protocol_derpp_confirmation.md).
#: Disjoint from CONFIRMATORY_SEEDS, from the exploratory multiples of 11, from
#: the LoRA confirmatory band (700001+) and from QB-2's band (1000003+).
#: Exploratory runs must never touch these.
DERPP_CONFIRMATORY_SEEDS: tuple[int, ...] = (
    2_000_003, 2_025_011, 2_050_017, 2_075_037, 2_100_043,
    2_125_059, 2_150_061, 2_175_087, 2_200_089, 2_225_097,
    2_250_101, 2_275_103, 2_300_119, 2_325_127, 2_350_133,
    2_375_149, 2_400_151, 2_425_163, 2_450_169, 2_475_179,
)
```

**Step 4: Rodar o teste**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -m pytest tests/test_split_mnist.py::test_derpp_confirmatory_seeds_are_registered_and_disjoint -q
```

Expected: `1 passed`.

**Step 5: Rodar a suíte completa de Split-MNIST**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -m pytest tests/test_split_mnist.py tests/test_confirmatory_statistics.py -q
```

Expected: todos passam.

**Step 6: Commitar**

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
git add experiments/confirmatory_split_mnist.py tests/test_split_mnist.py
git commit -m "experiments: registra em codigo as seeds confirmatorias do DER++"
```

> **Nota:** a execução das 20 seeds é CPU e demora horas. Como toda run deste projeto, **pergunte ao Fachel antes de lançar**.

---

## Validação final

Depois de todas as tarefas, rode a suíte completa:

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -15
```

Expected: nenhuma falha nova em relação à linha de base. Se houver falhas
pré-existentes, elas devem ser as mesmas de antes do trabalho — capture a linha
de base **antes** da Tarefa 1 se quiser comparar com rigor.

Verificação de integridade documental:

```bash
cd /mnt/B-SSD/fachel/dual-heater-mlp
.venv/bin/python -c "
import re,os,glob
bad=[]
for p in glob.glob('goals/*.md')+glob.glob('docs/**/*.md',recursive=True):
    t=open(p,encoding='utf-8').read()
    for l in re.findall(r'\]\(([^)#]+)',t):
        if l.startswith('http'): continue
        tgt=os.path.normpath(os.path.join(os.path.dirname(p),l))
        if not os.path.exists(tgt): bad.append((p,l))
print('links quebrados:',bad or 'nenhum')
"
```

Expected: `links quebrados: nenhum`.

---

## Arquivos que mudam

**Criados:**
- `docs/related_work/protocol_prior_art.md` (L)
- `goals/protocol_lora_rank_matched.md` (QB-2)
- `docs/results/replay_selection_50seed_analysis.md` (C-3)
- `goals/protocol_derpp_confirmation.md` (M1)

**Modificados:**
- `experiments/qwen_lora_slowheat.py` — `build_parser`, flag `--alpha` (QB-2)
- `experiments/confirmatory_split_mnist.py` — `DERPP_CONFIRMATORY_SEEDS` (M1)
- `tests/test_lora_slowheat.py`, `tests/test_lora.py`, `tests/test_split_mnist.py`
- `goals/protocol_iso_plasticity.md` — correção da nota de desvio (QA-6)
- `goals/resultados_confirmacao.md`, `docs/architectures/arch_qwen.md` (QA-6)
- `docs/results/results_index.md`, `docs/architectures/arch_cnn.md`, `goals/proximo_passo_cnn.md` (C-3)
- `goals/proximo_passo_artigo.md` (L)

---

## Riscos, tradeoffs e questões abertas

### Riscos

| Risco | Probabilidade | Mitigação |
|---|---|---|
| **O levantamento (L) encontra o protocolo já publicado** | média | É o desfecho que o bloqueador existe para detectar. Se ocorrer, PARE na Tarefa 4 e escale — a §3 do hub precisa ser reescrita por decisão humana, não por agente. |
| **QB-2 dá nulo e derruba o resultado confirmado de 28/09** | média | Declarado em C do pré-registro como desfecho publicável. É a razão de o experimento existir. Não ajustar rank/alpha para "salvar" o resultado. |
| **O gate de parâmetros (G4) falha** | baixa | `r=32` com `A` congelada deveria dar ~8,4M contra ~6,8M — razão 1,24, **fora** da tolerância de 5%. Ver questão aberta 1. |
| **Rodar `--rank 32` sem `--alpha 32`** | alta se o plano não for seguido | Tarefas 5-7 travam isso por teste. O pré-registro (seção F) explica o porquê. |
| **Recalcular Holm sobre família reduzida em QA-6** | média — é tentador | Proibido explicitamente na Tarefa 10, item 3. Escolher a família depois de ver o resultado invalida a correção. |
| **Apresentar C-3 como confirmação** | média | O artefato já declara `exploratory_not_independent_confirmation`; a Tarefa 12 repete isso no cabeçalho do documento. |

### Tradeoffs

**QB-2 iguala parâmetros, não expressividade.** `exact_r32` tem 32 direções aleatórias fixas em `A`; `vanilla_r16` tem 16 aprendíveis. É a melhor aproximação disponível sem redesenhar o mecanismo, e está declarada na seção I do pré-registro.

**M1 usa `p < 0,05` nos dois testes (D7), mais estrito que o usual.** Custa poder estatístico, mas a confirmação existente já mostrou t e sinal discordando (0,043 contra 0,115). Escolher o teste favorável depois de ver ambos é a falha que D7 impede.

**C-3 é exploratório e sempre será.** Os dados precedem a pergunta. Nenhuma reanálise muda isso; a saída honesta é rotular.

### Questões abertas

1. **A razão de parâmetros em `r=32` provavelmente não fica dentro de 5%.** Estimativa: `exact` congela `A`, então treina só `B`; em `r=16` são 4.214.016 de 6.769.920. Dobrar o rank dobra aproximadamente a parte treinável de `B`, levando a ~8,4M contra 6,77M do `vanilla_r16` — razão ~1,24. Se o gate da Tarefa 9 falhar por isso, as opções são: (a) `exact` com `r=26` para ajuste fino da contagem, (b) aceitar tolerância maior e declarar, (c) parear por FLOPs em vez de parâmetros. **Todas exigem linha na tabela K antes da run.** Não escolher depois de ver acurácia.

2. **O `alpha` correto para `r=32` é 32, ou deveria ser 16 com `scaling` explicitamente fixado?** O plano assume `alpha=32` para manter `scaling=1,0`. A alternativa seria expor `--scaling` diretamente. A primeira é menos invasiva e usa a semântica padrão de LoRA.

3. **QA-6 enfraquece a parte A do Qwen a ponto de removê-la do artigo?** O plano registra o problema sem decidir o posicionamento. Essa é uma decisão de escopo do Fachel, e depende do veredito de L.

4. **M1 vale o custo?** São 20 seeds em CPU por um contraste em Split-MNIST, um benchmark com teto alto. Se a âncora final for protocolo (e não eficácia), M1 vira trabalho futuro. Confirmar a prioridade com o Fachel antes da Tarefa 16.
