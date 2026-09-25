# Revisão documental — 22 de setembro de 2026

Auditoria cruzada de **30 documentos** (`README.md`, `article/manuscript.md`,
25 arquivos em `docs/`, 4 em `goals/`) contra o código-fonte (`src/`,
`experiments/`), a suíte de testes e os artefatos em `results/` e `artifacts/`.

Método: leitura integral dos documentos; validação dinâmica por importação dos
registros de métodos (`.venv/bin/python`); verificação numérica por amostragem
em 94 `aggregate.json` e 2147 CSVs; execução de `ruff check` e
`pytest --collect-only`. Nenhum treinamento foi executado.

Estado verificado da suíte: **567 testes coletados**, **23 erros de Ruff**.

---

## Sumário executivo

| Categoria | Achados |
|---|---:|
| Falhas críticas de validade científica | 4 |
| Documentos com status falso (desatualizados) | 9 |
| Achados de auditoria abertos silenciosamente | 24 |
| Resultados órfãos (existem, não citados) | 13 |
| Resultados prometidos e nunca produzidos | 17 |
| Contradições entre documentos | 6 |
| APIs documentadas e inexistentes | 7 |

O problema dominante **não é falta de resultados — é o inverso**. O repositório
contém muito mais evidência executada do que os documentos reconhecem, e a
maior tabela publicada (Split-MNIST, §7 de `project_methods_and_results.md`)
não tem artefato correspondente em disco.

---

## 1. Falhas críticas

### 1.1 A confirmação congelada EXISTE, e três documentos afirmam que não

**Afirmações contestadas:**

- `article/manuscript.md:8-9` e Abstract: *"the frozen independent confirmation
  has no valid versioned execution artifacts"*
- `docs/protocols/confirmatory_protocol.md:3`: *"nenhum artefato confirmatório está
  versionado"* e `:25`: *"O repositório continua sem artefatos confirmatórios
  versionados"*
- `docs/results/split_mnist_experiment_log.md:§11`: artefatos não versionados
- `README.md:126-127`: *"no confirmatory result artifacts are versioned"*

**Realidade verificada em disco — duas execuções completas e independentes:**

```
results/protocol_post_eval_fix/confirmation/          (20 seeds, 2026-09-03)
results/protocol_post_eval_fix_d5b22ad/confirmation/  (20 seeds, 2026-09-04)
```

Ambas contêm `preregistration.lock.json` com
`status = frozen_before_execution`, `preregistered_at = 2026-08-15`,
20 seeds confirmatórias e `sha256 = 015b3162…a9b2` **idêntico nas duas** —
prova de que o pré-registro não foi editado entre as execuções.

**Endpoint primário (`final_average_accuracy`), diferença pareada
SlowHeat+Replay − Replay:**

| Estatística | Valor |
|---|---:|
| Diferença média | **+0,869 p.p.** |
| IC95% t pareado (gl=19) | [+0,293; +1,445] p.p. |
| t | 3,157 |
| p bilateral | **0,00520** |
| Bootstrap pareado IC95% | [+0,316; +1,381] p.p. |
| Teste de sinais | 17 positivos / 3 negativos / 0 empates, p = 0,00258 |

Secundário, `average_forgetting`: −1,4625 p.p., t = −4,112, p = 0,00059,
18 negativos / 2 positivos.

**Validade quanto à correção de eval-mode.** O documento afirma que qualquer
execução anterior a 2026-09-02 é inválida. Verifiquei que
`experiments/evaluation.py` — o contexto de avaliação sem efeito colateral que
implementa a correção — foi introduzido **no commit `d5b22ad` (2026-09-03)**,
que é exatamente o commit registrado em
`results/protocol_post_eval_fix/confirmation/environment.json`. As duas
execuções são **posteriores à correção** e usaram diretórios de saída novos,
como o protocolo exige.

**Replicação determinística confirmada.** Comparei os 20 `results.json` par a
par entre os dois diretórios: 100 campos escalares, dos quais **21 diferem — e
todos são de custo** (`elapsed_seconds`, `optimizer_step_seconds`,
`peak_memory_*`, `selection_seconds`). **Nenhuma métrica científica difere.**
Não é duplicação de arquivos (todos os hashes SHA-256 são distintos): são duas
execuções independentes, em commits diferentes, que reproduziram resultados
bit-idênticos. Isso **fortalece** a confirmação em vez de enfraquecê-la.

**Ressalva real:** ambas registram `git dirty = true`. O diff executado não foi
fingerprintado, então não há reprodução bit a bit pela proveniência.

**Ação:** corrigir as quatro afirmações, reportar a tabela acima como resultado
confirmatório e declarar a ressalva de árvore suja. Esta é a conclusão
científica mais forte do repositório e hoje ela não aparece em documento nenhum.

---

### 1.2 A tabela principal de Split-MNIST não tem artefato

`docs/audits/project_methods_and_results.md` §7 (13 métodos), §8.1–8.10 e §10
sustentam todo o documento. Busca exaustiva nos 94 `aggregate.json` e 2147
CSVs: **nenhum arquivo contém esses valores**.

| Métrica (doc §7) | Doc | Disco mais próximo | Fonte |
|---|---:|---:|---|
| Vanilla | 19,588% | 19,596% / 19,629% / 19,602% | CSV solto / 20 seeds / 100 seeds |
| Replay | 76,524% | 76,268% / 76,536% | CSV solto / 20 seeds |
| SlowHeat+Replay | 76,824% | 76,758% / 77,671% | CSV solto / 20 seeds |
| DER++ | 81,984% | 82,504% / 82,691% | CSV solto / 20 seeds |
| SlowHeat+DER++ | 84,708% | 85,062% / 85,835% | CSV solto / 20 seeds |

O delta central do documento (**+2,724 p.p.** para SlowHeat+DER++) não se
reproduz: o CSV dá +2,558 p.p. e o agregado de 20 seeds dá +3,144 p.p.

O candidato mais próximo é `results/split_mniist_results.csv` — sem seeds, sem
commit, sem configuração — e **ainda assim seus números divergem**. O
documento admite em §9.4 que a proveniência falta; o que não está registrado é
que os números tampouco batem com o único arquivo candidato.

**Ação:** marcar §7–§10 como não rastreáveis, ou substituí-las pelo agregado de
20 seeds (`protocol_post_eval_fix_d5b22ad/all_baselines_equal_epochs/`) ou pelo
de 100 seeds, ambos completos e em disco.

---

### 1.3 `docs/results/split_mnist_experiment_log.md` §6 também diverge

Doc: Replay 0,76288 vs SlowHeat 0,77549, delta **+1,261 p.p.**, IC t
[+0,669; +1,853].
Disco (único par replay/candidato de 20 seeds): 0,77040 vs 0,77909, delta
**+0,869 p.p.**, IC t [+0,293; +1,445].

Os três números do documento divergem. As agregações de 8 e 19 seeds citadas
em §5.5/§5.6 não têm nenhum artefato.

---

### 1.4 Confirmação Qwen rodando sem os gates que o próprio pré-registro exige

`goals/protocol_iso_plasticity.md` §J declara que **o Gate 3 precede a
sequência de 10 tarefas**. Nem Gate 1 nem Gate 3 foram registrados em documento
algum.

Estado no momento desta revisão — três processos ativos há ~1h36:

```
PID 3011516  --seed 10  --steps-per-task 120  --domains <10 domínios>
PID 3011517  --seed 11
PID 3011518  --seed 12
```

- Declaradas 10 seeds (10–19); **3 iniciadas, 7 não**.
- `results/qwen_iso_plasticity/` contém `confirm120_seed*`: **zero diretórios**.
- O driver é `run_confirmation.sh`, na raiz e **não versionado**
  (`git status` → `?? run_confirmation.sh`). O comando exato que produzirá a
  confirmação não está sob controle de versão.
- Existe execução anterior abortada a 30 passos (`results/run_logs/confirm/`)
  com FAA ≈ 0,047 e retenção 0,013 — piso do acaso —, que motivou a mudança
  para 120 passos (commit `4265b1c`).

**Consequência:** as 6 runs existentes (`seed0-2`, `hard_seed0-2`) usam
**30 passos**, valor revogado pela tabela K do protocolo congelado. Todos os
resultados Qwen de calibração estão obsoletos.

**Desvio de pré-registro não documentado:** os manifestos `hard_seed*` têm
**8 braços por família**, incluindo `hard_b0.75`/`hard_b0.5`. A seção E do
protocolo congelado declara 7 braços em E\*=0,75 e 6 em E\*=0,50, sem braço
hard. O commit `095b1c8` adicionou o braço **sem a linha correspondente na
tabela K**, violando a regra das linhas 262-263 do próprio protocolo.

---

## 2. Documentos com status falso

| Documento | Afirma | Realidade |
|---|---|---|
| `article/manuscript.md:12,31` | "Split-CIFAR-100 … no completed aggregate result" | **Dois agregados completos**: `split_mnist_protocol/split_cifar100/` (10 seeds × 19 métodos) e `dualheat_pairs/split_cifar100/` (10 seeds × 8 métodos), além de 5 agregados de 50 seeds no sweep de replay |
| `article/manuscript.md:8-9` | confirmação sem artefatos | ver §1.1 |
| `docs/protocols/confirmatory_protocol.md:3,25` | idem | ver §1.1 |
| `docs/protocols/split_cifar.md:3-5` | "quatro seeds parciais … não há agregado CIFAR completo" | 8 diretórios CIFAR completos com agregado |
| `docs/results/bert_full_coverage_ablation.md:3` | "ainda sem resultados experimentais" | **30 runs completas** em `results/bert_full_coverage/` (10 métodos × 3 seeds), com `aggregate.json`, `resultados_resumo.md` e `resultados_por_seed.csv` |
| `docs/protocols/qwen_iso_plasticity_ablation.md:3-4` | "O diagnóstico no checkpoint real ainda não foi executado" | contradiz as próprias §9 e §10, que reportam medições do Qwen real; e 15 manifestos em disco |
| `docs/architectures/functional_slowheat_qwen.md:13,127` | "Smoke no checkpoint real (ainda não executado)" | `results/run_logs/results_gpu_smoke.log` e `results_gpu_memory_smoke.log` registram execução em GTX 1080 Ti; o pico medido (5,447 GiB) **refuta a estimativa de ~8,3 GB** da §6 do mesmo documento |
| `goals/plano_execucao_p6_p1_p2_p3.md:3-5` | "NÃO APLICADO. Nada deste documento foi implementado, executado ou commitado" | P6.1, P1, P2 e P3 implementados e executados (commits `c907433`, `6461827`, `8c589ad`) |
| `goals/qwen_heat_roadmap.md:206` | "Runner não existe" | `experiments/qwen_iso_plasticity.py`, 33.958 bytes, 42 testes, 5 seeds de resultados |

---

## 3. `docs/audits/experiments_audit.md` — estado real dos 36 achados

A tabela "Status de correção" cobre **12 dos 36 achados**. Verifiquei os 36
individualmente contra o código atual (ignorando os números de linha, que estão
obsoletos).

**Corrigidos e confirmados (9):** 1 (Fisher EWC), 2 (pareamento de replay),
3 (Classifier Expander), 4 (RNG no resume), 5 (teste CLINC fechado por padrão),
6 (identidade do checkpoint), 9 (scheduler por tarefa), 13 (custo após resume —
corrigido mas **ausente da tabela e sem teste nomeado**), mais os dois achados
extras de fp16.

**Pendentes confirmados (4):** 7 (pré-registro removível), 8 (manifesto
FastHeat não verificável), 17 (validadores aceitam `1.5`, `True` e NaN),
22 (`ReplayBuffer` aceita `inputs=None` com targets não vazios).

**Abertos silenciosamente — 24 achados ausentes da tabela de status:**

| Severidade | Achados |
|---|---|
| Média | 10, 11, 12, 14, 15, 16, 18, 19, 20, 21, 23, 24 |
| Média/alta (telemetria) | 25, 26, 27, 28, **29** (escrita fora do run root por symlink), 30, 31 |
| Baixa | 32, 33, 34, 35, 36 |

Destaques:

- **#10** — o contraste `dualheat_global_topk − slowheat_global` continua fora
  de `pairs` em `split_clinc150.py:1827`. É exatamente o contraste que
  `docs/bert_clinc150_results.md` §9 pede há duas revisões.
- **#11** — `--seeds` (default `[11,22,33]`) continua compartilhada entre
  `--calibrate` e a execução final, sem validação de disjunção.
- **#16** — a validação de `data_identity.json` é apenas sintática; o
  fingerprint nunca é recalculado no caminho de reuso.
- **#29** — `heat-latest.json`, o catálogo e o diretório `telemetry` do writer
  continuam sem `_is_within`.

**Números da própria seção de status estão errados:**

| Afirmação do doc | Verificado |
|---|---|
| "301 passed" no fim do milestone | **567 testes coletados** |
| "os 11 erros de Ruff permanecem" | **23 erros** (RUF100=7, ISC004=4, PYI036=3, I001=3, UP037=2, UP035, SIM117, PYI034, F401); 15 auto-corrigíveis |

**Referências de fase não rastreáveis:** "Fase E1/E2/D3/D4/H" só existem em
`.hermes/plans/2026-09-14_171737-auditoria-e-refatoracao-dual-heater.md`, que o
documento nunca cita. Pior, os rótulos E1/E2 são **sobrecarregados** — o mesmo
plano usa E1/E2 em dois contextos distintos (linhas 253/259 vs 931/941).

**"Máscara fail-closed ao limpar"** aparece na tabela mas não está entre os 36
achados. Refere-se a `clear_plasticity_masks()` em `src/dual_heater/optim.py:398`:
limpa `_plasticity_masks` mas **não** `_expected_mask_signatures`, e o fallback
em `_state_dict_with_mask_metadata` grava silenciosamente as assinaturas
antigas. **Nenhum teste cobre esse método** (`grep clear_plasticity_masks tests/`
→ zero).

---

## 4. Resultados órfãos — existem, nenhum documento os cita

| Diretório | Conteúdo | Gravidade |
|---|---|---|
| `results/protocol_post_eval_fix_d5b22ad` | 1,7 GB — 9 agregados: confirmação (20 seeds), baselines equal-epochs (20 seeds × 12 métodos), SlowHeat+DER++ (20 seeds), ablações de método e de memória 5/10/20/50/100 | **crítica** |
| `results/protocol_post_eval_fix` | 137 MB — confirmação de 20 seeds + lock de pré-registro | **crítica** |
| `results/100seeds` | 1,1 GB — **100 seeds** × 12 métodos. Maior *n* do repositório; nenhuma tabela publicada o usa | alta |
| `results/cache_derpp_10seeds` | **100 GB** — sweep de seleção de replay completo: 5 datasets × 5 caches, 25 agregados de 50 seeds cada. Maior artefato em disco, totalmente órfão | alta |
| `results/replay_selection_full` | 1,1 GB — 10 agregados de 10 seeds | alta |
| `results/bert_slowheat_diagnostic` | 2,4 GB — versão de **3 seeds** do mesmo diagnóstico, com números diferentes de `bert_slowheat_review` (vanilla 46,83% vs 48,47%; hard 71,61% vs 71,38%). O nome do documento (`bert_slowheat_diagnostic_results.md`) sugere este diretório, mas as "Fontes primárias" apontam para o outro | **alta — risco de confusão** |
| `results/qwen_layer_anomaly` | 7 manifestos, 2 ordens × 3 seeds. Executado para decidir se a anomalia de L3/L21 é artefato de seed/ordem; **a resposta nunca foi escrita**. Os dados mostram PR agregado variando 11.567–27.955 entre seeds, sugerindo forte dependência de seed | alta |
| `results/qwen_iso_plasticity` | resultado mais recente do repositório (22/09 16:26), e o doc correspondente ainda diz "não executado" | alta |
| `results/secondary_post_eval_fix_d5b22ad` | 819 MB — inclui a **única análise `equal_examples`** do repositório, que o manuscrito §4.2 discute mas nunca reporta | média |
| `results/new_bert` + `results/secrets` | 402 MB + 445 MB — duas execuções independentes do mesmo protocolo (mesma estrutura, todos os hashes diferentes). `new_bert/ANALISE.md` tem tabela própria que não aparece em doc nenhum. O nome `secrets` é opaco e sem README | média |
| `results/protocol_post_eval_fix_d5b22` | 274 MB — run parcial (só `slowheat_derpp_exploratory`) | média |
| CSVs soltos na raiz de `results/` | `split_mniist_results.csv`, `permutated_mnist_download.csv`, `pair_differences.csv`, `paired_differences.csv` — sem seeds, sem commit | alta |
| `results/dualheat_pairs_portable`, `_existing`, `_smoke` | relatórios derivados sem dados brutos | baixa |

`docs/mechanisms/replay_selection.md` referencia
`results/cache_all_datasets_10seeds/replay_selection_sweep` em quatro blocos de
comando. **Esse diretório não existe.** O sweep real está em dois outros
caminhos, nenhum deles citado.

---

## 5. Resultados prometidos e nunca produzidos

**`article/manuscript.md` §8 — 10 experimentos exigidos antes da submissão:**

| Item | Estado |
|---|---|
| 1. Tuning de LR e β por família de otimizador | **nada** — todos os agregados usam `lr=1e-3` fixo |
| 4. MAS, activation-based importance, UCB, HAT, SLNID, joint training | **nada** — nenhum aparece entre os 57 nomes de método em disco, nem há implementação |
| 6. Ablação functional vs activation importance | **nada** |
| 8. `follow_update` vs `native` | apenas nos CSVs sem proveniência; **nenhum agregado** |
| 9. Sweep β × budget com fronteira de Pareto | **nada** |
| 10. Medições de eficiência após warm-up | **nada** — todos os relatórios declaram "tempo descritivo, sem medição isolada com aquecimento" |

**Outros:**

- `project_methods_and_results.md` §15.3 — pré-registro separado de
  SlowHeat+DER++: os resultados existem em três diretórios, mas **nenhum tem
  `preregistration.lock.json`**. O único lock do repositório é o de
  replay vs slowheat_replay.
- §15.5 (profiling do otimizador), §15.7 (matrizes de confusão, distribuição de
  logits): **nada**.
- `bert_clinc150_results.md` §9 — os 10 itens do protocolo recomendado:
  **nenhum executado**.
- `bert_full_coverage_ablation.md` — fluxo confirmatório de 5 passos: não
  executado; o `results/bert_full_coverage_smoke` citado não existe.
- `qwen_capacity_calibration.md` §4/§7 — o critério declarado
  (`--min-effective-plasticity 0.6`) **nunca foi aplicado**:
  `results/qwen_capacity_diagnostic/manifest.json` tem
  `minimum_effective_plasticity: null`, `declared_before_run: false`,
  `selection: null`.
- `docs/qwen_layer_anomaly.md` — prometido como entregável de P6 e base do
  Gate 2: **não existe**.
- Gate 1, Gate 2 e Gate 3 do roadmap Qwen: **nenhum registrado**.
- `protocol_iso_plasticity.md` §G declara "Drift protegido" e "Drift plástico"
  como endpoints obrigatórios e tokens/s como métrica de custo.
  `capture_parameter_drift_reference` é chamado, mas o bloco `endpoints` do
  manifesto **não tem chave de drift nem de tokens/s**.
- `functional_slowheat_rnn_lstm.md` — 581 linhas de contrato: `TemporalSlowHeatTracker`,
  `SlowHeatRNNCell`, `SlowHeatLSTMCell`, `finish_backward()`,
  `begin_backward_window()`, 12 testes de aceitação. `grep -riE 'rnn|lstm|gru|recurrent'
  src/` → **zero**. O documento se rotula corretamente como proposta; registrado
  aqui só para o inventário.

---

## 6. Contradições entre documentos

### 6.1 SwiGLU: declarado "futuro" em dois documentos, implementado no código

- `docs/mechanisms/functional_slowheat_transformers.md:8,67,625,670` — *"SwiGLU … continua
  como extensão futura"*, *"### Etapa E — SwiGLU (futura)"*
- `docs/README.md:18` — *"SwiGLU, GQA, QKV fundido e distribuição ainda
  planejados"*

`src/dual_heater/qwen.py` **implementa a Etapa E**:
`SlowHeatQwen2ForSequenceClassification` rastreia o produto gated do MLP SwiGLU
via `forward_pre_hook` em `down_proj` (`qwen.py:289-304`) e registra exatamente
o agrupamento pedido — linhas de `gate_proj`/`up_proj` como produtoras, colunas
de `down_proj` como consumidora (`qwen.py:430/435/441`), fixado pelo teste
`test_producer_masks_are_rows_and_consumer_mask_is_columns`.

GQA e QKV fundido continuam corretamente pendentes.
`docs/architectures/functional_slowheat_qwen.md` é o único dos três que está certo.

### 6.2 `docs/README.md` se declara autoritativo e ignora Qwen por completo

`grep -ci qwen docs/README.md` → **0**. Não há Qwen na tabela "Implementação",
nem em "Estado atual", nem em "O que foi adicionado", nem em "Resultados".

Ficam órfãos: `src/dual_heater/qwen.py` (22 KB), três runners
(`qwen_slowheat_smoke.py`, `qwen_capacity_diagnostic.py`,
`qwen_iso_plasticity.py`), **87 testes** em três arquivos, e os três documentos
Qwen — todos datados de 22/09, posteriores ao índice de 19/09.

### 6.3 Limitações obsoletas em `docs/audits/methods_catalog.md`

| Linhas | Afirma | Código |
|---|---|---|
| 119-121 | "o EWC atual usa o quadrado do gradiente médio do minibatch" | `split_mnist.py:1297-1322` faz gradiente **por exemplo**, `.square()` antes da redução, e divide por `fisher_examples`. É o Fisher empírico correto |
| 148-150 | "Classifier Expander destila sobre todas as classes vistas" | `_old_class_distillation_loss` (`:1277-1294`) faz `index_select` apenas em `old_classes` |

Ambas foram corrigidas (com testes de regressão) e o catálogo ainda as anuncia
como limitações conhecidas.

### 6.4 Restrição de backbone omitida

A tabela "Métodos visuais adicionados" lista os 9 identificadores
LPR/Classifier Expander/SCROLL sem restrição. `split_mnist.py:631-645` levanta
`ValueError` quando `backbone != 'cnn'`. São despacháveis **só** com
`--backbone cnn`.

### 6.5 `project_methods_and_results.md` §16 — estrutura do repositório

Lista 6 módulos em `src/dual_heater/`; existem **12**. Omite `bert.py` (47 KB,
o maior), `qwen.py`, `transformer.py`, `resnet.py`, `state.py`, `_layers.py`.

### 6.6 Presets e contagens verificados como **corretos**

Para registro — estas afirmações foram testadas e conferem: `--heat-variants`
= 4 métodos; `--full-coverage-variants` = 10; `sections.py` = 21 seções;
runner visual = 39 identificadores (correspondência exata nos dois sentidos);
sintético = 11; estratégias de replay = 4; o parser aceita de fato
`slowheat_unidirectional_hidden_beta_30_budget_0.25`.

---

## 7. APIs documentadas que não existem

| API | Citada em | Real |
|---|---|---|
| `FunctionalSlowHeatMixin` | `functional_slowheat_cnn.md:285` | `_SlowHeatImportanceMixin` (privado), `slow_heat.py:91` |
| `update_task_importance()` | `functional_slowheat_cnn.md:294` | não existe; equivalente é `_update_task_ema` |
| `registry.connect()` / `registry.merge()` | `functional_slowheat_cnn.md:260-262` | não existem |
| `capacity_metrics()` nos modelos visuais | `methods_catalog.md:24-27` | existe **só** nas camadas e nos hosts BERT/Qwen. `SlowHeatMLP/CNN/VGG11/ResNet18` → `AttributeError` |
| `TemporalSlowHeatTracker`, `SlowHeatRNNCell`, `SlowHeatLSTMCell`, `finish_backward()`, `begin_backward_window()` | `functional_slowheat_rnn_lstm.md` | não existem (proposta) |

**Módulos e símbolos públicos não documentados em lugar nenhum:**
`state.py` (`FP32ScientificStateMixin`, `register_counter` — garante buffers
FP32, relevante para reprodutibilidade numérica), `_layers.py`, `CIFARResNet18`,
`CLMetrics`, `compute_cl_metrics`, `BertSlowHeatConfig`,
`ExactSlowHeatLoRAConfig`, `exact_lora_mask_bindings`,
`register_exact_lora_masks`, e os 4 runners de diagnóstico BERT.

**Caminhos citados e inexistentes:** `results/split_mnist_10seeds/`
(`README.md:119`), `results/split_mnist_protocol/primary_result.json`
(`README.md:726`), `results/split_mnist_protocol/split_cifar10_cnn_sweep/`
(`README.md:711`, `split_cifar.md:227`),
`results/cache_all_datasets_10seeds/…` (`replay_selection.md`, 4 ocorrências).
Os 80 demais caminhos verificados existem; `article/manuscript.md` §10 está
íntegro.

---

## 8. Divergências numéricas pontuais

Verifiquei 24 números por amostragem. **22 conferem exatamente** — todos os de
BERT/CLINC150 (vanilla 48,47%; β3/β10/β30 51,75/59,65/66,05%; hard 71,38%;
ganho 22,92 p.p.; T1 forgetting 90,90→41,03; drift protegido exatamente zero;
−0,83 p.p. e 2,21× do estudo de replay; +2,17/−3,67 p.p. do orçamento 1),
os do piloto sintético, os 11 runs históricos BERT e os contrastes
Functional DualHeat, inclusive contagens de sinais.

**Duas divergências**, ambas em `docs/protocols/qwen_iso_plasticity_ablation.md` §10:

| Grandeza | Doc | `qwen_capacity_diagnostic/manifest.json` | Δ |
|---|---:|---:|---:|
| Jaccard do conjunto protegido | 0,890 | **0,8699** | 0,020 |
| Turnover do pool livre | 0,296 | **0,3454** | 0,049 (~17%) |

Nenhum manifesto em disco produz esses valores — nem
`manifest.pre-concentration.json`, nem os 7 de `qwen_layer_anomaly`
(faixas 0,8646–0,8829 e 0,3145–0,3578). Os outros 8 números da mesma seção
conferem.

---

## 9. Ordem recomendada de correção

1. **Reportar a confirmação congelada** (§1.1). Corrigir as 4 afirmações falsas
   e publicar a tabela pareada. É a evidência mais forte do projeto e está
   invisível.
2. **Bloquear a confirmação Qwen** até registrar Gate 1 e Gate 3, versionar
   `run_confirmation.sh` e acrescentar a linha do braço hard à tabela K (§1.4).
3. **Marcar `project_methods_and_results.md` §7–§10 como não rastreáveis** ou
   substituí-las pelos agregados de 20/100 seeds (§1.2).
4. **Atualizar os 9 documentos com status falso** (§2) — mudanças de uma linha,
   alto impacto.
5. **Reescrever a tabela de status de `experiments_audit.md`**: cobrir os 36
   achados, marcar os 24 abertos, corrigir 301→567 testes e 11→23 Ruff,
   resolver a ambiguidade E1/E2 (§3).
6. **Documentar Qwen em `docs/README.md`** e corrigir o status de SwiGLU nos
   dois documentos (§6.1, §6.2).
7. **Indexar os 13 diretórios órfãos** (§4), com atenção especial à colisão de
   nomes `bert_slowheat_diagnostic` vs `bert_slowheat_review`, e renomear ou
   documentar `results/secrets`.
8. **Remover as duas limitações obsoletas** do `methods_catalog.md` e corrigir
   as 7 APIs inexistentes (§6.3, §7).
9. **Conferir os dois números de Jaccard/turnover** do §10 de
   `qwen_iso_plasticity_ablation.md` (§8).
10. **Fechar os pendentes P0**: pré-registro removível, manifesto FastHeat,
    validadores, `ReplayBuffer`, e cobrir `clear_plasticity_masks()` com teste.
