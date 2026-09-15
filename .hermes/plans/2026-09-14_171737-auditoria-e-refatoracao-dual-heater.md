# Auditoria e refatoração incremental do Dual Heater

> **Para Hermes:** use a skill `subagent-driven-development` para executar este plano tarefa por tarefa, sempre com revisão de conformidade e depois revisão de qualidade.

**Goal:** transformar o repositório em uma base de pesquisa reproduzível e modular sem reescrita geral, corrigindo primeiro os defeitos que podem invalidar resultados.

**Current context / assumptions:** o branch inicial é `main`, no commit `48ffa39`, e estava limpo na inspeção de 2026-09-14. A auditoria foi estática; nenhum teste ou treinamento foi executado nesta fase de planejamento. Os resultados documentados continuam exploratórios. O diretório `results/` citado na documentação não está versionado; somente `artifacts/synthetic_ablation_pilot/` permite inspeção direta de dados agregados históricos.

**Architecture:** manter `src/dual_heater` como biblioteca dos mecanismos e separar gradualmente o código de protocolo em um pacote instalável `src/dual_heater_experiments`, preservando fachadas de compatibilidade em `experiments/` e nos scripts da raiz. Corrigir contratos científicos em pequenos slices TDD antes de mover módulos; cada execução deve produzir identidade imutável de código, dados, configuração, seeds e protocolo.

**Tech stack:** Python 3.10-3.13, PyTorch, NumPy, pytest, Ruff, Transformers/PEFT opcionais, JSON/CSV versionados e GitHub Actions.

---

## 1. Resumo do que o projeto faz

O projeto pesquisa mecanismos de plasticidade por unidade para continual learning:

- `Functional SlowHeat`: estima utilidade por unidade com `|z * dL/dz|`, consolida a importância em fronteiras de tarefa, reserva uma fração mínima de unidades plásticas e mascara o delta final de SGD/AdamW.
- `Functional DualHeat`: combina SlowHeat com FastHeat, um gate divisivo de curto prazo aplicado às ativações.
- `DualHeat` legado: mantém heats online e usa hooks no gradiente bruto; sua semântica é diferente da implementação Functional.
- Runners comparam esses mecanismos a vanilla, replay, DER++, ER-ACE, A-GEM, EWC, SI, LwF, LPR, hard freeze e controles em dados sintéticos, MNIST, CIFAR e CLINC150/BERT.
- A infraestrutura também registra matrizes stage-by-task, custos, checkpoints, proveniência, estatística pareada e telemetria.

O objetivo científico correto neste estágio não é provar superioridade. É obter comparações identificáveis, pareadas e reproduzíveis que revelem a fronteira estabilidade-plasticidade.

## 2. Arquitetura atual

### Biblioteca

- `src/dual_heater/_layers.py`: validações e fábricas de ativação.
- `src/dual_heater/fast_heat.py`: estado e gate FastHeat.
- `src/dual_heater/slow_heat.py`: mixin de importância, Linear/Conv2d, MLP/CNN/VGG e wrappers Functional, em 1.019 linhas.
- `src/dual_heater/dual_heat.py`: API histórica DualHeat.
- `src/dual_heater/lora.py`: LoRA histórico.
- `src/dual_heater/optim.py`: aplicação de máscaras ao delta final e ao estado do otimizador.
- `src/dual_heater/transformer.py`: trackers FFN e atenção.
- `src/dual_heater/bert.py`: instrumentação BERT, FastHeat, máscara fechada, checkpoint e PEFT, em 912 linhas.
- `src/dual_heater/resnet.py`: ResNet-18 CIFAR e bindings de proteção.
- `src/dual_heater/metrics.py`: FAA, forgetting, BWT e FWT.

### Experimentos

- `experiments/split_mnist.py`: configuração, dados, treino, 32 métodos, custos, checkpoint e agregação, em 2.872 linhas.
- `experiments/split_clinc150.py`: dados, BERT, treino, resume, agregação e CLI, em 1.637 linhas.
- `experiments/model_factory.py`, `method_specs.py`, `sections.py`: pareamento e catálogos parciais.
- `experiments/artifacts.py`, `provenance.py`: escrita atômica e identidade.
- `experiments/replay_memory.py`, `lpr.py`: memória e precondicionamento.
- `run_all_tests.py`: orquestra experimentos; apesar do nome, não é o runner do pytest.

### Distribuição e testes

- `pyproject.toml` empacota somente módulos sob `src/`.
- `experiments/` e scripts raiz funcionam apenas no checkout, com `PYTHONPATH=src:.` ou alteração de `sys.path`.
- A CI testa Python 3.10/3.13 e Ruff, mas não instala NLP, não constrói wheel e não verifica formatação ou tipos.

## 3. Principais problemas encontrados

1. O baseline EWC calcula o quadrado do gradiente da loss média, não a média do quadrado dos gradientes por exemplo.
2. O Classifier Expander destila também classes novas a partir de um professor que ainda não as aprendeu.
3. Seletores adaptativos de replay usam o modelo de cada método, contrariando a alegação de memória idêntica entre pares.
4. CLINC consulta teste por padrão, não restaura RNG no resume e não vincula checkpoint ao fingerprint do código.
5. Os validadores de pré-registro podem ser enfraquecidos removendo campos do próprio manifesto.
6. O manifesto FastHeat aceita qualquer candidato da grade sem recomputar o vencedor do piloto.
7. `global_topk` é aceito por `FastHeatConfig`, mas fora do BERT executa silenciosamente `mean_others`.
8. Heats funcionais e assinaturas científicas podem ser convertidos para fp16/bf16 junto com o modelo.
9. Contadores float dos módulos legados podem parar de incrementar em precisão reduzida.
10. A proteção BERT só é realmente fechada quando opções não padrão são ativadas.

## 4. Bugs ou riscos de resultados incorretos

### P0, corrigir antes de qualquer novo resultado

| Problema | Evidência | Consequência |
|---|---|---|
| Fisher EWC incorreto | `experiments/split_mnist.py:1950-1967,2179-2184` | EWC subestimado/distorcido; comparação contra SlowHeat inválida. |
| Destilação de classes novas | `experiments/split_mnist.py:1871-1885` | aquisição da tarefa atual é empurrada para logits aleatórios do professor. |
| Replay adaptativo não pareado | `experiments/split_mnist.py:2065-2078`, `experiments/replay_memory.py:350-414` | mecanismo e composição da memória ficam confundidos. |
| Teste CLINC por padrão | `experiments/split_clinc150.py:173-205,1107-1128` | tuning indireto no teste. |
| Resume CLINC sem RNG | `experiments/split_clinc150.py:842-863,1149-1169` | run retomado diverge do contínuo por dropout. |
| Resume CLINC sem código | `experiments/split_clinc150.py:733-787` | checkpoint pode ser reutilizado após mudança de implementação. |
| Pré-registro removível | `experiments/confirmatory_split_mnist.py:102-141` | edição do JSON pode reduzir silenciosamente o conjunto congelado. |
| Manifesto FastHeat não verificável | `experiments/functional_dualheat.py:239-258` | escolha retrospectiva de qualquer ponto da grade. |
| `global_topk` silenciosamente errado | `src/dual_heater/fast_heat.py:101-105` | modelos não BERT rotulados top-k executam outra equação. |
| Estado científico em dtype reduzido | `fast_heat.py:86`, `slow_heat.py:142-148` | ranking e máscaras podem mudar por underflow/arredondamento. |

### P1, corrigir antes de comparações de custo ou confirmação

- Scheduler único em CLINC faz tarefas tardias receberem LR menor.
- Seeds de calibração e confirmação CLINC podem coincidir.
- O contraste `dualheat_global_topk - slowheat_global` não é agregado.
- Resume CLINC descarta tempo/pico anteriores.
- Reuse de resultados Split-MNIST não recalcula fingerprint dos dados atuais.
- `ReplayBuffer.load_state_dict()` aceita `inputs=None` com estado não vazio.
- `require_positive_integers()` aceita `True` e `1.5`; `require_nonnegative_values()` aceita NaN.
- `clear_plasticity_masks()` apaga registros depois de desativar hooks e permite seguir sem proteção.
- A contagem de custo de LPR em convoluções ignora o fator espacial.
- Telemetria relê JSONL desde o início e desserializa snapshots completos para montar catálogo.

### Guardrails científicos

- Não chamar hook de gradiente de taxa de aprendizado efetiva sob AdamW.
- Não chamar SlowHeat de EWC.
- Não tratar redução de forgetting com queda de FAA como superioridade.
- Não usar tarefas de uma mesma seed como réplicas independentes.
- Não comparar mecanismos com inicializações, batches, memória ou budgets de tuning diferentes.

## 5. Problemas de organização

- `split_mnist.py`, `split_clinc150.py`, `slow_heat.py`, `bert.py`, `optim.py`, `show_results.py` e `run_all_tests.py` concentram responsabilidades demais.
- `_holm_adjust` aparece em três módulos; `_json_matrix` em três runners.
- Métodos/seções são enumerados em `split_mnist.py`, `split_mnist_suite.py`, `sections.py` e `run_all_tests.py`.
- Os CLIs não existem no wheel.
- `README.md` tem 659 linhas e duplica protocolos já documentados em `docs/`.
- `run_all_tests.py` sugere teste unitário, mas inicia experimentos caros.
- `tmp/` e `output/` estão ignorados, porém arquivos grandes continuam rastreados; fontes e derivados do deck estão misturados.
- Configs JSON não compartilham `schema_version` e validação versionada.
- Não há markers claros para testes `nlp`, `integration`, `slow` e `gpu`.

## 6. Problemas de performance

- `.item()` em caminhos de forward/backward pode sincronizar CPU/GPU.
- `prepare_fast_heat()` ordena todos os heats em cada forward BERT; `topk`/`kthvalue` evita sort integral.
- `SlowHeatAdamW` cria temporários do tamanho do parâmetro e dos momentos para máscaras parciais.
- Trackers de atenção retêm Q/K/V/output até backward.
- `SlowHeatMLP.forward_features()` materializa `list(self.children())` por chamada.
- Polling da telemetria é O(n) por requisição e tende a O(n²) durante uma execução longa.
- O catálogo de snapshots lê vetores completos quando precisa apenas de metadados.
- A seleção/adaptação e fases auxiliares não entram uniformemente na contabilidade de custo.

Toda otimização deve ser precedida por medição. Não trocar algoritmo científico por uma versão supostamente mais rápida sem teste de equivalência numérica.

## 7. Arquitetura proposta

Adotar quatro limites claros:

1. **Mechanism layer:** `src/dual_heater/` contém apenas estados, gates, camadas, modelos, máscaras, métricas e contratos serializáveis.
2. **Protocol layer:** `src/dual_heater_experiments/` contém datasets, método specs, runners, seleção, checkpoints, estatística e CLI instalável.
3. **Compatibility layer:** `experiments/*.py` e scripts raiz reexportam APIs antigas durante uma versão, sem lógica nova.
4. **Evidence layer:** `configs/` contém protocolos versionados; cada run escreve identidade, matrizes cruas, custos e ambiente; `artifacts/` guarda somente snapshots científicos pequenos e deliberadamente revisados.

Fluxo por run:

```text
config versionada
  -> materializar dados e stream_id
  -> construir plano pareado (init, batches, memória)
  -> executar um método
  -> checkpoint atômico + RNG + custo acumulado
  -> matriz A[stage, task]
  -> agregação pareada por stream_id
  -> relatório sem claims acima da evidência
```

## 8. Estrutura de diretórios proposta

```text
src/
  dual_heater/
    __init__.py
    state.py                    # estado FP32/int64 e serialização científica
    fast_heat.py
    slow_heat.py                # fachada temporária
    slow_heat_core.py           # importância, consolidação e capacity budget
    slow_heat_layers.py         # Linear/Conv2d/trackers
    models/
      __init__.py
      mlp.py
      cnn.py
      vgg.py
      resnet.py
    legacy/
      __init__.py
      dual_heat.py
      lora.py
    optim.py
    metrics.py
    bert.py                     # fachada temporária
    bert_protocol.py
    bert_lora.py
  dual_heater_experiments/
    __init__.py
    artifacts.py
    validation.py
    statistics.py
    method_registry.py
    replay.py
    visual/
      data.py
      config.py
      train.py
      aggregate.py
      cli.py
    clinc/
      data.py
      config.py
      train.py
      aggregate.py
      cli.py
experiments/                    # fachadas compatíveis, remover só em major release
configs/
  schemas/
  synthetic/
  visual/
  clinc/
tests/
  unit/
  integration/
  nlp/
  packaging/
docs/
  methods/
  protocols/
  results/
  assets/presentations/
artifacts/                      # somente evidência pequena, imutável e citada
tmp/                            # sempre derivado, nunca rastreado
```

Não criar essa árvore inteira em um commit. Cada extração só ocorre após o comportamento correspondente estar coberto e verde.

## 9. Refatorações recomendadas

Ordem obrigatória:

1. Corrigir EWC, Classifier Expander, replay e CLINC sem mover arquivos.
2. Fechar contratos de estado científico, dtype, competição e máscaras.
3. Centralizar helpers puros (`statistics.py`, serialização de matriz, validação).
4. Tornar o registro de métodos a única fonte para runner, seções, dry-run e relatórios.
5. Extrair treino visual e CLINC mantendo fachadas.
6. Tornar o pacote de experimentos instalável e validar wheel fora do checkout.
7. Isolar legado sob namespace explícito, mantendo aliases depreciados.
8. Só então dividir modelos SlowHeat/BERT.
9. Limpar artefatos gerados depois de produzir inventário aprovado.

Por quê: mover código antes de corrigir os contratos aumenta o diff, dificulta atribuir regressões e pode fazer bugs científicos sobreviverem com novos nomes.

## 10. Métodos alternativos

Comparações mínimas por família:

- Sem memória: vanilla, `slowheat_none`, reduced LR, EWC correto, SI, LwF, SlowHeat, FastHeat-only e Functional DualHeat.
- Com memória: replay, DER++, ER-ACE, A-GEM, LPR, SlowHeat+cada baseline e controles com o mesmo conteúdo de memória.
- Upper bound: joint/offline training com o mesmo backbone e orçamento documentado.
- Importância: `|z*dL/dz|`, magnitude de ativação, grad² por unidade, ranking aleatório fixo e máscara uniforme com mesma escala/budget.
- Consolidação: MAX e mean calibrados separadamente; `sum` somente se sua escala for normalizada ou calibrada.
- Legado: zero-strength, activation, sensitivity e janela, cruzados com SGD e AdamW.

Não declarar uma alternativa correta por intuição. Definir contraste, métrica, seeds de tuning e critério de decisão antes da execução.

## 11. Experimentos sugeridos

### E1. Fonte da importância

- Contraste: funcional vs ativação vs grad² vs aleatória vs uniforme.
- Parear inicialização, batches, quantidade de unidades protegidas e optimizer.
- Endpoints: FAA, forgetting, aquisição diagonal, correlação do ranking com aumento de loss após ablação.

### E2. Componentes SlowHeat

- Fatorial reduzido: `{MAX, mean} x {row-only, fatorada} x {budget 0.25, sem budget} x {follow_update, native}`.
- Calibrar beta separadamente por consolidação.
- Registrar norma do update por camada e distribuição das máscaras.

### E3. FastHeat

- SlowHeat, FastHeat-only, DualHeat persistente, reset por tarefa, train-only e gate identidade.
- Cruzamento com largura e overlap de tarefas.
- Endpoints mecanísticos: entropia de uso, overlap top-k, unidades mortas, drift de features e margem old/new.

### E4. Legado vs Functional

- Parear vanilla, legado zero, legado activation/sensitivity, Functional SlowHeat e Functional DualHeat.
- Cruzar SGD/AdamW.
- Medir `||delta aplicado|| / ||delta nativo||` por camada e quantil de heat.

### E5. Loss usada na importância

- Manter a loss de atualização fixa.
- Estimar importância com loss total, CE atual, replay-only, média balanceada e passagem separada na memória.

### E6. Replay fixo vs adaptativo

- Braço A: índices congelados comuns por `stream_id`.
- Braço B: seleção específica do learner, declarada como parte do tratamento.
- Registrar Jaccard entre seleções e decompor interação método x seletor.

### E7. Cabeça global

- Sem calibração, offset old/new validado, balanced softmax pré-especificado e head-only rehearsal.
- Endpoints: Class-IL, task-aware, classifier gap, macro-F1, ECE e bias de logit.

### E8. Robustez a fronteiras

- Fronteira correta, atraso de 10%/25%, fronteira espúria e consolidação periódica.
- Pelo menos cinco ordens held-out.

### Desenho estatístico

- Screening: 5 streams por configuração.
- Tuning: 10 streams disjuntos.
- Final: pelo menos 20 streams held-out, distribuídos em cinco ordens.
- Unidade pareada: `dataset + task_order + data_seed + initialization_seed`.
- Reportar diferenças pareadas, IC95%, taxa de sinais positivos e Holm apenas na família primária pré-declarada.
- Persistir matriz completa `A[stage, task]`, não apenas métricas finais.

Critério inicial a congelar antes dos resultados:

- avançar se `delta FAA >= +1.0 pp`, limite inferior do IC95% > 0 e pelo menos 70% dos streams positivos;
- aceitar ganho de retenção apenas se `delta forgetting <= -1.0 pp` sem queda de aquisição > 1.0 pp;
- para ganho de custo, margem de não inferioridade de FAA de -0.5 pp;
- fora de `tempo <= 1.5x` e `pico <= 1.2x`, reportar Pareto, não vencedor único.

## 12. Ideias que a implementação atual ainda não explora

- Upper bound joint/offline no mesmo harness.
- MAS e importance por ablação causal aproximada.
- Fronteiras ruidosas ou ausentes.
- Múltiplas ordens de tarefa como eixo explícito, não só seeds.
- Decomposição sistemática de forgetting representacional versus descalibração da cabeça.
- Estimador de importância separado da loss usada para atualizar pesos.
- Máscara aleatória com distribuição exatamente casada, controle necessário para testar seletividade.
- Curvas de Pareto FAA, memória, wall time e pico de VRAM.
- Teste de reabertura de unidades e dinâmica dos momentos AdamW após longos períodos protegidos.
- Joint effect de FastHeat com capacidade/largura, em vez de apenas um ponto de arquitetura.
- Protocolo sem fronteira oracle para comparar honestamente com o legado online.

## 13. Plano incremental de implementação

### Convenções de execução

Antes de cada tarefa de código:

1. Escrever um teste que falha pelo motivo esperado.
2. Executar somente o teste alvo com CPU, GPU invisível, uma thread e sem cache/bytecode.
3. Implementar o mínimo.
4. Executar o teste alvo e depois a suíte CPU.
5. Executar Ruff nos arquivos tocados.
6. Fazer um commit pequeno.

Prefixo padrão:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
python -m pytest -q -p no:cacheprovider
```

Não executar downloads, treinamento GPU ou suítes experimentais caras durante estas tarefas.

### Fase A. Congelar o baseline antes das correções

#### Tarefa A1: registrar o baseline local

**Objetivo:** saber se qualquer falha posterior é nova.

**Arquivos:** nenhum.

**Passos:**

1. Executar:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
python -m pytest -q -p no:cacheprovider
```

Esperado: exit code 0; nenhum `failed` ou `error`. Testes NLP podem aparecer como `skipped` se extras não estiverem instalados.

2. Executar:

```bash
ruff check .
git status --short
```

Esperado: Ruff com exit code 0; `git status --short` vazio.

Não criar commit para uma tarefa sem mudança.

### Fase B. Corrigir validade científica P0 no runner visual

#### Tarefa B1: extrair acumulador correto de Fisher empírico

**Objetivo:** quadrar gradientes por exemplo antes da média.

**Arquivos:**

- Modificar: `experiments/split_mnist.py`
- Testar: `tests/test_split_mnist.py`

**RED:** adicionar ao import de `experiments.split_mnist` o símbolo `_accumulate_empirical_fisher` e inserir:

```python
def test_empirical_fisher_squares_per_example_before_averaging():
    model = torch.nn.Linear(1, 2, bias=False)
    torch.nn.init.zeros_(model.weight)
    inputs = torch.ones(2, 1)
    targets = torch.tensor([0, 1])
    logits = model(inputs)
    accumulator = {
        name: torch.zeros_like(parameter)
        for name, parameter in model.named_parameters()
    }

    sample_count = _accumulate_empirical_fisher(
        model,
        logits,
        targets,
        accumulator,
    )

    assert sample_count == 2
    assert accumulator["weight"] == pytest.approx(
        torch.full_like(model.weight, 0.5)
    )
    estimate = accumulator["weight"] / sample_count
    assert estimate == pytest.approx(torch.full_like(model.weight, 0.25))
```

Executar:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
python -m pytest tests/test_split_mnist.py::test_empirical_fisher_squares_per_example_before_averaging -q -p no:cacheprovider
```

Esperado RED: erro de import porque `_accumulate_empirical_fisher` ainda não existe.

**GREEN:** adicionar perto de `_parameter_penalty`:

```python
def _accumulate_empirical_fisher(
    model: nn.Module,
    logits: Tensor,
    targets: Tensor,
    accumulator: dict[str, Tensor],
) -> int:
    """Accumulate sum_i grad(log p(y_i|x_i))**2 for one batch."""

    named_parameters = tuple(model.named_parameters())
    parameters = tuple(parameter for _, parameter in named_parameters)
    for sample_index in range(len(targets)):
        sample_loss = F.cross_entropy(
            logits[sample_index : sample_index + 1],
            targets[sample_index : sample_index + 1],
            reduction="sum",
        )
        gradients = torch.autograd.grad(
            sample_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        for (name, _), gradient in zip(named_parameters, gradients, strict=True):
            if gradient is not None:
                accumulator[name].add_(gradient.detach().square())
    return len(targets)
```

Executar novamente o teste alvo. Esperado GREEN: `1 passed`.

Commit:

```bash
git add experiments/split_mnist.py tests/test_split_mnist.py
git commit -m "fix: compute empirical fisher per example"
```

#### Tarefa B2: integrar o Fisher por número de exemplos

**Objetivo:** remover `fisher_steps` e ponderar corretamente o último minibatch.

**Arquivos:** mesmos da B1.

**RED:** adicionar teste com 3 exemplos e `batch_size=2`; comparar a importância EWC produzida pelo runner a uma soma manual por exemplo. O teste deve acessar um helper de consolidação puro, não arquivos de resultado.

**Implementação mínima:** substituir o bloco `torch.autograd.grad(current_loss, ...)` por:

```python
if method == "ewc":
    fisher_examples += _accumulate_empirical_fisher(
        model,
        current_logits,
        current_y,
        fisher_sum,
    )
    cost["learner_backward_examples"] += current_count
```

Inicializar `fisher_examples = 0` no início do estágio e consolidar com:

```python
estimate = fisher_sum[name] / max(1, fisher_examples)
```

Executar teste alvo e `tests/test_split_mnist.py`. Esperado: todos passam.

Commit:

```bash
git add experiments/split_mnist.py tests/test_split_mnist.py
git commit -m "fix: weight ewc fisher by sample count"
```

#### Tarefa B3: isolar destilação do Classifier Expander às classes antigas

**Objetivo:** impedir que logits aleatórios das classes novas entrem na loss do professor.

**Arquivos:**

- Modificar: `experiments/split_mnist.py`
- Testar: `tests/test_split_mnist.py`

**RED:** expor um helper puro e testar invariância a classes novas:

```python
def test_classifier_expander_distillation_ignores_new_classes():
    student = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    teacher_a = torch.tensor([[1.5, 1.0, -100.0, 100.0]])
    teacher_b = torch.tensor([[1.5, 1.0, 100.0, -100.0]])

    loss_a = _old_class_distillation_loss(student, teacher_a, (0, 1))
    loss_b = _old_class_distillation_loss(student, teacher_b, (0, 1))

    assert torch.equal(loss_a, loss_b)
```

Esperado RED: import ausente.

**GREEN:** adicionar:

```python
def _old_class_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    old_classes: tuple[int, ...],
) -> Tensor:
    if not old_classes:
        return student_logits.new_zeros(())
    old_index = torch.tensor(
        old_classes,
        device=student_logits.device,
        dtype=torch.long,
    )
    return F.mse_loss(
        student_logits.index_select(1, old_index),
        teacher_logits.index_select(1, old_index),
    )
```

No runner, substituir a construção de `seen_index` e o `F.mse_loss` por:

```python
loss = loss + (
    config.classifier_expander_distillation_weight
    * _old_class_distillation_loss(
        replay_logits,
        teacher_logits,
        old_classes,
    )
)
```

Executar o teste alvo, todo `tests/test_split_mnist.py` e Ruff. Esperado: verde.

Commit:

```bash
git add experiments/split_mnist.py tests/test_split_mnist.py
git commit -m "fix: distill only old classifier classes"
```

#### Tarefa B4: tornar a semântica de replay explícita

**Objetivo:** nunca chamar seleção learner-adaptive de replay pareado/idêntico.

**Arquivos:**

- Modificar: `experiments/split_mnist.py`, `experiments/dualheat_pairs.py`
- Testar: `tests/test_split_mnist.py`, `tests/test_replay_memory.py`

**RED:** para `replay_selection="loss"`, exigir `replay_indices_paired is False` no resultado e rejeitar o modo em `run_dualheat_pairs`; para `first`, exigir `True`.

**GREEN:** centralizar:

```python
def replay_selection_is_method_independent(strategy: str) -> bool:
    return strategy == "first"
```

Adicionar ao payload de cada método:

```python
"replay_selection": config.replay_selection,
"replay_indices_paired": replay_selection_is_method_independent(
    config.replay_selection
),
```

Na suíte que promete pares isolados:

```python
if not replay_selection_is_method_independent(config.replay_selection):
    raise ValueError(
        "a suíte pareada requer replay_selection='first'; "
        "seleção adaptativa pertence ao tratamento experimental"
    )
```

Não implementar ainda um seletor comum oculto. Primeiro impedir claims falsos; o experimento E6 decidirá seletor comum versus learner-adaptive.

Verificação:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
python -m pytest tests/test_replay_memory.py tests/test_split_mnist.py -q -p no:cacheprovider
```

Esperado: nenhum failure.

Commit:

```bash
git add experiments/split_mnist.py experiments/dualheat_pairs.py tests/test_split_mnist.py tests/test_replay_memory.py
git commit -m "fix: label adaptive replay as unpaired"
```

### Fase C. Fechar o protocolo CLINC

#### Tarefa C1: desativar teste CLINC por padrão

**Arquivos:** `experiments/split_clinc150.py`, `tests/test_split_clinc150.py`.

**RED:** adicionar:

```python
def test_clinc_defaults_to_validation_only():
    assert SplitCLINC150Config().evaluate_test is False
```

Esperado RED: assertion failure, pois o default atual é `True`.

**GREEN:** trocar o default para:

```python
evaluate_test: bool = False
```

Executar teste alvo. Esperado: `1 passed`.

Commit: `fix: keep clinc test split closed by default`.

#### Tarefa C2: adicionar flags explícitas e fail-closed

**Objetivo:** teste só pode ser lido por execução explicitamente confirmatória.

**Arquivos:** `experiments/split_clinc150.py`, `tests/test_split_clinc150.py`.

**RED:** testar parser default false, `--evaluate-test` true e rejeição sem `--frozen-manifest`.

**GREEN:** no parser:

```python
parser.add_argument(
    "--evaluate-test",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="avaliar o split de teste; requer --frozen-manifest",
)
```

Antes de carregar dados/modelo:

```python
if args.evaluate_test and args.frozen_manifest is None:
    parser.error("--evaluate-test requer --frozen-manifest")
```

Construir `SplitCLINC150Config(..., evaluate_test=args.evaluate_test)` explicitamente. Não depender de default implícito.

Verificar:

```bash
PYTHONPATH=src:. python -m experiments.split_clinc150 --help
```

Esperado: `--evaluate-test | --no-evaluate-test` listado; nenhum download iniciado.

Commit: `fix: require frozen clinc manifest for test access`.

#### Tarefa C3: validar identidade antes de escrever `protocol.json`

**Objetivo:** uma tentativa incompatível não sobrescreve o protocolo existente.

**Arquivos:** `experiments/split_clinc150.py`, `experiments/artifacts.py`, `tests/test_split_clinc150.py`.

**RED:** criar `protocol.json` incompatível em `tmp_path`, chamar resume e afirmar que ocorre `RuntimeError` e o conteúdo original permanece byte a byte igual.

**GREEN:** reutilizar `source_fingerprint` e `read_json_object`:

```python
identity = {
    **_checkpoint_identity(config, metadata, data_sha256),
    "source_sha256": source_fingerprint(Path(__file__).resolve().parents[1]),
}
protocol_path = None if destination is None else destination / "protocol.json"
if protocol_path is not None and protocol_path.exists():
    existing = read_json_object(protocol_path)
    if existing != identity:
        raise RuntimeError("output CLINC150 pertence a outro protocolo")
if protocol_path is not None:
    write_json_atomic(protocol_path, identity)
```

Importar `read_json_object` e `source_fingerprint` de `experiments.artifacts`.

Executar o teste alvo. Esperado: passa e arquivo original não muda.

Commit: `fix: bind clinc resume to source identity`.

#### Tarefa C4: persistir/restaurar RNG no checkpoint CLINC

**Objetivo:** retomada em fronteira de tarefa ser numericamente equivalente com dropout real.

**Arquivos:** `experiments/split_clinc150.py`, `tests/test_split_clinc150.py`.

**RED:** adaptar o tiny BERT para dropout `0.1`; introduzir um hook de teste `stop_after_stage=0` ou monkeypatch de escrita que interrompa depois do primeiro checkpoint; comparar run contínuo e retomado por losses e todos os tensores do modelo. O teste atual, com dropout zero e resume de run já concluído, não é suficiente.

**GREEN:** salvar:

```python
"python_rng_state": random.getstate(),
"numpy_rng_state": np.random.get_state(),
"torch_rng_state": torch.get_rng_state(),
"cuda_rng_states": (
    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
),
```

Após carregar checkpoint e antes do próximo forward:

```python
random.setstate(checkpoint["python_rng_state"])
np.random.set_state(checkpoint["numpy_rng_state"])
torch.set_rng_state(checkpoint["torch_rng_state"])
cuda_rng_states = checkpoint.get("cuda_rng_states")
if cuda_rng_states is not None and torch.cuda.is_available():
    torch.cuda.set_rng_state_all(cuda_rng_states)
```

Incrementar `CHECKPOINT_SCHEMA_VERSION`; rejeitar v1 com mensagem de migração, sem adivinhar RNG ausente.

Verificação alvo: o teste de interrupção passa bit a bit. Depois executar ambos os arquivos NLP:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
python -m pytest tests/test_split_clinc150.py tests/test_slow_heat_bert.py -q -p no:cacheprovider
```

Esperado: verde quando extras NLP estiverem instalados; se estiverem ausentes, registrar `skipped` e deixar a validação NLP para a matriz dedicada, sem instalar dependências sem aprovação.

Commit: `fix: restore clinc rng state on resume`.

#### Tarefa C5: scheduler por tarefa como decisão experimental explícita

**Objetivo:** remover confusão entre posição da tarefa e LR.

**Arquivos:** `experiments/split_clinc150.py`, `tests/test_split_clinc150.py`, `docs/functional_slowheat_transformers.md`.

**RED:** helper de scheduler deve produzir a mesma sequência relativa para duas tarefas de mesmo tamanho e reiniciar no LR inicial.

**GREEN:** adicionar `scheduler_scope: Literal["task", "stream"] = "task"` à config; construir scheduler no início de cada estágio quando `task`, persistir scope e estado do scheduler no checkpoint. Manter `stream` apenas como ablação reproduzível dos resultados históricos.

Esperado: teste de sequência passa; `protocol.json` contém `scheduler_scope`.

Commit: `feat: make clinc scheduler scope explicit`.

### Fase D. Endurecer estado e configuração dos mecanismos

#### Tarefa D1: rejeitar `global_topk` sem coordenador

**Objetivo:** impedir execução silenciosa de `mean_others`.

**Arquivos:** `src/dual_heater/fast_heat.py`, `tests/test_fast_heat.py`.

**RED:** adicionar:

```python
def test_global_topk_requires_external_global_scale():
    gate = FastHeatGate(
        4,
        unit_dim=-1,
        config=FastHeatConfig(competition="global_topk"),
    )
    with pytest.raises(RuntimeError, match="escala global"):
        gate(torch.ones(2, 4))
```

Esperado RED: nenhuma exceção.

**GREEN:** em `current_scale()`:

```python
def current_scale(self) -> Tensor:
    if self._external_scale is not None:
        return self._external_scale
    if self.config.competition == "global_topk":
        raise RuntimeError(
            "global_topk requer escala global preparada pelo coordenador do modelo"
        )
    return self._lateral_scale()
```

Adicionar teste BERT existente para garantir que `prepare_fast_heat()` continua fornecendo a escala.

Commit: `fix: reject uncoordinated global topk fastheat`.

#### Tarefa D2: preservar buffers científicos em FP32 e contadores em int64

**Objetivo:** separar dtype computacional de estado do mecanismo.

**Arquivos:**

- Criar: `src/dual_heater/state.py`
- Modificar: `fast_heat.py`, `slow_heat.py`, `dual_heat.py`, `lora.py`, `transformer.py`
- Testar: `tests/test_fast_heat.py`, `test_slow_heat.py`, `test_dual_heat.py`, `test_lora.py`.

**RED:** parametrizar `.half()` e, se suportado, `.bfloat16()`; afirmar que pesos mudam de dtype, `fast_heat/slow_heat/task_ema/importance_memory` permanecem `torch.float32` e contadores permanecem `torch.int64`. Executar mais de 2.048 atualizações no contador legado e afirmar valor exato.

**GREEN:** usar um mixin único:

```python
from __future__ import annotations

from torch import nn


class FP32ScientificStateMixin(nn.Module):
    """Keep named scientific buffers FP32 while preserving device transforms."""

    _fp32_state_names: tuple[str, ...] = ()

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        for name in self._fp32_state_names:
            buffer = getattr(self, name, None)
            if buffer is not None and buffer.is_floating_point():
                setattr(self, name, buffer.float())
        return result
```

Cada módulo declara somente os nomes que controla. Registrar contadores com:

```python
self.register_buffer("slow_n", torch.zeros((), dtype=torch.int64))
```

Não manter assinatura BERT como vetor float sujeito a cast; persistir configuração em metadado JSON versionado e validar no construtor/checkpoint.

Verificar testes alvo, depois suíte CPU. Commit: `fix: preserve scientific state precision`.

#### Tarefa D3: fazer `clear_plasticity_masks()` falhar fechado

**Objetivo:** não permitir `optimizer.step()` desprotegido após limpar registros.

**Arquivos:** `src/dual_heater/optim.py`, `tests/test_optim.py`.

**RED:** registrar máscara, limpar, atribuir gradiente e exigir `RuntimeError` no próximo `step()`.

**GREEN:** preservar assinaturas esperadas antes de limpar:

```python
def clear_plasticity_masks(self) -> None:
    """Remove live bindings and require explicit re-registration before step."""

    current = self._current_mask_signatures()
    if current:
        self._expected_mask_signatures = current
    self._plasticity_masks.clear()
```

O `_ensure_checkpoint_masks_registered()` existente então bloqueia o step. Adicionar método separado `disable_plasticity_masks_for_ablation()` somente se houver requisito real; não criar agora por YAGNI.

Commit: `fix: fail closed after clearing plasticity masks`.

#### Tarefa D4: validar `ReplayBuffer` e configurações primitivas

**Arquivos:** `experiments/replay_memory.py`, `experiments/validation.py`, `tests/test_replay_memory.py`, `tests/test_synthetic_experiment.py`.

**RED:** cobrir `True`, `1.5`, NaN, `inputs=None` com targets não vazios e `score_components` com segunda dimensão diferente do contrato.

**GREEN:** substituir helpers por:

```python
def require_positive_integers(values: Mapping[str, int]) -> None:
    for name, value in values.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} deve ser um inteiro >= 1")


def require_nonnegative_values(values: Mapping[str, float]) -> None:
    invalid = {
        name: value
        for name, value in values.items()
        if not math.isfinite(value) or value < 0.0
    }
    if invalid:
        raise ValueError(f"parâmetros devem ser finitos e >= 0: {invalid}")
```

Em `ReplayBuffer.load_state_dict()`:

```python
if (inputs is None) != (count == 0):
    raise ValueError("inputs ausentes não correspondem a ReplayBuffer vazio")
if tensor_fields["score_components"].ndim != 2:
    raise ValueError("score_components deve ser uma matriz")
```

Se o número de componentes é fixo, validar a largura contra a constante usada em `append()`.

Commit: `fix: validate experiment state without coercion`.

### Fase E. Fechar pré-registros e seleção

#### Tarefa E1: schema obrigatório e digest do pré-registro Split-MNIST

**Arquivos:** `experiments/confirmatory_split_mnist.py`, `configs/split_mnist_confirmation_preregistration.json`, `tests/test_confirmatory_statistics.py` ou novo `tests/test_confirmatory_protocol.py`.

**RED:** remover cada campo congelado em cópia temporária e exigir erro específico.

**GREEN:** declarar `REQUIRED_FROZEN_PATHS` em código e um digest esperado calculado sobre payload canônico. Não derivar campos obrigatórios das chaves presentes no arquivo. O teste deve cobrir campo ausente, extra incompatível e valor alterado.

Commit: `fix: make split mnist preregistration immutable`.

#### Tarefa E2: vincular manifesto FastHeat aos resultados do piloto

**Arquivos:** `experiments/functional_dualheat.py`, testes em `tests/test_fast_heat.py`.

**RED:** manifesto com candidato válido da grade, mas não vencedor, deve falhar; alteração de um resultado deve invalidar hash.

**GREEN:** manifesto inclui schema, seeds, arquiteturas, objetivo, caminhos relativos e hashes dos resultados. Loader lê resultados, verifica hashes, chama `select_fastheat_candidate()` e exige igualdade exata com `selected`.

Commit: `fix: verify frozen fastheat selection provenance`.

### Fase F. Centralizar contratos puros sem mudar comportamento

#### Tarefa F1: extrair estatística duplicada

**Arquivos:** criar `experiments/statistics.py`; modificar `dualheat_pairs.py`, `functional_dualheat.py`, `replay_selection_sweep.py`; criar `tests/test_statistics.py`.

**RED:** testes de Holm com empates, ordem original, valores 0/1 e NaN rejeitado.

**GREEN:** mover uma única implementação, manter aliases importados nos módulos antigos por compatibilidade e provar resultados byte a byte nos testes existentes.

Commit: `refactor: centralize paired statistics`.

#### Tarefa F2: extrair serialização de matrizes

**Arquivos:** criar `experiments/serialization.py`; modificar `split_mnist.py`, `split_clinc150.py`, `synthetic_cl.py`; criar `tests/test_serialization.py`.

**RED:** matriz triangular com NaN deve produzir `None` somente onde ausente e rejeitar infinito.

**GREEN:** mover `_json_matrix` para função pública única, manter aliases temporários.

Commit: `refactor: centralize matrix serialization`.

#### Tarefa F3: tornar `method_specs.py` fonte única

**Arquivos:** `experiments/method_specs.py`, `split_mnist.py`, `split_mnist_suite.py`, `sections.py`, `run_all_tests.py`, testes de benchmark/dry-run.

**RED:** teste percorre todas as seções e afirma que todo método existe exatamente uma vez no registry e que o dry-run deriva dele.

**GREEN:** adicionar dataclass `MethodSpec` com capacidades necessárias (`replay`, `derpp`, `slowheat`, arquitetura compatível, budget, custo extra). Remover listas duplicadas uma por commit, começando pela menor.

Commits separados:

```text
refactor: derive visual methods from registry
refactor: derive benchmark sections from registry
refactor: derive dry run output from registry
```

### Fase G. Separar runners mantendo fachadas

Cada extração abaixo é uma tarefa independente e deve preservar imports antigos.

1. `experiments/visual_data.py`: mover dataclasses/loaders e reexportar de `split_mnist.py`.
2. `experiments/visual_config.py`: mover `SplitMNISTConfig` e payload.
3. `experiments/visual_train.py`: mover treino de uma seed somente após P0 verde.
4. `experiments/visual_aggregate.py`: mover resume/agregação.
5. `experiments/clinc_data.py`: mover domínio, dataclasses e tokenização.
6. `experiments/clinc_config.py`: mover config/manifesto.
7. `experiments/clinc_train.py`: mover execução de um método/seed.
8. `experiments/clinc_aggregate.py`: mover estatística multi-seed.
9. `experiments/clinc_cli.py`: mover parser/main.

Para cada slice:

- RED: teste de import e uma chamada mínima pela interface nova.
- GREEN: mover apenas símbolos daquele slice.
- Compatibilidade: `from experiments.visual_data import ...` no módulo antigo.
- Verificação: teste alvo, arquivo de testes correspondente, suíte CPU e Ruff.
- Commit: um `refactor:` por slice.

Não alterar nomes de métodos, schema de artefatos ou resultados numéricos durante esta fase.

### Fase H. Packaging e CI

#### Tarefa H1: pacote instalável de experimentos

Mover módulos extraídos para `src/dual_heater_experiments/` apenas depois da Fase G. Manter `experiments/` como fachadas. Adicionar em `pyproject.toml`:

```toml
[project.scripts]
dual-heater-benchmark = "dual_heater_experiments.visual.cli:main"
dual-heater-clinc = "dual_heater_experiments.clinc.cli:main"
dual-heater-results = "dual_heater_experiments.results.cli:main"
```

Criar `tests/test_packaging.py` que constrói wheel, instala em venv temporária e executa imports/`--help` fora do checkout, sem `PYTHONPATH`.

Verificação:

```bash
python -m build
python -m twine check dist/*
```

Esperado: sdist e wheel criados; `twine check` retorna `PASSED` para ambos.

Commit: `build: package research command line tools`.

#### Tarefa H2: renomear interface perigosa sem quebrar compatibilidade

Criar `run_benchmarks.py`; transformar `run_all_tests.py` em shim que emite `DeprecationWarning` e delega. README usa o nome novo. Teste garante que `--dry-run` é idêntico.

Commit: `refactor: rename experiment orchestrator`.

#### Tarefa H3: matriz CI core e NLP

Modificar `.github/workflows/ci.yml`:

- job core: Python 3.10 e 3.13, pytest, `ruff check`, `ruff format --check`, build/wheel smoke;
- job NLP: uma versão de Python com `.[dev,nlp]`, somente testes BERT/CLINC;
- nenhum job baixa dataset ou usa GPU.

Adicionar markers em `pyproject.toml` antes de usá-los.

Esperado em PR: jobs `core (3.10)`, `core (3.13)` e `nlp` verdes.

### Fase I. Limpeza de artefatos, somente com aprovação explícita

1. Gerar inventário de `tmp/` e `output/` separando fonte, entrega e derivado.
2. Preservar fontes do deck em `docs/assets/presentations/`.
3. Preservar PDFs finais apenas se forem entregáveis deliberados.
4. Remover do índice Git renders, contact sheets, inspeções e binários duplicados.
5. Não reescrever histórico e não tocar `artifacts/synthetic_ablation_pilot/`.

Esta fase é destrutiva no índice Git e exige aprovação específica antes da execução.

### Fase J. Experimentos, somente depois dos gates de correção

1. Rodar smoke CPU com 1 seed para validar artefatos e matriz, não eficácia.
2. Rodar screening de E1-E8 com seeds de desenvolvimento.
3. Congelar configs, contrastes e critérios.
4. Solicitar aprovação para GPU/datasets.
5. Executar tuning e final em pools disjuntos.
6. Auditar cada run identity e matriz antes de agregar.
7. Atualizar documentação com resultados negativos e positivos, sem cherry-pick.

Nenhuma execução anterior à correção EWC/CLINC pode ser promovida como nova evidência confirmatória.

## 14. Prioridades: impacto x esforço

| Prioridade | Mudança | Impacto | Esforço | Decisão |
|---|---|---:|---:|---|
| P0 | Fisher EWC por exemplo | Muito alto | Médio | Fazer primeiro. |
| P0 | Destilar só classes antigas | Muito alto | Baixo | Fazer imediatamente após Fisher. |
| P0 | Teste CLINC fechado por default | Muito alto | Baixo | Fazer antes de qualquer CLINC. |
| P0 | RNG + source identity no resume CLINC | Muito alto | Médio | Obrigatório antes de retomar runs. |
| P0 | Claims corretos para replay adaptativo | Alto | Baixo | Fazer antes de relatórios pareados. |
| P0 | Pré-registro e manifesto FastHeat verificáveis | Alto | Médio | Fazer antes de confirmação. |
| P0 | Rejeitar top-k sem coordenador | Alto | Baixo | Evita método rotulado incorretamente. |
| P0 | Estado FP32/contadores int64 | Alto | Médio | Fazer antes de mixed precision. |
| P1 | Máscara fail-closed | Alto | Baixo | Evita treino silenciosamente desprotegido. |
| P1 | Scheduler CLINC explícito | Alto | Médio | Necessário para comparar ordens. |
| P1 | Validadores e ReplayBuffer | Médio | Baixo | Robustez barata. |
| P1 | Custo completo e resume | Médio | Médio | Antes de claims de eficiência. |
| P2 | Registry/helpers únicos | Médio | Médio | Reduz divergência após correções. |
| P2 | Separar runners | Médio | Alto | Fazer em slices, sem rewrite. |
| P2 | Packaging + CI NLP | Médio | Médio | Torna a base instalável e verificável. |
| P3 | Telemetria indexada | Baixo para ciência | Médio | Só após medir gargalo. |
| P3 | Limpeza de binários | Baixo para correção | Baixo | Exige aprovação por ser destrutivo. |

## 15. Mudanças concretas de código

### Arquivos a modificar no primeiro milestone

```text
experiments/split_mnist.py
experiments/dualheat_pairs.py
experiments/split_clinc150.py
experiments/replay_memory.py
experiments/validation.py
experiments/confirmatory_split_mnist.py
experiments/functional_dualheat.py
src/dual_heater/state.py                     # novo
src/dual_heater/fast_heat.py
src/dual_heater/slow_heat.py
src/dual_heater/dual_heat.py
src/dual_heater/lora.py
src/dual_heater/transformer.py
src/dual_heater/optim.py
tests/test_split_mnist.py
tests/test_split_clinc150.py
tests/test_replay_memory.py
tests/test_fast_heat.py
tests/test_slow_heat.py
tests/test_dual_heat.py
tests/test_lora.py
tests/test_optim.py
```

### Contratos observáveis depois do primeiro milestone

- EWC armazena `sum_i(g_i**2) / N`, não `(mean_i g_i)**2`.
- Classifier Expander ignora logits das classes atuais na distilação do professor antigo.
- Resultado informa se os índices de replay são realmente pareados; suíte pareada rejeita seletor adaptativo.
- CLINC não lê teste sem flag explícita e manifesto congelado.
- Resume CLINC rejeita código/config/dados incompatíveis e reproduz run contínuo com dropout.
- `global_topk` sem coordenador falha claramente.
- Heats ficam em FP32; contadores ficam em int64 sob `.half()`/`.bfloat16()`.
- Limpar máscaras obriga novo registro antes de `step()`.
- Pré-registro e escolha FastHeat são verificáveis a partir de schema, digest e artefatos de origem.

### Gate final do milestone

Executar:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
python -m pytest -q -p no:cacheprovider
ruff check .
ruff format --check src experiments tests run_all_tests.py run_dualheat_pairs.py show_results.py
git status --short
```

Esperado:

- pytest com exit code 0 e nenhum failure/error;
- Ruff lint e format com exit code 0;
- `git status --short` vazio depois do último commit;
- nenhum dataset baixado, nenhum treino GPU e nenhum novo resultado científico produzido.

Depois do milestone, atualizar `docs/experiments_audit.md` marcando apenas itens comprovadamente corrigidos e citando os testes de regressão. Não apagar o histórico dos achados.

---

## Risks, tradeoffs, and open questions

1. **Fisher por exemplo é caro.** Primeiro implementar a referência correta. Depois comparar `torch.func.vmap` ou aproximações validadas por equivalência, sem trocar correção por velocidade silenciosamente.
2. **Replay adaptativo pode ser um método legítimo.** O defeito é chamá-lo pareado. Manter dois braços explícitos, comum e learner-adaptive.
3. **Scheduler por tarefa muda o protocolo histórico.** Preservar `stream` como ablação identificada; não misturar artefatos dos dois scopes.
4. **FP32 state sob `_apply` exige testes de device, dtype, `share_memory()` e checkpoint.** Aplicar primeiro a transformação real e desfazer apenas cast de dtype nos buffers científicos.
5. **Exact LoRA com classificador plástico não é proteção fechada.** Manter como método separado ou adicionar variante com cabeça protegida; não alterar silenciosamente o método histórico.
6. **Empacotar experimentos aumenta superfície pública.** Se o objetivo for somente desenvolvimento no checkout, a alternativa é documentar essa limitação e não criar console scripts. A recomendação é empacotar porque a meta inclui reprodução fora do clone original.
7. **Type checking amplo pode gerar trabalho sem ganho científico imediato.** Começar por configs, artefatos e contracts; não bloquear P0 em anotações do runner monolítico.
8. **Limpeza de `tmp/` pode apagar fontes úteis.** Exige inventário e aprovação explícita.
9. **Contagens exatas de testes não foram congeladas.** Verificações devem afirmar exit code e ausência de falhas, pois extras NLP alteram o número de skips/coletas.
10. **Resultados antigos não estão todos disponíveis em JSON.** Não validar números documentados sem recuperar os artefatos originais e conferir identidade; nunca reconstruir outputs plausíveis.
