# Auditoria dos módulos `experiments/*.py`

## Escopo

Foram revisados integralmente os 27 arquivos Python diretamente contidos em
`experiments/`, aproximadamente 10 mil linhas. A revisão cobriu correção
numérica, pareamento experimental, splits, métricas, checkpoints, retomada,
proveniência, estatística, custo computacional, telemetria, segurança de paths e
alcance das opções de CLI.

A auditoria foi somente leitura em relação ao código-fonte. Este relatório é o
único artefato criado pela auditoria. A árvore já continha muitas alterações não
relacionadas.

## Veredito executivo

A infraestrutura é ampla e, em geral, possui bons contratos de configuração,
escrita atômica, matrizes completas de continual learning e testes. Entretanto,
há três problemas que precisam ser resolvidos antes de tratar os resultados como
base de paper:

1. o baseline EWC não calcula o Fisher empírico definido pelo método;
2. seletores adaptativos de replay quebram o pareamento anunciado entre métodos;
3. o runner CLINC/BERT consulta o teste por padrão e não retoma de forma
   numericamente equivalente após uma interrupção.

Também há dois controles de pré-registro que podem ser enfraquecidos por edição
do próprio manifesto, além de problemas de custo, proveniência e telemetria.

## Prioridade 0: afetam diretamente validade científica

### 1. Fisher incorreto no EWC

- Severidade: alta
- Local: `experiments/split_mnist.py:1950-1967,2179-2184`
- Evidência: o código eleva ao quadrado o gradiente da loss média do minibatch.
  Isso calcula `(mean_i g_i)^2`, enquanto o Fisher empírico exige
  `mean_i(g_i^2)`. Gradientes de exemplos diferentes podem se cancelar antes do
  quadrado. O último minibatch também recebe o mesmo peso dos demais.
- Impacto: a importância EWC é subestimada ou distorcida. Comparações
  SlowHeat-versus-EWC não representam fielmente o baseline.
- Correção mínima: calcular gradientes por exemplo da NLL, quadrar antes da
  redução e dividir pelo total de exemplos.

### 2. Replay adaptativo quebra o pareamento anunciado

- Severidade: alta
- Local: `experiments/split_mnist.py:1597-1608,2065-2078`
- Evidência: o docstring afirma replay indices idênticos por seed, mas
  `select_task_exemplars()` recebe o modelo específico de cada método. Nas
  estratégias `loss`, `representative` e `hybrid`, logits/features diferentes
  produzem índices diferentes. Somente `first` independe do modelo.
- Impacto: diferenças entre Replay e SlowHeat+Replay misturam mecanismo e
  composição da memória.
- Correção mínima: usar um seletor congelado comum por seed/estágio e reutilizar
  seus índices, ou declarar seleção adaptativa como parte do tratamento e não
  alegar replay idêntico.

### 3. Classifier Expander destila classes ainda não aprendidas

- Severidade: alta
- Local: `experiments/split_mnist.py:1844-1885`
- Evidência: o índice usado no MSE é `seen_classes`, incluindo classes da tarefa
  atual. O professor foi copiado antes dessa tarefa e possui logits aleatórios
  para as classes novas.
- Impacto: o aluno é forçado a imitar saídas arbitrárias justamente nas classes
  que deveria adquirir.
- Correção mínima: destilar somente `old_classes`.

### 4. Resume do CLINC/BERT não restaura RNG

- Severidade: alta
- Local: `experiments/split_clinc150.py:819-863,917-925,1149-1169`
- Evidência: modelo, optimizer, scheduler e replay são persistidos, mas estados
  RNG de CPU/CUDA não. A ordem dos exemplos é reconstruída, porém dropout usa o
  RNG global. `split_mnist.py` já persiste os RNGs em `:2281-2285`.
- Impacto: uma execução retomada na fronteira de tarefas diverge da execução
  contínua.
- Correção mínima: salvar/restaurar `torch.get_rng_state()` e
  `torch.cuda.get_rng_state_all()` e criar teste de interrupção real após um
  estágio incompleto.

### 5. Teste CLINC é consultado durante desenvolvimento por padrão

- Severidade: alta para uso confirmatório; preocupação de protocolo para uso
  exploratório
- Local: `experiments/split_clinc150.py:194-205,1107-1128,1528-1633`
- Evidência: `evaluate_test=True`, a avaliação ocorre após cada estágio e não há
  opção CLI normal para desligar o teste. O manifesto congelado só é obrigatório
  no caminho BERT-base.
- Impacto: iterações em BERT-Mini e nas novas variantes podem ser escolhidas após
  observar repetidamente o teste.
- Correção mínima: padrão `evaluate_test=False`, opções explícitas
  `--evaluate-test/--no-evaluate-test` e exigência de manifesto para runs
  confirmatórios.

### 6. Identidade do checkpoint CLINC não inclui a implementação

- Severidade: alta
- Local: `experiments/split_clinc150.py:733-743,764-787,842-851`
- Evidência: a identidade inclui config, dados e metadados, mas não fingerprint do
  código. `protocol.json` é sobrescrito antes de validar o checkpoint existente.
- Impacto: `--resume` pode combinar estados gerados por implementações distintas;
  uma tentativa incompatível ainda pode deixar o protocolo em disco descrevendo
  dados antigos de forma errada.
- Correção mínima: usar `source_fingerprint/build_run_identity`, validar o
  protocolo existente antes de escrever e incluir o hash na identidade do
  checkpoint.

### 7. Pré-registro Split-MNIST pode ser enfraquecido removendo campos

- Severidade: alta
- Local: `experiments/confirmatory_split_mnist.py:102-141`
- Evidência: `expected_frozen` é construído iterando apenas pelas chaves que ainda
  existem no JSON comprometido. Se uma chave congelada for removida, ela deixa de
  ser verificada. As checagens fixas cobrem somente quatro hiperparâmetros.
- Impacto: o validador aceita um pré-registro incompleto após edição do arquivo.
- Correção mínima: declarar em código o conjunto completo de campos obrigatórios
  e validar um digest constante independente do conteúdo lido.

### 8. Manifesto FastHeat não prova que o vencedor veio do piloto

- Severidade: alta
- Local: `experiments/functional_dualheat.py:21-47,112-125,220-258`
- Evidência: o loader exige apenas schema/status e que a configuração pertença à
  grade. Não valida seeds, arquiteturas, resultados do piloto, objetivo, hash ou
  vencedor recomputado.
- Impacto: qualquer candidato da grade pode ser colocado retrospectivamente no
  manifesto e aceito como seleção congelada.
- Correção mínima: vincular hashes dos resultados, recomputar
  `select_fastheat_candidate()` e exigir igualdade do vencedor e do protocolo.

## Prioridade 1: podem alterar resultados ou sua interpretação

### 9. Scheduler global do CLINC confunde tarefa com learning rate

- Severidade: média
- Local: `experiments/split_clinc150.py:696-725,811-829,967-976`
- Evidência: existe um único scheduler linear para os dez estágios. Somente o
  início da sequência recebe warmup e a última tarefa termina próxima de LR zero.
- Impacto: aquisição por tarefa fica fortemente dependente da posição.
- Correção mínima: reiniciar scheduler por tarefa ou avaliar múltiplas ordens e
  declarar explicitamente o scheduler global.

### 10. Nova ablação Heat não agrega seu contraste principal

- Severidade: média
- Local: `experiments/split_clinc150.py:136-144,1355-1388`
- Evidência: `--heat-variants` inclui `slowheat_global` e
  `dualheat_global_topk`, mas a lista fixa de pares não calcula
  `dualheat_global_topk - slowheat_global`.
- Impacto: o agregado não fornece o efeito pareado de adicionar FastHeat ao
  envelope global correspondente.
- Correção mínima: adicionar esse par e declarar os contrastes local/global e
  local/hierárquico planejados.

### 11. Seeds de calibração e confirmação CLINC podem coincidir

- Severidade: média
- Local: `experiments/split_clinc150.py:1397-1435,1530-1633`
- Evidência: a CLI usa a mesma lista `--seeds` tanto em `--calibrate` quanto na
  execução final; o padrão é `[11,22,33]`. O manifesto não rejeita sobreposição.
- Impacto: a confirmação reutiliza o ruído usado para escolher hiperparâmetros.
- Correção mínima: pools distintos e validação explícita de disjunção.

### 12. Incerteza CLINC padrão é frágil

- Severidade: média
- Local: `experiments/split_clinc150.py:1355-1388,1532`
- Evidência: três seeds são resumidas por aproximação normal; os pares novos nem
  entram na inferência pareada.
- Impacto: intervalos podem aparentar estabilidade inexistente.
- Correção mínima: mais seeds/ordens, intervalo t ou bootstrap pareado e rótulo
  explícito de diagnóstico para n=3.

### 13. Custo CLINC perde histórico após resume

- Severidade: média
- Local: `experiments/split_clinc150.py:819-829,899,1197-1237`
- Evidência: checkpoint não contém tempo acumulado nem pico anterior. Após resume,
  o resultado mede apenas a sessão atual.
- Impacto: tempo, pico e overhead deixam de ser comparáveis.
- Correção mínima: persistir e mesclar tempo/pico por segmento.

### 14. Custo Split-MNIST omite fases auxiliares

- Severidade: média
- Local: `experiments/split_mnist.py:997-1010,2007-2027,2115-2153,2226-2233`
- Evidência: SCROLL e Classifier Expander incrementam exemplos/passos sem todos os
  custos de optimizer/hooks; nove varreduras de calibração registram tempo, mas
  não forward examples/FLOPs; treino classifier-only ainda recebe aproximação de
  backward completo.
- Impacto: comparações de custo entre métodos não são homogêneas.
- Correção mínima: contador único por fase e custo específico por tipo de
  backward/forward.

### 15. Calibração de bias tem desempate extremo

- Severidade: média
- Local: `experiments/split_mnist.py:65,1366-1397`
- Evidência: offsets começam em `-2.0` e o melhor só muda para `score > best`.
  Empates mantêm a intervenção mais extrema.
- Impacto: platôs de acurácia podem introduzir viés máximo sem ganho.
- Correção mínima: desempatar por menor `abs(offset)`, preferindo zero.

### 16. Resume completo pode reutilizar dados diferentes

- Severidade: média
- Local: `experiments/split_mnist.py:2550-2608`
- Evidência: no caminho que reutiliza `results.json`, o digest salvo só é validado
  sintaticamente; as tasks atuais não são carregadas e fingerprintadas novamente.
- Impacto: conteúdo alterado sob o mesmo loader pode reutilizar resultado antigo.
- Correção mínima: comparar com manifesto imutável do dataset ou recalcular o
  fingerprint antes do reuse.

### 17. Validadores genéricos aceitam tipos/NaN inválidos

- Severidade: média
- Local: `experiments/validation.py:9-24` e
  `experiments/synthetic_cl.py:71-93`
- Evidência: `require_positive_integers` aceita `1.5` e `True`;
  `require_nonnegative_values` aceita NaN; seed e tipos de `hidden_dims` não são
  verificados integralmente.
- Impacto: erro tardio em dimensões/range ou propagação silenciosa de NaN.
- Correção mínima: `type(value) is int`, finitude antes do sinal e validação de
  seed/dimensões.

### 18. `evaluating()` não preserva modos heterogêneos

- Severidade: média
- Local: `experiments/evaluation.py:11-20`
- Evidência: salva apenas o modo do módulo raiz. `eval()` e `train()` alteram todos
  os filhos recursivamente.
- Impacto: um submódulo deliberadamente mantido em eval pode voltar para train.
- Correção mínima: salvar/restaurar `training` de cada submódulo.

### 19. LPR falha em topologias válidas

- Severidade: média
- Local: `experiments/lpr.py:62-70,89-125`
- Evidência: `if name` exclui um modelo cuja raiz é `Linear`; o loop acessa
  `activations[name]` mesmo para ramos lineares não executados.
- Impacto: no-op silencioso para raiz afim ou `KeyError` em caminhos condicionais.
- Correção mínima: incluir raiz afim e processar somente módulos capturados, com
  contrato explícito para módulos não executados.

### 20. FLOPs do LPR em convoluções são subestimados

- Severidade: média
- Local: `experiments/lpr.py:93-135`
- Evidência: `unfold` cria `batch × spatial` linhas, mas `sample_counts` soma apenas
  batch e a estimativa usa esse valor.
- Impacto: custo de covariância pode ser subestimado por grande fator em CNNs.
- Correção mínima: contar `values.shape[0]` ou incluir o fator espacial.

### 21. Proveniência registra falha do Git como árvore limpa

- Severidade: média
- Local: `experiments/provenance.py:49-77`
- Evidência: erro do Git vira `None`; `bool(None)` vira `False` em `dirty`.
- Impacto: manifesto afirma árvore limpa quando o estado é desconhecido.
- Correção mínima: `dirty: bool | None` ou falha explícita.

### 22. ReplayBuffer aceita estado internamente impossível

- Severidade: média
- Local: `experiments/replay_memory.py:130-183`
- Evidência: aceita `inputs=None` com targets/metadados não vazios; nesse estado,
  `len(buffer)>0`, mas `as_memory()` retorna `None`. A forma `(N,3)` de
  `score_components` também não é revalidada.
- Impacto: checkpoint corrompido pode desativar replay silenciosamente.
- Correção mínima: reusar todas as invariantes de `append()` no loader e exigir
  `inputs is None` se e somente se o buffer estiver vazio.

### 23. Agregador sintético não possui identidade/resume seguro

- Severidade: média
- Local: `experiments/multi_seed.py:30-51,95-99`
- Evidência: reutiliza/reescreve diretórios sem identidade de código/config/dados.
- Impacto: mistura ou sobrescrita de artefatos sem vínculo verificável.
- Correção mínima: `build_run_identity/ensure_run_identity`, validação por seed e
  escrita atômica no produtor sintético.

### 24. Status da suíte pareada depende só da existência de um arquivo

- Severidade: média
- Local: `experiments/dualheat_pairs.py:339-355`
- Evidência: qualquer `pair_protocol.json` existente muda o status para
  `exploratory_paired_suite`; conteúdo não é lido nem validado.
- Impacto: árvore histórica/manipulada recebe rótulo de protocolo dedicado.
- Correção mínima: validar schema, seeds, configuração, pares e hash do lock.

## Telemetria e viewer

### 25. Timeline abandonada após resume permanece visível

- Severidade: média
- Local: `experiments/live_dashboard.py:102-115,233-246` e
  `experiments/split_clinc150.py:1130-1170`
- Evidência: avaliação é emitida antes do checkpoint. Se houver interrupção entre
  ambos, `_accuracy_views()` pode servir temporariamente uma avaliação de estado
  descartado; snapshots antigos e reexecutados aparecem juntos.
- Impacto: viewer pode mostrar estado não ancestral ao checkpoint válido. Não
  altera os resultados persistidos do treinamento.
- Correção mínima: timeline efetiva com rollback por `session_id/next_stage`.

### 26. Resumo de runs tem limite silencioso de 10 mil eventos

- Severidade: média
- Local: `experiments/live_dashboard.py:57-80` e
  `experiments/live_telemetry.py:38-68`
- Evidência: `_run_summaries()` usa o limite padrão e chama `events[-1]`.
- Impacto: runs maiores ficam com status e ordenação obsoletos.
- Correção mínima: índice atômico do último evento ou leitura eficiente do fim.

### 27. Polling relê o JSONL desde o início

- Severidade: média
- Local: `experiments/live_dashboard.py:190-214` e
  `experiments/live_telemetry.py:50-67`
- Evidência: mesmo com `after`, o leitor percorre fisicamente todas as linhas
  anteriores. A rota de accuracy relê o stream inteiro.
- Impacto: custo acumulado quadrático ao longo de runs longos.
- Correção mínima: offset de byte/índice incremental e cache das últimas matrizes.

### 28. Catálogo de snapshots desserializa todos os vetores

- Severidade: média
- Local: `experiments/live_dashboard.py:233-246`
- Evidência: para obter apenas nome, sequência e contexto, cada JSON completo é
  carregado.
- Impacto medido no run BERT atual: 60 snapshots ocupam 27,32 MB e uma varredura
  local levou aproximadamente 0,254 s. O custo cresce com épocas e tamanho do
  modelo.
- Correção mínima: índice leve escrito junto dos snapshots.

### 29. Leitura/escrita de telemetria não confina todos os symlinks

- Severidade: média para leitura; alta para escrita em árvore não confiável
- Local: `experiments/live_dashboard.py:93-99,207-246` e
  `experiments/live_telemetry.py:167-180`
- Evidência: `/api/snapshot` valida destino resolvido, mas `heat-latest.json`, o
  catálogo e o diretório `telemetry` do writer não recebem a mesma validação.
- Impacto: leitura ou escrita pode escapar do run root por link simbólico.
- Correção mínima: resolver cada destino e exigir `_is_within`; rejeitar links
  externos antes de abrir.

### 30. Dois writers podem produzir sequências duplicadas

- Severidade: média
- Local: `experiments/live_telemetry.py:167-180,197-206,225-244`
- Evidência: cada processo lê o mesmo último número e abre append sem lock.
- Impacto: eventos duplicados/intercalados quebram o cursor `after`.
- Correção mínima: lock exclusivo por diretório durante toda a sessão.

### 31. Corrupção no meio do JSONL é ignorada

- Severidade: média
- Local: `experiments/live_telemetry.py:38-68`
- Evidência: qualquer `JSONDecodeError` executa `continue`, embora a docstring
  prometa tolerar apenas truncamento final.
- Impacto: perda silenciosa de eventos e matrizes.
- Correção mínima: tolerar somente uma última linha sem newline; falhar ou
  registrar diagnóstico nas demais.

## Prioridade 2: robustez e manutenção

### 32. `completed_epochs` não representa épocas auxiliares reais

- Severidade: baixa
- Local: `experiments/split_mnist.py:2115-2166`
- Evidência: o valor é `len(stage_validation)`, não a soma explícita das épocas de
  treino principal/SCROLL/Classifier Expander.
- Impacto: orçamento reportado fica incorreto quando épocas auxiliares diferem de
  um.

### 33. Seleção de replay CLINC usa os primeiros registros

- Severidade: baixa
- Local: `experiments/split_clinc150.py:515-528`
- Evidência: usa `indices[:per_class]` sem gerador seedado.
- Impacto: memória depende da ordenação e não varia entre seeds.

### 34. CLINC não registra sobreposição entre splits

- Severidade: baixa
- Local: `experiments/split_clinc150.py:419-457`
- Evidência: splits são materializados independentemente sem auditoria de pares
  `(texto,intenção)` repetidos.
- Impacto: uma revisão de dataset com duplicatas produziria vazamento silencioso.

### 35. Teste de sinais transborda em amostras muito grandes

- Severidade: baixa
- Local: `experiments/confirmatory_statistics.py:31-48`
- Evidência: `math.comb` seguido de conversão/divisão float pode lançar
  `OverflowError` com milhares de pares.
- Impacto: não afeta os protocolos atuais de 3–20 seeds, mas limita reuso.

### 36. Artefatos sintéticos finais não são atômicos

- Severidade: baixa
- Local: `experiments/synthetic_cl.py:414-448`
- Evidência: JSON e CSV finais usam `open('w')`, apesar dos helpers atômicos.
- Impacto: interrupção pode deixar resultado truncado.

## Arquivos sem achado material próprio

Os seguintes arquivos foram lidos integralmente e não apresentaram defeito
material independente; seus contratos foram avaliados por meio dos consumidores:

- `experiments/__init__.py`
- `experiments/artifacts.py`
- `experiments/contracts.py`
- `experiments/method_specs.py`
- `experiments/model_factory.py`
- `experiments/peak_memory.py`
- `experiments/replay_selection_sweep.py`
- `experiments/run_split_mnist_5seeds.py`
- `experiments/run_split_mnist_10seeds.py`
- `experiments/sections.py`
- `experiments/split_mnist_suite.py`
- `experiments/visual_generalization.py`

## Verificação executada

- 27/27 arquivos analisados sintaticamente por AST: nenhum erro.
- Busca de `torch.load`, pickle, YAML inseguro, `eval`, `exec`, `shell=True` e
  `os.system`: o único `torch.load` usa `weights_only=True` em
  `experiments/artifacts.py:83-89`; nenhuma desserialização insegura encontrada.
- Suíte CPU executada antes da auditoria, sem mudanças posteriores no código:
  274 testes passaram.
- Run BERT inspecionado: 4.800 eventos (3,62 MB), 60 snapshots (27,32 MB).
- Não foi executado treinamento nem download de dataset.

## Ordem recomendada de correção

1. EWC Fisher, Classifier Expander e pareamento do replay.
2. CLINC: teste somente após protocolo congelado, RNG no resume e identidade de
   código.
3. Contraste agregado das quatro variantes Heat e separação de seeds.
4. Pré-registros Split-MNIST/FastHeat.
5. ReplayBuffer, custo e validação de configs.
6. Índice/locks/confinamento da telemetria.
7. Robustez de LPR, proveniência e artefatos sintéticos.

---

## Status de correção (milestone P0)

Os achados acima são preservados como registro histórico. Esta seção marca
apenas o que foi corrigido e verificado por teste de regressão. Suíte no fim do
milestone: 301 passed, 0 failed (CPU, GPU invisível, uma thread).

Baseline antes das correções: 273 passed, **1 failed**
(`test_huggingface_from_pretrained_loads_native_bert_checkpoint`) e 11 erros de
Ruff. A falha era real e foi corrigida neste milestone; os 11 erros de Ruff
permanecem e pertencem à Fase H.

| Achado | Estado | Teste de regressão |
|---|---|---|
| Fisher EWC incorreto | corrigido | `test_empirical_fisher_squares_per_example_before_averaging`, `test_ewc_consolidation_divides_fisher_by_example_count` |
| Destilação de classes novas | corrigido | `test_classifier_expander_distillation_ignores_new_classes` |
| Replay adaptativo não pareado | corrigido | `test_replay_selection_pairing_flag_is_reported_per_method`, `test_paired_suite_rejects_learner_adaptive_replay_selection` |
| Teste CLINC por padrão | corrigido | `test_clinc_defaults_to_validation_only`, `test_cli_evaluate_test_defaults_to_false`, `test_cli_rejects_test_access_without_frozen_manifest` |
| Resume CLINC sem código | corrigido | `test_incompatible_protocol_is_not_overwritten`, `test_protocol_records_source_fingerprint` |
| Resume CLINC sem RNG | corrigido | `test_resume_with_dropout_matches_uninterrupted_run`, `test_checkpoint_without_rng_state_is_rejected` |
| Scheduler único em CLINC | corrigido | `test_task_scoped_scheduler_repeats_the_same_relative_schedule`, `test_stream_scoped_scheduler_spreads_decay_over_the_whole_stream`, `test_protocol_records_scheduler_scope` |
| `global_topk` silenciosamente errado | corrigido | `test_global_topk_requires_external_global_scale`, `test_global_topk_runs_once_the_coordinator_supplies_the_scale` |
| Estado científico em dtype reduzido | corrigido | `test_fast_heat_buffer_stays_fp32_under_reduced_precision`, `test_slow_heat_scientific_buffers_resist_dtype_casts`, `test_dual_heat_state_keeps_precision_and_counter_dtype`, `test_legacy_counter_increments_exactly_beyond_fp16_resolution`, `test_bert_scientific_state_survives_half_precision_cast` |
| Pré-registro removível | **pendente** | Fase E1 |
| Manifesto FastHeat não verificável | **pendente** | Fase E2 |
| Máscara fail-closed ao limpar | **pendente** | Fase D3 |
| Validadores e `ReplayBuffer` | **pendente** | Fase D4 |

### Achados adicionais durante a implementação

Dois defeitos não previstos na auditoria original apareceram ao escrever os
testes e foram corrigidos no mesmo milestone:

1. **Assinatura BERT colapsava configurações distintas sob fp16.**
   `_slowheat_signature` era um vetor float; sob `.half()` os valores pequenos
   sofriam underflow, de modo que `importance_eps=1e-8` e `1e-9` viravam ambos
   `0.0` e um checkpoint de outro protocolo era aceito silenciosamente. A
   assinatura passou a ser um digest SHA-256 em int64 sobre o payload canônico,
   imune a qualquer cast. Teste:
   `test_half_precision_does_not_collapse_distinct_slowheat_configs`.

2. **`from_pretrained` deixava `fast_heat` com memória arbitrária.**
   Transformers materializa módulos preguiçosamente e marca os gates com
   `_is_hf_initialized`, logo nunca os inicializava; o buffer começava com lixo
   em vez de zeros. O reset foi colocado em
   `_adjust_missing_and_unexpected_keys`, único hook que roda após o
   carregamento e ainda distingue um `fast_heat` genuinamente ausente de um que
   o checkpoint restaurou. Teste:
   `test_huggingface_from_pretrained_loads_native_bert_checkpoint`.

### Regressão introduzida e corrigida no próprio milestone

Fechar `evaluate_test` por padrão (C1) colidiu com um guard antigo em
`run_split_clinc150_multi_seed()`, que exigia `evaluate_test=True` para agregar.
Enquanto o default era `True` o guard nunca disparava; com o default `False` ele
passou a abortar **toda** execução multi-seed, inclusive exploratória. Como
`--evaluate-test` exige `--frozen-manifest`, o escape também estava fechado: na
prática, nenhuma run CLINC exploratória era possível.

A correção separa os dois modos que o guard misturava:

- run exploratória (padrão): agrega os endpoints de **validação**
  (`validation_metrics`), sem tocar no split de teste;
- run confirmatória (`confirmatory=True`, ligado pelo CLI a `--evaluate-test` e
  portanto a `--frozen-manifest`): exige `evaluate_test=True` e agrega teste.

`aggregate.json` passou a registrar `endpoint_source` (`validation` ou `test`),
`evaluate_test` e `confirmatory`, de modo que nenhuma agregação seja lida sem
saber de qual split ela veio. Testes:
`test_multi_seed_aggregates_validation_when_test_split_is_closed`,
`test_multi_seed_confirmatory_mode_requires_the_test_split`,
`test_multi_seed_uses_test_endpoints_when_explicitly_opened`.

Causa do ponto cego: a cobertura do milestone exercitava `run_split_clinc150`
(seed única), o parser e o gate do manifesto, mas nunca
`run_split_clinc150_multi_seed`, que não tinha teste algum.

### Decisão de protocolo registrada

`apply_frozen_slowheat_manifest()` não força mais `evaluate_test=True`. O acesso
ao split de teste passou a depender exclusivamente da flag explícita
`--evaluate-test`, que por sua vez exige `--frozen-manifest`. O teste histórico
que exigia o comportamento antigo foi reescrito para afirmar que o manifesto
preserva a flag.

`CHECKPOINT_SCHEMA_VERSION` foi para `2`. Checkpoints v1 não guardam estado de
RNG e são rejeitados com mensagem de migração, sem adivinhar o estado ausente.

### Limite de evidência

Nenhum resultado científico novo foi produzido neste milestone. As correções de
EWC, destilação, pareamento de replay e protocolo CLINC alteram números, logo
**nenhuma execução anterior a estes commits pode ser promovida como evidência
confirmatória**.
