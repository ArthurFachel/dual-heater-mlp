# Catálogo atual de métodos

Estado do inventário: 16 de setembro de 2026. A fonte de verdade para os nomes
aceitos é o código, especialmente `experiments/split_mnist.py`,
`experiments/method_specs.py`, `experiments/synthetic_cl.py` e
`experiments/split_clinc150.py`.

## 1. Mecanismos centrais

### Functional SlowHeat

Implementado em `src/dual_heater/slow_heat.py` para unidades lineares, canais
convolucionais, MLP, CNN pequena e VGG11; a variante ResNet18 está em
`src/dual_heater/resnet.py`.

O sinal instantâneo por unidade é `|z * dL/dz|`. Ele é normalizado, acumulado na
tarefa e consolidado na fronteira por `max`, `mean` ou `sum`. A importância
consolidada produz uma escala plástica e um budget reserva capacidade livre. A
proteção pode cobrir os dois lados do caminho funcional: linhas produtoras e
colunas consumidoras.

APIs principais:

- `SlowHeatLinear`, `SlowHeatConv2d`, `SlowHeatChannelTracker`;
- `SlowHeatMLP`, `SlowHeatCNN`, `SlowHeatVGG11`, `SlowHeatResNet18`;
- `consolidate()`, `get_lr_scales()`, `capacity_metrics()` e
  `adapt_capacity()`.

### FastHeat

Implementado em `src/dual_heater/fast_heat.py`. Mantém um estado transitório por
unidade/canal e aplica competição divisiva às ativações ocultas. O gate é
aplicado em treino e avaliação; seu estado só muda em treino. O estado faz parte
do `state_dict` e pode ser lido ou zerado por `get_fast_states()` e
`reset_fast_heat()`.

`FastHeatConfig` expõe decay, força, limiar, epsilon, modo de competição e fração
top-k. `mean_others` é o comportamento autônomo do `FastHeatGate`.
`global_topk` requer uma escala externa preparada pelo host BERT; as APIs
visuais atuais não o ativam isoladamente.

### Functional DualHeat

É a composição `Functional SlowHeat + FastHeat`, diferente de `DualHeatMLP`
legado. As classes são:

- `FunctionalDualHeatMLP`;
- `FunctionalDualHeatCNN`;
- `FunctionalDualHeatVGG11`;
- `FunctionalDualHeatResNet18`.

### DualHeat legado

`DualHeatLinear` e `DualHeatMLP`, em `src/dual_heater/dual_heat.py`, combinam
atividade rápida, slow window opcional e máscara no gradiente bruto. Não usam o
lifecycle de consolidação por fronteira do Functional SlowHeat. Permanecem para
compatibilidade e ablações históricas.

### Otimizadores SlowHeat

`SlowHeatAdamW` e `SlowHeatSGD`, em `src/dual_heater/optim.py`, aplicam a máscara
ao delta final, incluindo weight decay. `follow_update` também interpola os
estados do otimizador; `native` preserva a trajetória nativa dos estados como
ablação. O caminho mascarado atual é pontual e não mantém snapshots completos
até o fim do step.

### Transformers e BERT

- `SlowHeatFFNTracker`: utilidade de unidades intermediárias, excluindo padding;
- `SlowHeatAttentionTracker`: importância por cabeça a partir de Q/K/V/saída;
- `SlowHeatBertForSequenceClassification`: instrumentação de BERT, máscaras,
  consolidação, budgets local/global/hierárquico, cobertura opcional de
  embeddings/residual/LayerNorm/pooler/classificador e persistência do protocolo;
- `build_exact_slowheat_lora`: LoRA produtor-only, com A congelado e B
  mascarado;
- `FastHeatActivation`: competição pós-GELU antes de `output.dense`.

As dependências de NLP são opcionais: `dual-heater[nlp]`.

## 2. Métodos aceitos pelo runner visual

O registro completo atual está em `_METHOD_SPECS`, em
`experiments/split_mnist.py`. `SUPPORTED_METHODS` é deliberadamente histórico e
omite os cinco identificadores Functional DualHeat; não deve ser usado como
inventário completo.

### Controles e SlowHeat

| Identificador | Composição |
|---|---|
| `vanilla` | treino sequencial com AdamW |
| `fastheat` | somente FastHeat |
| `slowheat` | Functional SlowHeat |
| `dualheat` | Functional SlowHeat + FastHeat |
| `slowheat_adaptive` | budget adaptado por aquisição de validação |
| `slowheat_native_state` | máscara de parâmetros com estado AdamW nativo |
| `slowheat_unidirectional` | proteção somente por linha |
| `slowheat_unbudgeted` | sem garantia mínima de plasticidade |
| `slowheat_none` | wiring SlowHeat sem consolidação |
| `hard_freeze` | congelamento binário das unidades consolidadas |

### Replay, distillation e baselines

| Identificador | Composição |
|---|---|
| `replay`, `replay_calibrated` | replay simples, com calibração opcional |
| `replay_balanced` | pesos 0,5/0,5 para losses atual e replay |
| `replay_more_epochs` | replay com orçamento maior de épocas |
| `replay_early_stopping` | replay com early stopping por validação |
| `replay_global_lr_reduction` | controle de LR global reduzido |
| `distillation` | professor congelado, sem memória de imagens |
| `lwf_calibrated` | Learning without Forgetting calibrado |
| `derpp` | replay de rótulos e logits, com CE + MSE |
| `er_ace` | loss atual restrita às classes correntes e replay global |
| `agem` | projeção do gradiente conflitante pela memória |
| `ewc` | penalidade quadrática com importância diagonal online |
| `si` | Synaptic Intelligence |

Limitação conhecida: o EWC atual usa o quadrado do gradiente médio do minibatch,
não a média dos quadrados por exemplo. Os resultados EWC devem ser descritos
como a aproximação implementada, não como Fisher empírico exato.

### Combinações

- `slowheat_replay`, `slowheat_distillation`;
- `slowheat_derpp_hidden_beta_30_budget_0.25`;
- `slowheat_er_ace_hidden_beta_30_budget_0.25`;
- `slowheat_replay_hidden_adaptive_beta_30_budget_0.25`;
- `slowheat_replay_partial_output_beta_30_budget_0.25`;
- `slowheat_replay_hidden_beta_30_budget_0.25_calibrated`.

O parser também aceita a família
`slowheat[_replay|_distillation|_unidirectional][_hidden]_beta_X[_budget_Y]`.

### Métodos visuais adicionados

| Família | Base | SlowHeat | Functional DualHeat |
|---|---|---|---|
| LPR | `lpr` | `slowheat_lpr` | `dualheat_lpr` |
| Classifier Expander | `classifier_expander` | `slowheat_classifier_expander` | `dualheat_classifier_expander` |
| SCROLL | `scroll` | `slowheat_scroll` | `dualheat_scroll` |

LPR (`experiments/lpr.py`) pré-condiciona gradientes por covariâncias das
ativações de replay. Classifier Expander possui uma fase auxiliar da cabeça com
replay e distillation. SCROLL congela o extrator após a primeira tarefa e ajusta
a cabeça por estatísticas suficientes e regressão ridge.

Limitação conhecida: Classifier Expander atualmente destila sobre todas as
classes vistas, incluindo as novas, embora o professor tenha sido congelado
antes da tarefa. Esse caminho deve ser tratado como experimental até a correção.

## 3. Seleção de replay

`experiments/replay_memory.py` implementa:

- `first`: primeiros exemplos de cada classe;
- `loss`: maior loss do modelo seletor;
- `representative`: proximidade ao centro de features da classe;
- `hybrid`: combinação ponderada de loss, representatividade e diversidade.

As estratégias adaptativas dependem do modelo. Portanto, índices de replay só
são idênticos entre métodos quando um seletor comum é congelado e reutilizado;
`first` é o único modo naturalmente independente do modelo.

## 4. Métodos sintéticos

O runner sintético aceita:

- `vanilla`, `reduced_lr`;
- `slowheat_max`, `slowheat_mean`, `slowheat_sum`, `slowheat_none`;
- `slowheat_max_native_state`, `slowheat_max_unidirectional`;
- `slowheat_max_unbudgeted`, `slowheat_max_sgd`;
- `slowheat_max_legacy_adamw`.

## 5. Métodos BERT/CLINC150

| Identificador | Descrição |
|---|---|
| `vanilla` | fine-tuning sequencial completo |
| `slowheat_none` | instrumentação sem consolidação |
| `slowheat_ffn` | SlowHeat apenas na FFN |
| `slowheat` | FFN + atenção, parâmetros não vinculados treináveis |
| `replay` | replay textual |
| `slowheat_replay` | SlowHeat histórico (FFN + atenção) + replay |
| `lora_replay` | LoRA + replay |
| `slowheat_lora_replay` | LoRA exato mascarado + replay |
| `slowheat_bound` | SlowHeat com parâmetros sem máscara congelados |
| `dualheat` | `slowheat_bound` + FastHeat pós-GELU |
| `slowheat_bound_replay` | protocolo bound + replay |
| `dualheat_replay` | protocolo bound + FastHeat + replay |
| `slowheat_global` | capacidade global por família |
| `slowheat_hierarchical` | mínimos locais + redistribuição global |
| `dualheat_global_topk` | escopo global + FastHeat top-k global |
| `slowheat_full_coverage` | cobertura hierárquica de embeddings, LayerNorm, residual, atenção, FFN, pooler e classificador |
| `slowheat_all_minus_embeddings` | cobertura completa sem o fator de embeddings |
| `slowheat_all_minus_layernorm` | cobertura completa sem máscaras nos parâmetros afins de LayerNorm |
| `slowheat_all_minus_residual` | cobertura completa sem fatores residuais de linha/coluna |
| `slowheat_all_minus_attention` | cobertura completa sem importância específica por cabeça |
| `slowheat_all_minus_ffn` | cobertura completa sem importância específica por neurônio FFN |
| `slowheat_all_minus_pooler` | cobertura completa sem o fator específico do pooler |
| `slowheat_all_minus_classifier` | cobertura completa sem o fator específico por logit |
| `slowheat_ffn_attention` | baseline explícito FFN + atenção, não bound, com capacidade hierárquica |

O preset `--heat-variants` executa quatro protocolos comparáveis de
alocação: `slowheat_bound`, `slowheat_global`, `slowheat_hierarchical` e
`dualheat_global_topk`. Resultados históricos: [bert_clinc150_results.md](bert_clinc150_results.md).

O preset `--full-coverage-variants` executa exatamente dez métodos: `vanilla`,
`slowheat_full_coverage`, as sete variantes `slowheat_all_minus_*` da tabela e
`slowheat_ffn_attention`. Todos os nove métodos SlowHeat desse preset mantêm
parâmetros sem binding treináveis, desabilitam FastHeat e replay e usam capacidade
hierárquica. As ablações removem um **fator**; outro endpoint ainda pode mascarar
parte do mesmo módulo. Elas também não são pareadas por quantidade de parâmetros
mascarados, portanto seus resultados devem incluir `mask_coverage`. Contrato
completo: [bert_full_coverage_ablation.md](bert_full_coverage_ablation.md).

## 6. Backbones e cenários

- MLP: Split-MNIST, Permuted-MNIST, Split-CIFAR achatado e sintético;
- CNN pequena, VGG11 e ResNet18: Split-CIFAR-10;
- VGG11 e ResNet18: benchmark pareado Functional DualHeat;
- MLP: Split-CIFAR-100 no runner visual;
- BERT-Mini e BERT-base: CLINC150 por domínio. BERT-base exige manifesto
  congelado na CLI.

As 21 seções disponíveis e seus diretórios são declaradas em
`experiments/sections.py`; várias seções grandes são opt-in.
