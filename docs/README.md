# Documentação do DualHeat

Estado deste índice: 19 de setembro de 2026.

Este arquivo é a entrada autoritativa para a documentação. O projeto é um
protótipo de pesquisa em aprendizagem contínua. A presença de uma implementação
ou de um resultado em `results/` não implica validação confirmatória nem estado
da arte.

## Estado atual

| Área | Estado | Documento principal |
|---|---|---|
| Functional SlowHeat para MLP | implementado e testado | [functional_slowheat.md](functional_slowheat.md) |
| SlowHeat para CNN, VGG11 e ResNet18 | implementado e testado | [functional_slowheat_cnn.md](functional_slowheat_cnn.md) |
| Functional DualHeat e FastHeat | implementado; benchmarks visuais exploratórios concluídos | [functional_dualheat.md](functional_dualheat.md) |
| BERT/CLINC150 | implementado; diagnósticos exploratórios de mecanismo e replay concluídos | [bert_slowheat_diagnostic_results.md](bert_slowheat_diagnostic_results.md) |
| Trackers para Transformers | BERT implementado; SwiGLU implementado no host Qwen2; GQA, QKV fundido e distribuição ainda planejados | [functional_slowheat_transformers.md](functional_slowheat_transformers.md) |
| Qwen2 / SlowHeat em LLM | host, diagnóstico de capacidade e runner de iso-plasticidade implementados e testados; smoke em GPU executado; sem resultado de continual learning agregado | [functional_slowheat_qwen.md](functional_slowheat_qwen.md) |
| Calibração de capacidade Qwen | aritmética implementada; critério declarado ainda não aplicado numa run | [qwen_capacity_calibration.md](qwen_capacity_calibration.md) |
| Ablação iso-plasticidade Qwen | protocolo congelado em `goals/protocol_iso_plasticity.md`; runs de calibração obsoletas (30 passos); confirmação em andamento | [qwen_iso_plasticity_ablation.md](qwen_iso_plasticity_ablation.md) |
| RNN/LSTM | proposta de design, não implementada | [functional_slowheat_rnn_lstm.md](functional_slowheat_rnn_lstm.md) |
| Replay seletivo | implementado com quatro estratégias | [replay_selection.md](replay_selection.md) |
| Dashboard e telemetria ao vivo | implementados | [live_dashboard.md](live_dashboard.md) |
| Otimizadores mascarados | implementados e testados | [optimizer_semantics.md](optimizer_semantics.md) |
| Auditoria do engine experimental | 12 de 36 achados corrigidos; 24 ainda abertos | [experiments_audit.md](experiments_audit.md) |
| Índice de resultados versionados | mapa de todos os diretórios de `results/` | [results_index.md](results_index.md) |
| Protocolo confirmatório | executado; 20 seeds, duas execuções concordantes | [confirmatory_protocol.md](confirmatory_protocol.md) |

## O que foi adicionado ao projeto

As adições que não estavam cobertas pelo catálogo histórico original são:

- Functional DualHeat em MLP, CNN pequena, VGG11 e ResNet18;
- FastHeat aplicado em treino e avaliação, com estado atualizado apenas em
  treino;
- integração opcional com BERT da Hugging Face;
- tracking funcional de FFN e cabeças de atenção;
- capacity budget local, global por família e hierárquico;
- FastHeat `global_topk` no host BERT;
- LoRA exato produtor-only, com A congelado e B mascarado;
- round-trip do protocolo BERT por `save_pretrained()` e `from_pretrained()`;
- benchmark CLINC150 com replay, calibração, checkpoint, retomada e telemetria;
- métodos visuais LPR, Classifier Expander e SCROLL, com variantes SlowHeat e
  Functional DualHeat;
- quatro políticas de seleção da memória de replay;
- benchmarks pareados Functional DualHeat em VGG11 e ResNet18;
- dashboard HTTP com histórico de acurácia, Heat/FastHeat, snapshots por época e
  estágio e comparação entre métodos.

O inventário das implementações e identificadores aceitos pelos runners está em
[methods_catalog.md](methods_catalog.md).

## Resultados

- [Diagnósticos BERT SlowHeat](bert_slowheat_diagnostic_results.md): comparação
  de dez seeds contra BERT sequencial, controles hard aleatórios, interação com
  replay, orçamento de memória e decisão de não escalar a configuração atual.
- [Resultados históricos BERT/CLINC150](bert_clinc150_results.md): protocolo,
  tabela das 11 runs, artefatos-fonte e limitações.
- [Functional DualHeat](functional_dualheat.md): piloto de FastHeat e resultados
  pareados em Split-CIFAR-10.
- [Catálogo histórico de métodos e resultados](project_methods_and_results.md):
  análise do CSV Split-MNIST de cinco seeds. O documento preserva o recorte
  histórico e não deve ser usado como inventário do estado atual.
- [Log Split-MNIST](split_mnist_experiment_log.md).
- [Piloto sintético](synthetic_ablation_pilot.md).
- [Split-CIFAR](split_cifar.md).

## Protocolos e reprodução

- [Protocolo confirmatório](confirmatory_protocol.md)
- [Protocolo pareado DualHeat](dualheat_paired_protocol.md)
- [Reprodutibilidade do runner sintético](reproducibility.md)
- [Auditoria dos experimentos](experiments_audit.md)

## Implementação

| Componente | Fonte principal |
|---|---|
| SlowHeat linear/convolucional/MLP/VGG | `src/dual_heater/slow_heat.py` |
| FastHeat | `src/dual_heater/fast_heat.py` |
| ResNet18 e variantes | `src/dual_heater/resnet.py` |
| DualHeat legado | `src/dual_heater/dual_heat.py` |
| LoRA legado | `src/dual_heater/lora.py` |
| BERT e LoRA exato | `src/dual_heater/bert.py` |
| Trackers Transformer | `src/dual_heater/transformer.py` |
| Otimizadores | `src/dual_heater/optim.py` |
| Métricas | `src/dual_heater/metrics.py` |
| Runner visual | `experiments/split_mnist.py` |
| Runner CLINC150 | `experiments/split_clinc150.py` |
| Registro de seções | `experiments/sections.py` |
| Registro declarativo de métodos | `experiments/method_specs.py` |

## Como interpretar o status científico

- `implementado`: existe caminho executável e cobertura automatizada relevante;
- `experimental`: implementado, mas sem evidência confirmatória suficiente;
- `histórico`: resultado real preservado, porém produzido por protocolo antigo,
  incompleto ou com proveniência limitada;
- `planejado`: contrato ou proposta sem implementação disponível.

Os resultados BERT incluem artefatos históricos e diagnósticos exploratórios
atuais. O SlowHeat superou o BERT sequencial no diagnóstico de duas tarefas, mas
não demonstrou vantagem aceitável sobre replay e não avançou para confirmação
em dez tarefas. Os benchmarks visuais de Functional DualHeat usam dez seeds
pareadas, mas continuam marcados nos próprios artefatos como
`exploratory_paired_benchmark`.
