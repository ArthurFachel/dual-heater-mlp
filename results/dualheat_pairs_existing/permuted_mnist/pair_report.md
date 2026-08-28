# Método versus método + DualHeat

Implementação: **Functional SlowHeat**, não a classe legada DualHeatMLP.

Execução de origem: `/home/fachel/dual-heater-mlp-research/results/split_mnist_protocol/permuted_mnist`.

MLP [512, 256]; cenário `domain_incremental`; 10 épocas por tarefa.

Análise exploratória; 10 seeds pareadas. Não constitui confirmação independente.

Delta = com componente − sem componente. Acurácia e forgetting em pontos percentuais; esquecer menos só é útil se a aquisição for preservada.

| Método | Sem (%) | Com (%) | Delta (pp) | IC95% t (pp) | Delta forgetting (pp) | Tempo com/sem |
|---|---:|---:|---:|---|---:|---:|
| Convencional (AdamW) | 71.947 | 79.943 | +7.996 | [+6.502, +9.490] | -10.499 | 2.09× |
| Replay | 93.204 | 93.826 | +0.622 | [+0.469, +0.776] | -1.332 | 1.71× |
| DER++ | 95.486 | 95.572 | +0.085 | [+0.029, +0.142] | -0.556 | 1.66× |
| ER-ACE | 93.240 | 93.853 | +0.613 | [+0.384, +0.842] | -1.386 | 1.63× |

## Interpretação e limites

- ICs são pontuais; o JSON/CSV inclui p de acurácia ajustado por Holm para os quatro pares deste dataset. Não há correção entre datasets.
- Ganhos negativos e empates permanecem no relatório. Não se assume benefício universal.
- Mesmos parâmetros treináveis, épocas, passos, exemplos e memória de replay/logits são verificados dentro de cada par. Isso não iguala o custo computacional.
- Tempo observado é descritivo, sem medição isolada com aquecimento; a ordem dos métodos é fixa. FLOPs são estimativas.
- Bytes de replay/logits não são memória total. Pico de memória não foi medido.
- Fronteiras de tarefa conhecidas; inferência sem task ID. Baselines usam defaults, sem ajuste individual nesta suíte.
- Consulte pair_report.json para bootstrap pareado, sinais, demais métricas, configuração e hashes dos arquivos de origem.
