# Método versus método + DualHeat

Implementação: **Functional SlowHeat**, não a classe legada DualHeatMLP.

Execução de origem: `/home/fachel/dual-heater-mlp-research/results/split_mnist_protocol/split_mnist_all_methods`.

MLP [256, 128]; cenário `class_incremental`; 10 épocas por tarefa.

Análise exploratória; 10 seeds pareadas. Não constitui confirmação independente.

Delta = com componente − sem componente. Acurácia e forgetting em pontos percentuais; esquecer menos só é útil se a aquisição for preservada.

| Método | Sem (%) | Com (%) | Delta (pp) | IC95% t (pp) | Delta forgetting (pp) | Tempo com/sem |
|---|---:|---:|---:|---|---:|---:|
| Convencional (AdamW) | 19.618 | 19.488 | -0.130 | [-0.211, -0.049] | -0.130 | 2.03× |
| Replay | 75.936 | 76.426 | +0.490 | [-0.399, +1.379] | -1.033 | 1.69× |
| DER++ | 82.496 | 85.212 | +2.716 | [+2.271, +3.161] | -4.383 | 1.62× |
| ER-ACE | 70.710 | 70.106 | -0.604 | [-1.430, +0.222] | +0.640 | 1.62× |

## Interpretação e limites

- ICs são pontuais; o JSON/CSV inclui p de acurácia ajustado por Holm para os quatro pares deste dataset. Não há correção entre datasets.
- Ganhos negativos e empates permanecem no relatório. Não se assume benefício universal.
- Mesmos parâmetros treináveis, épocas, passos, exemplos e memória de replay/logits são verificados dentro de cada par. Isso não iguala o custo computacional.
- Tempo observado é descritivo, sem medição isolada com aquecimento; a ordem dos métodos é fixa. FLOPs são estimativas.
- Bytes de replay/logits não são memória total. Pico de memória não foi medido.
- Fronteiras de tarefa conhecidas; inferência sem task ID. Baselines usam defaults, sem ajuste individual nesta suíte.
- Consulte pair_report.json para bootstrap pareado, sinais, demais métricas, configuração e hashes dos arquivos de origem.
