# Método versus método + DualHeat

Implementação: **Functional SlowHeat**, não a classe legada DualHeatMLP.

Execução de origem: `.` (caminho relativo à pasta deste relatório).

MLP [256, 128]; cenário `class_incremental`; 10 épocas por tarefa.

Análise exploratória; 10 seeds pareadas. Não constitui confirmação independente.

Delta = com componente − sem componente. Acurácia e forgetting em pontos percentuais; esquecer menos só é útil se a aquisição for preservada.

| Método | Sem (%) | Com (%) | Delta (pp) | IC95% t (pp) | Delta forgetting (pp) | Tempo com/sem |
|---|---:|---:|---:|---|---:|---:|
| Convencional (AdamW) | 19.638 | 19.516 | -0.122 | [-0.166, -0.078] | -0.168 | 1.99× |
| Replay | 76.244 | 76.380 | +0.136 | [-0.811, +1.083] | -0.510 | 1.73× |
| DER++ | 82.402 | 85.584 | +3.182 | [+2.696, +3.668] | -4.865 | 1.63× |
| ER-ACE | 71.208 | 71.128 | -0.080 | [-1.021, +0.861] | +0.525 | 1.62× |

## Interpretação e limites

- ICs são pontuais; o JSON/CSV inclui p de acurácia ajustado por Holm para os quatro pares deste dataset. Não há correção entre datasets.
- Ganhos negativos e empates permanecem no relatório. Não se assume benefício universal.
- Mesmos parâmetros treináveis, épocas, passos, exemplos e memória de replay/logits são verificados dentro de cada par. Isso não iguala o custo computacional.
- Tempo observado é descritivo, sem medição isolada com aquecimento; a ordem dos métodos é fixa. FLOPs são estimativas.
- Bytes de replay/logits não são memória total. Pico de memória não foi medido.
- Fronteiras de tarefa conhecidas; inferência sem task ID. Baselines usam defaults, sem ajuste individual nesta suíte.
- Consulte pair_report.json para bootstrap pareado, sinais, demais métricas, configuração e hashes dos arquivos de origem.
