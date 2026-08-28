# Método versus método + DualHeat

Implementação: **Functional SlowHeat**, não a classe legada DualHeatMLP.

Execução de origem: `.` (caminho relativo à pasta deste relatório).

MLP [1024, 512]; cenário `class_incremental`; 10 épocas por tarefa.

Análise exploratória; 10 seeds pareadas. Não constitui confirmação independente.

Delta = com componente − sem componente. Acurácia e forgetting em pontos percentuais; esquecer menos só é útil se a aquisição for preservada.

| Método | Sem (%) | Com (%) | Delta (pp) | IC95% t (pp) | Delta forgetting (pp) | Tempo com/sem |
|---|---:|---:|---:|---|---:|---:|
| Convencional (AdamW) | 5.437 | 5.488 | +0.051 | [-0.083, +0.185] | +0.300 | 1.86× |
| Replay | 9.269 | 9.890 | +0.621 | [+0.333, +0.909] | +0.679 | 1.60× |
| DER++ | 11.360 | 10.830 | -0.530 | [-0.819, -0.241] | +0.071 | 1.56× |
| ER-ACE | 13.425 | 14.699 | +1.274 | [+0.833, +1.715] | -0.421 | 1.57× |

## Interpretação e limites

- ICs são pontuais; o JSON/CSV inclui p de acurácia ajustado por Holm para os quatro pares deste dataset. Não há correção entre datasets.
- Ganhos negativos e empates permanecem no relatório. Não se assume benefício universal.
- Mesmos parâmetros treináveis, épocas, passos, exemplos e memória de replay/logits são verificados dentro de cada par. Isso não iguala o custo computacional.
- Tempo observado é descritivo, sem medição isolada com aquecimento; a ordem dos métodos é fixa. FLOPs são estimativas.
- Bytes de replay/logits não são memória total. Pico de memória não foi medido.
- Fronteiras de tarefa conhecidas; inferência sem task ID. Baselines usam defaults, sem ajuste individual nesta suíte.
- Consulte pair_report.json para bootstrap pareado, sinais, demais métricas, configuração e hashes dos arquivos de origem.
