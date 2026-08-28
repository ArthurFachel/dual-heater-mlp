# Método versus método + DualHeat

Implementação: **Functional SlowHeat**, não a classe legada DualHeatMLP.

Análise exploratória; 2 seeds pareadas. Não constitui confirmação independente.

Delta = com componente − sem componente. Acurácia e forgetting em pontos percentuais; esquecer menos só é útil se a aquisição for preservada.

| Método | Sem (%) | Com (%) | Delta (pp) | IC95% t (pp) | Delta forgetting (pp) | Tempo com/sem |
|---|---:|---:|---:|---|---:|---:|
| Convencional (AdamW) | 11.250 | 10.500 | -0.750 | [-16.633, +15.133] | -10.625 | 1.37× |
| Replay | 11.500 | 10.250 | -1.250 | [-10.780, +8.280] | -6.875 | 1.82× |
| DER++ | 13.500 | 11.250 | -2.250 | [-11.780, +7.280] | -3.437 | 1.87× |
| ER-ACE | 16.750 | 13.250 | -3.500 | [-41.619, +34.619] | +9.375 | 1.75× |

## Interpretação e limites

- ICs são pontuais; o JSON/CSV inclui p de acurácia ajustado por Holm para os quatro pares deste dataset. Não há correção entre datasets.
- Ganhos negativos e empates permanecem no relatório. Não se assume benefício universal.
- Mesmos parâmetros treináveis, épocas, passos, exemplos e memória de replay/logits são verificados dentro de cada par. Isso não iguala o custo computacional.
- Tempo observado é descritivo, sem medição isolada com aquecimento; a ordem dos métodos é fixa. FLOPs são estimativas.
- Bytes de replay/logits não são memória total. Pico de memória não foi medido.
- Fronteiras de tarefa conhecidas; inferência sem task ID. Baselines usam defaults, sem ajuste individual nesta suíte.
- Consulte pair_report.json para bootstrap pareado, sinais, demais métricas, configuração e hashes dos arquivos de origem.
