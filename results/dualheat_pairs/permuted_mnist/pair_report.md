# Método versus método + DualHeat

Implementação: **Functional SlowHeat**, não a classe legada DualHeatMLP.

Execução de origem: `.` (caminho relativo à pasta deste relatório).

MLP [512, 256]; cenário `domain_incremental`; 10 épocas por tarefa.

Análise exploratória; 10 seeds pareadas. Não constitui confirmação independente.

Delta = com componente − sem componente. Acurácia e forgetting em pontos percentuais; esquecer menos só é útil se a aquisição for preservada.

| Método | Sem (%) | Com (%) | Delta (pp) | IC95% t (pp) | Delta forgetting (pp) | Tempo com/sem |
|---|---:|---:|---:|---|---:|---:|
| Convencional (AdamW) | 72.340 | 80.013 | +7.673 | [+5.193, +10.153] | -10.008 | 2.05× |
| Replay | 93.554 | 93.975 | +0.421 | [+0.238, +0.603] | -1.081 | 1.75× |
| DER++ | 95.633 | 95.734 | +0.101 | [+0.014, +0.189] | -0.462 | 1.65× |
| ER-ACE | 93.542 | 94.076 | +0.534 | [+0.341, +0.727] | -1.225 | 1.66× |

## Interpretação e limites

- ICs são pontuais; o JSON/CSV inclui p de acurácia ajustado por Holm para os quatro pares deste dataset. Não há correção entre datasets.
- Ganhos negativos e empates permanecem no relatório. Não se assume benefício universal.
- Mesmos parâmetros treináveis, épocas, passos, exemplos e memória de replay/logits são verificados dentro de cada par. Isso não iguala o custo computacional.
- Tempo observado é descritivo, sem medição isolada com aquecimento; a ordem dos métodos é fixa. FLOPs são estimativas.
- Bytes de replay/logits não são memória total. Pico de memória não foi medido.
- Fronteiras de tarefa conhecidas; inferência sem task ID. Baselines usam defaults, sem ajuste individual nesta suíte.
- Consulte pair_report.json para bootstrap pareado, sinais, demais métricas, configuração e hashes dos arquivos de origem.
