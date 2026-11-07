# Protocolo confirmatório: seleção contra lr uniforme pareado (Split-MNIST, 4 baselines)

Status: CONGELADO antes de qualquer run (commit que adiciona este arquivo). Autor: Venus,
aprovado pelo Fachel ("rode tudo", 06-07/11/2026). Gatilho: F4, F7 e F10 (exploratórios,
seeds da confirmação antiga) mostraram SlowHeat > lr uniforme pareado nos 4 baselines.

## Pergunta
Para cada baseline B, adicionar SlowHeat (seleção por importância funcional) rende mais
que adicionar a MESMA plasticidade média sem seleção (UniformPlasticityLevel)?

## Família de contrastes (4, Holm)
C_B = FAA(SlowHeat+B) - FAA(B+uniforme), pareado por seed, para
B em {replay, derpp, replay_calibrated, er_ace}. Braços:
- replay / slowheat_replay_hidden_beta_30_budget_0.25 / uniform_replay_hidden_beta_30_budget_0.25
- derpp / slowheat_derpp_hidden_beta_30_budget_0.25 / uniform_derpp_hidden_beta_30_budget_0.25
- replay_calibrated / slowheat_replay_hidden_beta_30_budget_0.25_calibrated / uniform_replay_hidden_beta_30_budget_0.25_calibrated
- er_ace / slowheat_er_ace_hidden_beta_30_budget_0.25 / uniform_er_ace_hidden_beta_30_budget_0.25

## Endpoint primário
final_average_accuracy (FAA) no teste, ao fim da última tarefa.

## Configuração
FROZEN_CONFIG de experiments/confirmatory_split_mnist.py, sem mudança, device=cpu,
todos os 12 braços no mesmo processo e device. Runner: scripts/run_confirm_uniform.py.

## Seeds (novas, nunca usadas)
[16000057, 16007976, 16015895, 16023814, 16031733, 16039652, 16047571, 16055490, 16063409, 16071328, 16079247, 16087166, 16095085, 16103004, 16110923, 16118842, 16126761, 16134680, 16142599, 16150518]
(16.000.057 + 7.919·k, k = 0..19; disjuntas de CONFIRMATORY_SEEDS e das seeds F4/F7/F10.)

## Análise (scripts/analyze_confirm_uniform.py, escrita antes da run)
- Teste t pareado bicaudal por contraste; Holm sobre os 4 p-valores, alfa 0,05.
- Confirmado para B: média de C_B > 0 e p ajustado (Holm) < 0,05.
- Refutado para B: IC 95% de C_B inclui 0 ou média <= 0.
- Secundários (descritivos, sem teste de família): teste do sinal, esquecimento médio,
  SlowHeat+B - B, B+uniforme - B.
- Nenhuma seed é descartada. Run que falhar é rerodada com a mesma seed e o fato é reportado.

## Limitações declaradas
Hiperparâmetros fixos (beta 30, budget 0,25) herdados da confirmação antiga, não tunados
por baseline. Um só benchmark (Split-MNIST, MLP). Não diz nada sobre LM.
