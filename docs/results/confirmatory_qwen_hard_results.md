F14 CONFIRMATÓRIO (08/11 05:13 - 06:53). Qwen2.5-0.5B como classificador class-IL no CLINC150 "plus",
10 domínios em sequência (banking, credit_cards, kitchen_and_dining, home, auto_and_commute, travel, utility,
work, small_talk, meta), 120 passos por tarefa, lr 1e-5, batch 2, seq 64, fp32, fine-tuning completo com
freeze_unbound_parameters (FFN + cabeça), capacity local, família E* = 0,75.
Braços: hard_b0.75 (25% das unidades FFN por camada congeladas por importância, máscara binária),
reduced_lr (lr x 0,75 em toda a superfície treinável, sem máscara), vanilla. Os 3 no mesmo processo e GPU.
10 seeds novas (200-209). GPUs 0, 1, 2 do Lab (1080 Ti x2, Titan Xp), uma seed por processo.
Código: worktree /mnt/B-SSD/fachel/dh-qwen-f14, branch exp/qwen-hard-confirm, protocolo congelado no commit
c495d14 (docs/protocols/confirmatory_qwen_hard.md), runner scripts/f14_confirm_qwen_hard.sh,
análise scripts/analyze_f14.py (novoenv, scipy), validada antes da run na confirm120 (seeds 10-19).
Runs: results/f14_confirm_qwen_hard/seed<k>/manifest.json (10/10, todas exit=0). Nenhuma seed descartada
nem rerodada.

GATILHO: análise post hoc da confirm120 (protocol_iso_plasticity.md) deu hard_b0.75 - reduced_lr = +6,30 pp
de retenção [+3,42; +9,18], 10/10, contraste fora da família pré-registrada daquela run. Por isso o hard
estava marcado como exploratório em goals/resultados_confirmacao.md.

LEITURA CONGELADA (protocolo, escrita antes da run): D = retention_first_task(hard_b0.75) -
retention_first_task(reduced_lr), pareado por seed, t pareado bicaudal, alfa 0,05, contraste único.
CONFIRMADO se média de D > 0 e p < 0,05; REFUTADO se IC 95% inclui 0 ou média <= 0.

SAÍDA DA ANÁLISE PRÉ-REGISTRADA (literal):
manifests: results/f14_confirm_qwen_hard/seed*/manifest.json
seeds completas: 10 [200, 201, 202, 203, 204, 205, 206, 207, 208, 209]

Primário retenção t0 hard_b0.75 - reduced_lr: +5.67 pp IC95 [+3.00; +8.33] p 9.65e-04 a favor 9/10 p sinal 0.0215 -> CONFIRMADO

Secundários (descritivos):
  hard_b0.75 - reduced_lr final_average_accuracy: +5.67 pp [+3.11; +8.24] 9/10 p 0.001
  hard_b0.75 - reduced_lr acquisition_last_task: -0.10 pp [-1.99; +1.79] 7/10 p 0.907
  hard_b0.75 - vanilla retention_first_task: +7.27 pp [+3.58; +10.95] 9/10 p 0.002
  reduced_lr - vanilla retention_first_task: +1.60 pp [+0.29; +2.91] 8/10 p 0.022
  média hard_b0.75: ret 0.0900 FAA 0.4872
  média reduced_lr: ret 0.0333 FAA 0.4305
  média vanilla: ret 0.0173 FAA 0.4136

APLICAÇÃO: média de D +5,67 pp > 0, p 9,7e-4 < 0,05, IC fora de 0. => CONFIRMADO. O efeito do braço hard
deixa de ser só exploratório: em seeds novas e protocolo congelado ele retém a 1a tarefa melhor que o lr
reduzido com a mesma plasticidade média. Exploratório +6,30 pp (10/10) -> confirmatório +5,67 pp (9/10).

RESSALVAS:
- O contraste é "como" (máscara dura em unidades selecionadas) contra "quanto" (mesmo E espalhado
  uniformemente). Ele NÃO separa seleção de forma: o hard protege as unidades mais importantes E é binário.
  Um hard com unidades aleatórias (mesmo E, mesma dureza) é o controle que falta para dizer que a seleção
  importa sob máscara dura. Na confirm120 o iso-permutado soft não sobreviveu a Holm.
- Aquisição da última tarefa não cai (-0,10 pp, IC [-1,99; +1,79]): sem custo de plasticidade visível.
- Retenção absoluta baixa: 9,0% no hard contra 3,3% (reduced_lr) e 1,7% (vanilla). O regime ainda é de
  esquecimento severo; o ganho é relativo.
- Classificação, não geração. Um modelo, um dataset, 120 passos por tarefa, só E* = 0,75.
- Fine-tuning completo; não é o host LoRA do R-C. O LoRA-FA (frozen_a_control) não tem análogo aqui.
- GPUs 1080 Ti e Titan Xp misturadas entre seeds (não dentro de uma seed).
