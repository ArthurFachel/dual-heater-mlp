F16 CONFIRMATÓRIO (09/11 12:45 - 14:40). Qwen2.5-0.5B, CLINC150 class-IL, 10 domínios, 120 passos por tarefa,
lr 1e-5, fp32, E* = 0,75 (configuração do F14/F15). Braços: hard_b0.75, hard_random_fixed_b0.75 (k primeiras
unidades de uma ordem aleatória por camada sorteada uma vez por run, mesma contagem do hard), hard_permuted_b0.75
(ressorteado a cada fronteira, como no F15). Seeds 400-409, GPUs 0 1 2, 10/10 exit=0, nenhuma descartada nem
rerodada. Protocolo congelado no commit cc81913 (docs/protocols/confirmatory_qwen_hard_random_fixed.md), branch
exp/f16-hard-random-fixed. Runs: results/f16_confirm_qwen_hard_random_fixed/seed<k>/manifest.json.

LEITURA CONGELADA: D = retenção t0 (hard - aleatório fixo), t pareado bicaudal, alfa 0,05. CONFIRMADO se
média > 0 e p < 0,05; REFUTADO se IC 95% inclui 0 ou média <= 0.

SAÍDA DA ANÁLISE PRÉ-REGISTRADA (literal):
manifests: results/f16_confirm_qwen_hard_random_fixed/seed*/manifest.json
seeds completas (hard e fixo): 10 [400, 401, 402, 403, 404, 405, 406, 407, 408, 409]
integridade (máscara hard, contagem igual, fixo constante e != hard, permutado ressorteado): OK

Primário retenção t0 hard_b0.75 - hard_random_fixed_b0.75: +1.50 pp IC95 [-0.07; +3.07] p 5.96e-02 a favor 8/10 p sinal 0.1094 -> REFUTADO

Secundários (descritivos):
  hard_b0.75 - hard_random_fixed_b0.75 final_average_accuracy: +2.81 pp [+0.24; +5.38] 9/10 p 0.035
  hard_b0.75 - hard_random_fixed_b0.75 acquisition_last_task: -0.97 pp [-3.14; +1.21] 4/10 p 0.342
  hard_random_fixed_b0.75 - hard_permuted_b0.75 retention_first_task: +0.30 pp [-1.06; +1.66] 5/10 p 0.629
  hard_b0.75 - hard_permuted_b0.75 retention_first_task: +1.80 pp [+0.38; +3.22] 9/10 p 0.019
  média hard_b0.75 (n 10): ret 0.0480 FAA 0.4192 aq 0.8180
  média hard_random_fixed_b0.75 (n 10): ret 0.0330 FAA 0.3911 aq 0.8277
  média hard_permuted_b0.75 (n 10): ret 0.0300 FAA 0.3819 aq 0.8263

APLICAÇÃO: D = +1,50 pp, IC [-0,07; +3,07] inclui 0, p 0,060. => REFUTADO pela regra congelada. A seleção
não vence o aleatório fixo com significância nesta banda de seeds.

LEITURA:
- A suspeita do F15 (o ressorteio enfraquece o controle) NÃO se sustentou: fixo - ressorteado = +0,30 pp
  [-1,06; +1,66], 5/10. Os dois aleatórios são equivalentes aqui.
- O que encolheu foi o próprio efeito da seleção: hard - ressorteado nesta banda = +1,80 pp [+0,38; +3,22]
  (9/10, p 0,019), contra +4,10 pp no F15. O sinal é consistente (8/10 e 9/10 a favor), o tamanho é
  pequeno e instável entre bandas de seeds.
- A retenção absoluta caiu nesta banda: hard 4,8% (F15 7,9%, F14 9,0%). Seis das dez seeds têm o hard em
  3% ou menos. O contraste está sendo medido muito perto do piso, o que limita o poder.
- FAA (secundário): hard - fixo +2,81 pp [+0,24; +5,38], 9/10.
- Consequência para a leitura do F15: "cerca de 2/3 do ganho do hard sobre o lr reduzido vem da seleção"
  não se sustenta como estimativa estável. O que se sustenta: a máscara dura bate o lr reduzido (F14, e
  réplica no F15); a seleção dentro da máscara dura tem sinal positivo, mas efeito pequeno e não robusto.
- Observação de mecanismo: o protected_digest do hard mudou em todas as fronteiras comparadas, nas 10
  seeds. A sobreposição entre fronteiras está sendo medida à parte (results/explore_hard_churn, seed 900,
  exploratório).

RESSALVAS: as do F14/F15 (classificação, um modelo, um dataset, 120 passos, só E* = 0,75).
