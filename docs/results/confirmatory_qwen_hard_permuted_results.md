F15 CONFIRMATÓRIO (09/11 09:50 - 11:37). Qwen2.5-0.5B, CLINC150 class-IL, 10 domínios, 120 passos por tarefa,
lr 1e-5, fp32, E* = 0,75 (configuração do F14). Braços: hard_b0.75 (top 25% unidades FFN por importância,
congeladas), hard_permuted_b0.75 (mesma contagem por camada, mesma máscara binária, identidade sorteada a
cada fronteira), reduced_lr. Seeds 300-309, GPUs 0 1 2, 10/10 exit=0, nenhuma descartada nem rerodada.
Protocolo congelado no commit dab50ed (docs/protocols/confirmatory_qwen_hard_permuted.md), branch
exp/f15-hard-permuted. Runs: results/f15_confirm_qwen_hard_permuted/seed<k>/manifest.json.

LEITURA CONGELADA: D = retenção t0 (hard - hard_permuted), t pareado bicaudal, alfa 0,05. CONFIRMADO se
média > 0 e p < 0,05; REFUTADO se IC 95% inclui 0 ou média <= 0.

SAÍDA DA ANÁLISE PRÉ-REGISTRADA (literal):
manifests: results/f15_confirm_qwen_hard_permuted/seed*/manifest.json
seeds completas (hard e hard_permuted): 10 [300, 301, 302, 303, 304, 305, 306, 307, 308, 309]
integridade (permutação real, mesma contagem protegida, máscara hard): OK

Primário retenção t0 hard_b0.75 - hard_permuted_b0.75: +4.10 pp IC95 [+0.61; +7.59] p 2.62e-02 a favor 9/10 p sinal 0.0215 -> CONFIRMADO

Secundários (descritivos):
  hard_b0.75 - hard_permuted_b0.75 final_average_accuracy: +4.13 pp [+0.90; +7.35] 8/10 p 0.018
  hard_b0.75 - hard_permuted_b0.75 acquisition_last_task: -0.10 pp [-1.78; +1.58] 6/10 p 0.896
  hard_b0.75 - reduced_lr retention_first_task: +6.07 pp [+1.90; +10.24] 10/10 p 0.009
  hard_permuted_b0.75 - reduced_lr retention_first_task: +1.97 pp [+0.23; +3.71] 5/10 p 0.031
  hard_b0.75 - reduced_lr final_average_accuracy: +3.82 pp [-0.61; +8.24] 8/10 p 0.083
  média hard_b0.75 (n 10): ret 0.0793 FAA 0.4428 aq 0.8433
  média hard_permuted_b0.75 (n 10): ret 0.0383 FAA 0.4015 aq 0.8443
  média reduced_lr (n 10): ret 0.0187 FAA 0.4046 aq 0.8473

APLICAÇÃO: D = +4,10 pp, p 0,026, IC [+0,61; +7,59] fora de 0, integridade OK. => CONFIRMADO. Sob máscara
dura e E pareado, congelar as unidades importantes retém mais a 1a tarefa que congelar unidades sorteadas.
Do ganho do hard sobre o reduced_lr (+6,07 pp), ~2/3 vêm da seleção (+4,10) e ~1/3 da forma (hard
sorteado - reduced_lr = +1,97 pp).

RESSALVAS:
- Margem estreita: p 0,026, limite inferior do IC +0,61 pp. O efeito se concentra em três seeds (303, 305,
  309: +11,7, +6,7, +13,3 pp); três seeds (300, 304, 308) estão perto do piso de retenção (<= 1,7%) nos
  três braços. Retenção absoluta continua baixa (7,9% no melhor braço).
- O sorteio é refeito a cada fronteira (o consolidate reconstrói o calor). O conjunto do hard selecionado
  tende a ser estável; o sorteado não: nenhuma unidade fica congelada a sequência inteira. Um aleatório
  FIXO seria um controle mais duro. Viés a favor de confirmar. Em teste no F16 (protocolo cc81913).
  Observado no smoke do F16: o conjunto do hard também muda entre fronteiras; o F16 grava um digest
  do conjunto por fronteira e vai medir quanto.
- Réplica do F14 nas seeds novas: hard - reduced_lr +6,07 pp de retenção (10/10, p 0,009), mas FAA
  +3,82 pp com IC tocando 0 (no F14 o FAA tinha IC fora de 0).
- Aquisição da última tarefa igual nos três braços (0,843-0,847).
- Mesmas limitações do F14: classificação, um modelo, um dataset, 120 passos, só E* = 0,75.
