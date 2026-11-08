F12 CONFIRMATÓRIO (07/11 18:29 - ~20:25). BERT CLINC150 (google/bert_uncased_L-4_H-256_A-4), CLINC150
"plus" sem OOS, sem replay, class-incremental de 2 tarefas (15 intenções cada), 4 épocas por tarefa,
lr 5e-5, batch 32, seq 128, avaliação na validação. Orçamento 0,50 (FFN e atenção), máscara soft força 30,
critério magnitude. Braços: vanilla, slowheat_magnitude_soft_b30, slowheat_uniform_soft_b30, no mesmo processo.
20 seeds novas (17.000.029 + 8.011k). CPU do Lab, 5 processos (1 por par) x 4 threads, nice 10, 99-116 min por par.
Código: worktree /mnt/B-SSD/fachel/dh-conc2x2, branch exp/concentration-2x2, protocolo congelado no commit
77ebece (docs/protocols/confirmatory_bert_soft30.md), runner scripts/run_confirm_bert_soft30.py,
análise scripts/analyze_confirm_bert_soft30.py (novoenv, scipy).
Runs: results/confirm_bert_soft30/<par>/budget_0.5/ (300/300 results.json, 20/20 seeds x 3 braços em cada par).
Nenhuma seed descartada nem rerodada.

LEITURA CONGELADA (protocolo, escrita antes da run): endpoint primário retenção T1 (SlowHeat - uniforme soft),
t pareado bicaudal por par, Holm sobre 5 pares, alfa 0,05. Confirmado no par se média > 0 e p Holm < 0,05;
refutado se IC 95% inclui 0 ou média <= 0. Geral: CONFIRMA se >= 4 de 5 pares confirmados.

SAÍDA DA ANÁLISE PRÉ-REGISTRADA (literal):
Endpoint primário: task1_retention, SlowHeat soft b30 - uniforme soft b30 (t pareado, Holm sobre 5 pares)
banking -> credit_cards          n 20  +4.07 pp IC95 [+2.98; +5.15] p 2.17e-07 Holm 4.61e-07 a favor 19/20 p sinal 0.0000 -> CONFIRMADO
auto_and_commute -> travel       n 20  +4.40 pp IC95 [+3.10; +5.70] p 1.01e-06 Holm 1.01e-06 a favor 20/20 p sinal 0.0000 -> CONFIRMADO
kitchen_and_dining -> home       n 20  +3.20 pp IC95 [+2.37; +4.03] p 1.54e-07 Holm 4.61e-07 a favor 19/20 p sinal 0.0000 -> CONFIRMADO
small_talk -> meta               n 20  +2.93 pp IC95 [+2.22; +3.65] p 6.00e-08 Holm 2.40e-07 a favor 20/20 p sinal 0.0000 -> CONFIRMADO
utility -> work                  n 20  +5.20 pp IC95 [+4.05; +6.35] p 1.27e-08 Holm 6.36e-08 a favor 20/20 p sinal 0.0000 -> CONFIRMADO

Pares confirmados: 5/5. Leitura geral (>= 4 de 5): CONFIRMA

Secundários (descritivos):
banking -> credit_cards          ret T1 VA 0.0910 SH 0.2063 UN 0.1657 | fina SH-UN +1.49 [+0.85; +2.14] 18/20 | task SH-UN -1.08 [-1.67; -0.50] 3/20
                                 FAA VA 0.4716 SH 0.5037 UN 0.4887 | SH-VA ret +11.53 pp, UN-VA ret +7.47 pp
auto_and_commute -> travel       ret T1 VA 0.0498 SH 0.1635 UN 0.1195 | fina SH-UN +1.83 [+1.20; +2.45] 20/20 | task SH-UN -0.75 [-1.09; -0.41] 2/20
                                 FAA VA 0.5094 SH 0.5516 UN 0.5333 | SH-VA ret +11.37 pp, UN-VA ret +6.97 pp
kitchen_and_dining -> home       ret T1 VA 0.0565 SH 0.1433 UN 0.1113 | fina SH-UN +0.88 [+0.32; +1.44] 16/20 | task SH-UN -1.43 [-2.07; -0.80] 2/20
                                 FAA VA 0.4014 SH 0.4109 UN 0.4021 | SH-VA ret +8.68 pp, UN-VA ret +5.48 pp
small_talk -> meta               ret T1 VA 0.0348 SH 0.1193 UN 0.0900 | fina SH-UN +0.84 [+0.45; +1.23] 17/20 | task SH-UN -1.25 [-1.61; -0.89] 1/20
                                 FAA VA 0.4191 SH 0.4342 UN 0.4258 | SH-VA ret +8.45 pp, UN-VA ret +5.52 pp
utility -> work                  ret T1 VA 0.1588 SH 0.3318 UN 0.2798 | fina SH-UN +2.40 [+1.82; +2.98] 20/20 | task SH-UN -0.40 [-0.85; +0.05] 7/20
                                 FAA VA 0.5082 SH 0.5794 UN 0.5554 | SH-VA ret +17.30 pp, UN-VA ret +12.10 pp

APLICAÇÃO: 5/5 pares confirmados (limiar 4 de 5), retenção T1 +2,93 a +5,20 pp, 19-20/20 seeds a favor,
p Holm <= 1,0e-6 em todos. => CONFIRMA. O ganho exploratório do BERT se mantém em seeds novas com
protocolo congelado, como o F11 fez no MLP.

Comparação com o exploratório (10 seeds, banda 14): banking +5,00 -> +4,07; auto +5,17 -> +4,40;
kitchen +2,70 -> +3,20; small_talk +2,57 -> +2,93; utility +3,80 -> +5,20. Mesma ordem de grandeza;
dois pares encolheram, três cresceram.

RESSALVAS:
- Troca retenção x aquisição confirmada como descritiva: aquisição da T2 cai 0,4 a 1,4 pp (SH < UN em
  13-19/20 seeds; no utility o IC toca 0).
- FAA (secundário, descritivo) a favor do SlowHeat nos 5 pares, +0,84 a +2,40 pp, IC fora de 0 nos 5.
- Retenção absoluta continua baixa (11,9% a 33,2% com SlowHeat); o vanilla esquece quase tudo (3,5-15,9%).
- Um modelo pequeno, um dataset, 2 tarefas, sem replay, força e orçamento escolhidos no exploratório.
- Rodou na CPU; os exploratórios também (F5) ou em device não verificado (soft_2x2).
