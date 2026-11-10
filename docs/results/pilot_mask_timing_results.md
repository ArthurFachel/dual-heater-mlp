PILOTO EXPLORATÓRIO P1 (09/11 16:25 - 19:55): tempo da máscara no Qwen2.5-0.5B (CLINC150 class-IL, 10 domínios,
120 passos, lr 1e-5, fp32, E* = 0,75). Plano e leitura declarados antes da run em goals/piloto_tempo_da_mascara.md
(commit c9fafda, branch exp/pilot-mask-timing). Seeds 500-509 (exploratórias), GPUs 0 1 2, 10/10 exit=0.
Análise descritiva: scripts/analyze_pilot.py. Runs: results/pilot_mask_timing/seed<k>/manifest.json.

SAÍDA (literal):
manifests: results/pilot_mask_timing/seed*/manifest.json
braços: ['hard_b0.75', 'hard_freeze1_b0.75', 'hard_freeze3_b0.75', 'hard_warmup20_b0.75', 'reduced_lr', 'reduced_lr_warmup20']
seeds completas: 10 [500, 501, 502, 503, 504, 505, 506, 507, 508, 509]
integridade hard_freeze1_b0.75: OK
integridade hard_freeze3_b0.75: OK

Médias (n = 10):
  hard_b0.75               FAA 0.4781  ret T1 0.0933  aq 0.8400  esq 0.4416
  hard_freeze1_b0.75       FAA 0.4649  ret T1 0.0897  aq 0.8370  esq 0.4549
  hard_freeze3_b0.75       FAA 0.4651  ret T1 0.0847  aq 0.8373  esq 0.4557
  hard_warmup20_b0.75      FAA 0.4490  ret T1 0.0973  aq 0.8213  esq 0.4566
  reduced_lr               FAA 0.4298  ret T1 0.0293  aq 0.8413  esq 0.4899
  reduced_lr_warmup20      FAA 0.4130  ret T1 0.0267  aq 0.8230  esq 0.4904

Freeze (endpoint FAA; candidato se >= +2 pp e >= 8/10, guarda aquisição >= -2 pp):
  hard_freeze1_b0.75 - hard  FAA                              -1.31 pp [ -2.71;  +0.08] a favor 3/10 p 0.062
  hard_freeze1_b0.75 - hard  retenção T1 (secundário)         -0.37 pp [ -1.13;  +0.40] a favor 3/10 p 0.307
  hard_freeze1_b0.75 - hard  aquisição (guarda)               -0.30 pp [ -1.32;  +0.72] a favor 5/10 p 0.523
    -> não candidato
  hard_freeze3_b0.75 - hard  FAA                              -1.30 pp [ -2.24;  -0.35] a favor 1/10 p 0.013
  hard_freeze3_b0.75 - hard  retenção T1 (secundário)         -0.87 pp [ -1.68;  -0.05] a favor 1/10 p 0.039
  hard_freeze3_b0.75 - hard  aquisição (guarda)               -0.27 pp [ -1.27;  +0.74] a favor 5/10 p 0.563
    -> não candidato

Warmup (candidato se a diferença-em-diferenças de FAA >= +2 pp e >= 8/10, guarda ok):
  hard_warmup20_b0.75 - hard  FAA                             -2.90 pp [ -5.14;  -0.67] a favor 2/10 p 0.016
  reduced_lr_warmup20 - reduced_lr  FAA                       -1.68 pp [ -5.40;  +2.04] a favor 3/10 p 0.335
  DiD FAA (warmup ajuda mais a máscara que o controle)        -1.23 pp [ -4.70;  +2.25] a favor 4/10 p 0.445
  hard_warmup20_b0.75 - hard  retenção T1                     +0.40 pp [ -2.69;  +3.49] a favor 5/10 p 0.777
  hard_warmup20_b0.75 - hard  aquisição (guarda)              -1.87 pp [ -3.63;  -0.10] a favor 2/10 p 0.040
    -> não candidato

Referência:
  hard - reduced_lr  FAA                                      +4.83 pp [ +2.03;  +7.63] a favor 10/10 p 0.004
  hard - reduced_lr  retenção T1                              +6.40 pp [ +4.05;  +8.75] a favor 9/10 p 0.000

APLICAÇÃO DA LEITURA DECLARADA: nenhum braço é candidato a confirmatório.
- Limitar as atualizações da máscara PIORA: freeze1 -1,31 pp de FAA (3/10), freeze3 -1,30 pp (1/10, IC fora
  de 0). Nem a retenção da T1 melhora (freeze1 -0,37 pp, freeze3 -0,87 pp), apesar de o congelamento manter
  as unidades protegidas após a T1. Integridade OK nos dois braços.
- O warmup da cabeça também PIORA: -2,90 pp de FAA (2/10) e fura a guarda de aquisição (-1,87 pp). Ele piora
  menos o controle (-1,68 pp), e a diferença-em-diferenças é -1,23 pp (4/10, n.s.).
- Referência, réplica pela terceira banda de seeds: hard - reduced_lr = +4,83 pp de FAA (10/10) e +6,40 pp
  de retenção T1 (9/10).

LEITURA:
- O recálculo da máscara a cada fronteira ajuda: proteger o que é importante para as tarefas novas vale
  mais do que manter o conjunto da T1. E manter esse conjunto nem protege a T1: o esquecimento da T1 não
  mora (só) nas unidades que a máscara trava, o que é consistente com o esquecimento difuso visto no NCP.
- Treinar só a cabeça nos primeiros passos gasta orçamento de aquisição sem comprar retenção neste regime
  (120 passos por tarefa).
- Linha "tempo da máscara" encerrada como resultado negativo, sem confirmatório.

RESSALVAS: exploratório, sem correção múltipla; X (1, 3) e N (20) não foram varridos; um modelo, um dataset,
classificação, 120 passos, só E* = 0,75.
