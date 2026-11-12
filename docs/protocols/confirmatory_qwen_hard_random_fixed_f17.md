# Protocolo confirmatório F17: máscara dura selecionada vs aleatória FIXA, 20 seeds com poder calculado (Qwen2.5-0.5B)

Status: CONGELADO antes de qualquer seed (commit que adiciona este arquivo). Autor: Venus, liberado pelo
Fachel ("rode o f17 primeiro", 11/11/2026). Proposta original: ~/Desktop/dual-heat-direcoes/fila.md, F17 (09/11).

## Por que este run existe, e o que ele NÃO muda
O F16 (docs/protocols/confirmatory_qwen_hard_random_fixed.md @ cc81913, seeds 400-409) foi REFUTADO pela regra
congelada: retenção t0 hard - aleatório fixo +1,50 pp, IC95 [-0,07; +3,07], 8/10, p 0,060. A média é positiva e o
IC quase exclui zero; com 10 seeds o teste tinha pouco poder para um efeito desse tamanho. O F17 repete o mesmo
desenho com n calculado para ~80% de poder.
O F16 continua REFUTADO no registro, em qualquer resultado do F17. Este é uma segunda tentativa declarada como tal;
o artigo reporta os dois runs, com este parágrafo.

## Conta de poder (feita antes, a partir do F16 apenas)
Efeito plausível d = 1,5 pp (média do F16). Desvio-padrão da diferença pareada estimado do IC do F16:
meia-largura 1,57 pp = t(0,975; 9) x sd / sqrt(10) -> sd ~ 2,2 pp.
n = ((z_0,975 + z_0,80) x sd / d)^2 = ((1,96 + 0,84) x 2,2 / 1,5)^2 ~ 16,9; correção para t ~ 19. n = 20.

## Desenho (idêntico ao F16; nada mudou no código)
Código: o commit de F16 (branch exp/f16-hard-random-fixed, código de cc81913; os commits seguintes até 460b9ae
só tocam docs/ e results/). Executado num worktree separado (/mnt/B-SSD/fachel/dh-f17) para não usar o código
posterior do branch exp/3c-random-fixed, que mudou experiments/qwen_iso_plasticity.py.
experiments/qwen_iso_plasticity.py, 10 domínios CLINC150 (banking credit_cards kitchen_and_dining home
auto_and_commute travel utility work small_talk meta), 120 passos/tarefa, lr 1e-5, batch 2, max_length 64,
E* = 0,75, fp32, um processo por seed, braços hard_b0.75, hard_random_fixed_b0.75, hard_permuted_b0.75.
Runner: scripts/f17_confirm_qwen_hard_random_fixed.sh (cópia do runner do F16 trocando só o diretório de trabalho,
ROOT, a lista de seeds e o marcador de fim).

## Seeds
600 a 619 (20). Verificadas sem uso: nenhum diretório results/*/seed6xx ou s6xx, nenhum manifest com seed 600-619
em dh-qwen-f14 e dual-heater-mlp, nenhuma menção em docs/protocols ou goals. A proposta de 09/11 citava 500-519;
500-509 foram gastas pelo piloto P1 (exploratório) e por isso não são usadas.

## Integridade (antes do primário; falha invalida a run, não se reinterpreta)
As mesmas do F16, checadas por scripts/analyze_f16.py: máscara dura binária, contagem protegida igual entre os três
braços, conjunto do aleatório fixo constante entre fronteiras (digest) e diferente do hard, aleatório permutado
ressorteado a cada fronteira.

## Endpoint e regra (congelados)
D = retenção da tarefa 0 (hard_b0.75 - hard_random_fixed_b0.75), por seed, no eval final. t pareado bicaudal,
alfa 0,05, teste único (sem família).
- CONFIRMADO se média > 0 e p < 0,05.
- REFUTADO se IC 95% inclui 0 ou média <= 0.
Secundários (descritivos, todos reportados): FAA e aquisição da última tarefa (hard - fixo); fixo - permutado;
hard - permutado; teste do sinal; retenção absoluta dos três braços.
Nenhuma seed é descartada; seed que falhar é rerodada com a mesma seed e o fato é reportado. Sem parada antecipada
nem leitura do primário antes das 20 seeds.

## Análise
scripts/analyze_f16.py "results/f17_confirm_qwen_hard_random_fixed/seed*/manifest.json" (novoenv), o mesmo script
do F16, sem mudança. Validação antes da run: no worktree do F17 ele reproduz o F16 (+1,50 pp, 8/10).

## Custo
F16: ~30 min por seed. 20 seeds em 3 GPUs (0, 1, 2; planilha MOTOZ: Fachel em uso, sem reserva): ~3,5 h.
