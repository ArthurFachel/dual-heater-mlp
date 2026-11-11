# Protocolo confirmatório T3c: seleção contra unidades aleatórias FIXAS de mesma forma (MLP)

Status: CONGELADO antes de qualquer seed deste protocolo (commit que adiciona este arquivo).
Autor: Venus, liberado pelo Fachel ("liberado", 11/11/2026). Plano: ~/Desktop/dual-heat-direcoes/plano-icml/01_tarefas.md, T3c.

## Por que este run existe
A Tabela 1 do artigo ICML decompõe o ganho em plasticidade (U - V), forma (R - U) e seleção (M - R).
Nas linhas MLP-S (Split-MNIST, 4 baselines) e MLP-P (Permuted-MNIST, hard) não existe braço R:
só M - U foi confirmado (F11) ou explorado (F8). Este run fecha as duas linhas com os quatro braços
no mesmo processo e nas mesmas seeds.

## O braço novo: random_fixed
experiments/split_mnist.py neste commit. Cada random_fixed_* espelha campo a campo o MethodSpec do
seu braço SlowHeat. Depois de cada consolidação, os valores de slow_heat de cada camada são
ordenados e entregues às unidades de UMA ordem aleatória por camada, sorteada uma vez por run com
semente (seed + 500000) * 1009 + índice da camada (mesmo offset do F16 e do 3c). As unidades
protegidas são order[:k] em toda fronteira (aninhadas se k cresce). Valores, contagem protegida e
dureza são os que a importância do próprio braço produziu; importance_memory não é tocada.
Cada fronteira grava em results.json (random_fixed_history): digest dos índices protegidos por
camada, contagem por camada e sobreposição com a própria seleção.

Verificação antes do congelamento (CPU do Lab):
- tests/test_random_fixed_mlp.py: 10 testes, verdes; vermelhos antes da implementação (ImportError).
- 8 mutações dirigidas, 8 mortas (runner sem reatribuição; ordem ressorteada por fronteira; ordem
  sem semente; valores não movidos; hard não binário; spec do ER-ACE errado; valores sem ordenar,
  sem prefixo fixo; histórico não gravado). Uma primeira versão da mutação "ressortear" era
  equivalente (o gerador re-semeado sorteia a mesma ordem) e foi trocada por uma que muda a semente
  por fronteira.
- Suíte inteira: 1108 passed, 2 failed, 15 skipped. As 2 falhas (test_hard_vs_soft_documents::
  test_the_cnn_report_names_the_cnn_backbone, test_visual_generalization::
  test_generalization_configs_preserve_scenario_semantics) falham igual sem a mudança.
- Smoke (2 seeds, dados mínimos, 1 época) nos dois settings: exit 0, integridade OK, sobreposição
  média de R com a própria seleção 0,76 (acaso 0,75 com orçamento 0,25).

## Settings e braços
MLP-S: FROZEN_CONFIG de experiments/confirmatory_split_mnist.py, device cpu, 16 braços no mesmo
processo: para B em {replay, derpp, replay_calibrated, er_ace}: B, SlowHeat+B, B+uniforme, B+random_fixed
(nomes *_hidden_beta_30_budget_0.25[_calibrated]).
MLP-P: suite_config("permuted_mnist", "mlp", "cpu") de experiments/hard_vs_soft.py, braços vanilla,
hard_freeze, uniform_hard_freeze, random_fixed_hard_freeze.
Runner: scripts/run_t3c_random_mlp.py --setting split|permuted. Saída results/t3c_random_fixed_mlp/{split,permuted}.

## Seeds (bandas novas; grep -w em json/py/md/sh/yaml de dual-heater-mlp, dh-uniform-ctrl,
dh-conc2x2, dh-qwen-f14, dh-paper e ncp: zero ocorrências)
MLP-S: 19.000.033 + 8.209·k, k = 0..19.
MLP-P: 19.500.017 + 8.317·k, k = 0..9.

## Integridade (lida ANTES do primário; falha invalida, não se reinterpreta)
- G1: em todo braço random_fixed e seed, uma reatribuição por consolidação, e o digest de cada camada
  igual ao da fronteira anterior sempre que a contagem protegida não mudou.
- G2: fração protegida média de R igual à de M em cada fronteira, tolerância 0,01.
- G3: nenhuma fronteira com sobreposição de R com a própria seleção >= 0,999.
- G4: todos os braços presentes em todas as seeds. Nenhuma seed descartada; run que falhar é rerodada
  com a mesma seed e o fato é reportado.

## Endpoint e família
Primário: seleção = FAA(M) - FAA(R), pareado por seed, final_average_accuracy no teste ao fim da
última tarefa. Família de 5 contrastes: MLP-S replay, derpp, replay_calibrated, er_ace; MLP-P hard.
t pareado bicaudal, Holm sobre os 5, alfa 0,05.
- CONFIRMADO no contraste: média > 0 e p Holm < 0,05.
- REFUTADO: IC 95% inclui 0 ou média <= 0. Se o IC inteiro ficar dentro de +-0,5 pp, reportar como
  nulo limitado.
Leitura de conjunto (declarada antes): nenhuma. Cada contraste é uma célula da Tabela 1 e entra como
sair, inclusive refutado.

## Secundários (descritivos, sem correção, todos reportados, nenhum promovido)
Forma R - U; plasticidade U - B; M - U (réplica do F11/F8 em banda nova); M - B; esquecimento M - R;
sinal; sobreposição média de R com a própria seleção.

## Análise
scripts/analyze_t3c.py (rodar com /home/fachel/novoenv/bin/python, que tem scipy), escrita antes da
run. Validação: scripts/validate_t3c_on_f11.py usa as mesmas funções load/paired e reproduz o F11
(M - U: +0,62 / +1,91 / +1,49 / +0,39 pp, 17/19/20/15 de 20); o pipeline inteiro (integridade +
primário + secundários) rodou no smoke.

## Custo estimado
CPU do Lab, nice 10, 6 threads por processo, dois processos (split e permuted) em paralelo.
F7 levou ~30 s/seed com 6 braços e F8 ~80 s/seed com 3: estimativa ~80 s/seed no MLP-S (16 braços,
~27 min) e ~110 s/seed no MLP-P (~18 min). Estimativa, não medida.

## Limitações declaradas
Mesmos hiperparâmetros do F11/F8 (beta 30, orçamento 0,25; hard do hard_vs_soft), não tunados.
Um benchmark por setting. O R aqui é a variante fixa (a mais dura, F16); a ressorteada não roda.
