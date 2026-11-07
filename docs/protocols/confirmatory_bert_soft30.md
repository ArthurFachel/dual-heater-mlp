# Protocolo confirmatório: BERT CLINC150, SlowHeat soft força 30 vs uniforme soft, 5 pares de domínio (F12)

Status: CONGELADO antes de qualquer run (commit que adiciona este arquivo). Autor: Venus,
aprovado pelo Fachel (fila.md F12 "OK" e "execute o F12", 07/11/2026). Gatilho: o par original
banking -> credit_cards (+5,0 pp de retenção T1, 10/10, results/soft_2x2) e o F5 (4 pares, +2,57 a
+5,17 pp, 10/10 em cada, results/domain_pairs), ambos exploratórios na banda 14.000.003+.

## Pergunta
Em BERT (google/bert_uncased_L-4_H-256_A-4) no CLINC150 class-incremental de 2 tarefas, a máscara
soft do SlowHeat (|z|, força 30, orçamento 0,50) retém mais a tarefa 1 que um controle com o MESMO
fator soft médio aplicado sem seleção (uniform_soft)?

## Pares (5) e família de contrastes (5, Holm)
banking -> credit_cards, auto_and_commute -> travel, kitchen_and_dining -> home,
small_talk -> meta, utility -> work.
C_P = task1_retention(slowheat_magnitude_soft_b30) - task1_retention(slowheat_uniform_soft_b30),
pareado por seed, para cada par P.

## Braços (mesmo processo, mesmo device, por par)
- vanilla (referência; run_diagnostic exige)
- slowheat_magnitude_soft_b30: slowheat_ffn_attention, força 30, máscara soft, critério magnitude
- slowheat_uniform_soft_b30: slowheat_ffn_attention, força 30, uniform_soft, critério magnitude

## Endpoint primário
task1_retention: acurácia na tarefa 1 ao fim da tarefa 2 (validação; evaluate_test=False, igual ao
exploratório).

## Configuração
SplitCLINC150Config padrão (4 épocas por tarefa, lr 5e-5, batch 32, seq 128, task_limit 2), orçamento
0,50 em FFN e atenção, device=cpu. O par é movido para o início da lista de domínios (mesma função
reorder do F5; para banking -> credit_cards a ordem é a padrão). Runner:
scripts/run_confirm_bert_soft30.py, um processo por par, 4 threads, nice 10, os 5 em paralelo
(scripts/launch_confirm_bert_soft30.sh). Saída: results/confirm_bert_soft30/<T1>__<T2>/budget_0.5/.
Desvio declarado da proposta na fila: 5 processos x 4 threads em vez de 6 threads seriais (só muda
tempo de parede; os 3 braços de cada par seguem no mesmo processo).

## Seeds (novas, nunca usadas)
[17000029, 17008040, 17016051, 17024062, 17032073, 17040084, 17048095, 17056106, 17064117, 17072128, 17080139, 17088150, 17096161, 17104172, 17112183, 17120194, 17128205, 17136216, 17144227, 17152238]
(17.000.029 + 8.011·k, k = 0..19; banda 17 sem uso em goals/, scripts/ e experiments/ de
dual-heater-mlp, dh-conc2x2 e dh-uniform-ctrl; disjunta de 11-14.xxx.xxx e de 16.000.057+.)

## Análise (scripts/analyze_confirm_bert_soft30.py, escrita e validada antes da run)
- Teste t pareado bicaudal de C_P por par; Holm sobre os 5 p-valores, alfa 0,05.
- Confirmado no par P: média de C_P > 0 e p ajustado (Holm) < 0,05.
- Refutado no par P: IC 95% de C_P inclui 0 ou média <= 0. Outro caso: inconclusivo.
- Leitura geral: CONFIRMA se >= 4 dos 5 pares forem confirmados; caso contrário NÃO CONFIRMA.
- Secundários (descritivos, sem teste de família): FAA e aquisição da T2 (SlowHeat - uniforme),
  teste do sinal, SlowHeat - vanilla e uniforme - vanilla.
- Nenhuma seed é descartada. Run que falhar é rerodada com a mesma seed e o fato é reportado.

## Limitações declaradas
Um só modelo pequeno (BERT 4 camadas, 256), um dataset, 2 tarefas, sem replay. Força 30 e orçamento
0,50 escolhidos no exploratório (soft_2x2) e não tunados por par. Avaliação na validação, não no teste.
Já se sabe do exploratório que a aquisição da T2 cai 0,6 a 1,7 pp: o protocolo não testa FAA como
primário e não diz nada sobre LMs maiores.
