# Protocolo confirmatório: Qwen2.5-0.5B, máscara dura vs lr reduzido pareado (F14)

Status: CONGELADO antes de qualquer run (commit que adiciona este arquivo). Autor: Venus, autorizado
pelo Fachel ("consegue lançar o f14 assim que terminar as seeds?", 08/11/2026 ~04:35).
Gatilho: análise POST HOC da run confirmatória results/qwen_iso_plasticity/confirm120_seed{10..19}
(protocolo goals/protocol_iso_plasticity.md): hard_b0.75 - reduced_lr = +6,30 pp de retenção da 1a
tarefa [+3,42; +9,18], 10/10; esse contraste não estava na família pré-registrada daquela run.

## Pergunta
Num segundo LM (Qwen2.5-0.5B como classificador class-IL no CLINC150), a máscara dura do SlowHeat
(25% das unidades registradas congeladas por importância, plasticidade 0,75) retém mais a primeira
tarefa que o lr reduzido pareado (lr x 0,75 em toda a superfície treinável, sem máscara)?

## Configuração (idêntica à confirm120, só com filtro de braços)
experiments/qwen_iso_plasticity.py (worktree dh-qwen-f14, branch exp/qwen-hard-confirm, base aed2957 +
opção --arms, que só filtra braços declarados e grava protocol.arms_filter):
--domains banking credit_cards kitchen_and_dining home auto_and_commute travel utility work small_talk meta
--steps-per-task 120 --learning-rate 1e-5 --batch-size 2 --max-length 64 --target-plasticity 0.75
--arms hard_b0.75 reduced_lr vanilla --device cuda (fp32, capacity local, beta per_boundary).
Um processo por seed (os 3 braços no mesmo processo e na mesma GPU). Runner scripts/f14_confirm_qwen_hard.sh;
saída results/f14_confirm_qwen_hard/seed<k>/manifest.json.

## Seeds (novas, nunca usadas no Qwen)
200, 201, ..., 209 (10 seeds). A confirm120 usou 10-19.

## Endpoint primário
D = retention_first_task(hard_b0.75) - retention_first_task(reduced_lr), pareado por seed.

## Análise (scripts/analyze_f14.py, escrita e validada antes da run na confirm120)
- Teste t pareado bicaudal de D, alfa 0,05 (contraste único).
- CONFIRMADO: média de D > 0 e p < 0,05. REFUTADO: IC 95% inclui 0 ou média <= 0. Senão inconclusivo.
- Secundários (descritivos): sinal; FAA e aquisição da última tarefa (hard - reduced_lr);
  hard - vanilla e reduced_lr - vanilla na retenção.
- Nenhuma seed descartada; run que falhar é rerodada com a mesma seed e o fato é reportado.

## Limitações declaradas
Classificação class-IL, não perda de próximo token. Um modelo, um dataset, 120 passos por tarefa.
Só a família E*=0,75. reduced_lr espalha a mesma plasticidade média de modo uniforme, mas sobre toda a
superfície treinável, enquanto o hard atua nas unidades registradas: a média casa, a distribuição é o
contraste. GPUs 1080 Ti e Titan Xp misturadas entre seeds (não dentro de uma seed).
