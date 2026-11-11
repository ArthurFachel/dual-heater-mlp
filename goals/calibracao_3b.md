# Calibração da Fase 3b: épocas por tarefa no SuperNI-15

Status: CALIBRAÇÃO (3b.2), não é protocolo confirmatório. Critério de escolha escrito ANTES da run.
Pedido do Fachel em 10/11/2026 ("pode usar todas as gpus, quantas épocas seria ideal? creio que entre 1-5").

## O que roda
experiments/superni_cl.py (branch exp/3b-superni-runner), Qwen2.5-0.5B + LoRA (r 8, alpha 16, FFN), SuperNI-15 do
SAPT, ordem 1, conjuntos completos (treino até 1.000, teste até 100 por tarefa), fp32, batch 2, eval batch 8,
lr 1e-4, geração gulosa até 50 tokens. Seed 950 (fora de todas as bandas usadas). Uma run por GPU:
- GPU 0: 1 época por tarefa
- GPU 1: 3 épocas por tarefa
- GPU 2: 5 épocas por tarefa
Braços em cada run: vanilla, exact (SlowHeat no LoRA), frozen_a_control (LoRA-FA). Os dois lr_control (pareados ao
E medido do exact e do LoRA-FA) rodam depois, só para o número de épocas escolhido.
Saída: results/3b_calibration/ep<N>/manifest.json; log results/3b_calibration/ep<N>.log.

## Critério de escolha (declarado antes; só o vanilla decide, para não escolher o regime que favorece o método)
1. G-aq: no vanilla, ROUGE-L logo após treinar a tarefa t menos o zero-shot da tarefa t >= 0,05 em pelo menos 12
   das 15 tarefas.
2. G-disp: esquecimento do vanilla (Fgt do SAPT, média sobre t < T de melhor-já-visto menos final) >= 0,05.
3. Escolhe-se o MENOR número de épocas que passa 1 e 2.
4. Se nenhum passa 2: o regime não esquece o bastante para a pergunta; não se escolhe por AP, replaneja-se
   (ordem 2, lr maior) com o Fachel.
5. Se nenhum passa 1: aumentar lr ou épocas, replanejar.
6. Validade: G-imp do exact (importância viva em todas as camadas, todas as fronteiras) tem de passar; se
   falhar, a run é inválida para aquele número de épocas.
Também se registra o custo (horas por braço) para dimensionar o confirmatório.

## Limitações
Uma seed: a calibração mede regime e custo, não efeito. O AP e o Fgt do exact e do LoRA-FA aparecem no manifest, mas
não entram na escolha.

## Resultado e emenda post hoc (10/11/2026, decisão do Fachel)
EMENDA FEITA DEPOIS DE VER OS DADOS. O critério acima continua sendo o pré-registrado; o que segue é desvio declarado.

Resultado pelo critério original (manifests ep1 e ep3, commit c103f4b, seed 950):
| épocas | G-aq vanilla | Fgt vanilla | G-imp exact | AP vanilla / exact / LoRA-FA | horas por braço (vanilla / exact / LoRA-FA) |
|---|---|---|---|---|---|
| 1 | 11/15 | 0,133 | passa | 0,388 / 0,453 / 0,448 | 0,6 / 0,8 / 0,6 |
| 3 | 11/15 | 0,172 | passa | 0,345 / 0,424 / 0,410 | 1,4 / 1,7 / 1,4 |
| 5 | sem manifest: vanilla terminou (AP 0,195, Fgt 0,329), exact deu OOM na geração (Titan Xp) | | | | |

Nenhum número de épocas passa o item 1 (12/15). Pela regra 5 caberia aumentar lr ou épocas e replanejar.

Por que não se seguiu a regra 5: três tarefas de geração aberta não respondem ao treino em nenhum orçamento testado
(delta ROUGE-L ep1 / ep3): task1729_personachat_generate_next +0,018 / +0,010; task1590_diplomacy_text_generation
-0,006 / -0,003; task511_reddit_tifu_long_text_summarization +0,044 / +0,047. A quarta falha troca de tarefa e fica
colada no limiar (samsum +0,044 no ep1; multi_woz +0,049 no ep3). Mais épocas não muda o G-aq (11/15 nos dois) e ep5
degrada o vanilla (AP 0,195), então o limiar de 12/15 mede o teto do ROUGE-L nessas tarefas, não a falta de treino.

Emenda: G-aq passa a exigir >= 11/15 tarefas com delta >= 0,05. G-disp, G-imp e as 15 tarefas do benchmark (AP e Fgt)
ficam iguais. Com a emenda ep1 e ep3 passam; pela regra 3 (menor número de épocas) escolhe-se 1 época por tarefa.
Essa escolha vale para a calibração; qualquer texto que use o 3b deve citar este desvio.

Próximo passo previsto pelo plano: os dois lr_control (pareados ao E medido do exact e do LoRA-FA) em 1 época.

## lr_control em 1 época (10/11/2026, commit ce39311, seed 950)
Escala = E medido do braço pareado no manifest ep1 (scripts/lr_control_3b.sh). Saída: results/3b_calibration/ep1_lrc_*.
| par | AP tratado | AP lr_control | d_AP | Fgt tratado | Fgt lr_control | E | horas |
|---|---|---|---|---|---|---|---|
| exact vs lrc_exact | 0,453 | 0,406 | +0,047 | 0,061 | 0,109 | 0,560 | 0,6 |
| LoRA-FA vs lrc_lorafa | 0,448 | 0,395 | +0,053 | 0,073 | 0,120 | 0,615 | 0,6 |
Uma seed, calibração: mede regime e custo, não efeito. Os dois tratados batem o controle de lr pareado por margem
parecida, e exact contra LoRA-FA fica em +0,005 de AP; a pergunta do confirmatório é essa diferença, não o contraste
com lr_control. Custo por seed com os 5 braços em 1 época: cerca de 3,2 h-GPU.
