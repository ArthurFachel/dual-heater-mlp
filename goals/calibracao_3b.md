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
