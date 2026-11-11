# Fase 3b: confirmatório no SuperNI-15 (D_matched contra D_unmatched)

Status: PROTOCOLO CONGELADO antes de qualquer seed. Escrito em 11/11/2026 depois da calibração
(goals/calibracao_3b.md, commits 6d487a5 e ea4d029) e antes de lançar. Pedido do Fachel: "run the 10 seeds, all gpu".
Este arquivo é commitado antes do lançamento; o hash do commit vai no log do lançador. Mudança depois disso é desvio
declarado, numa seção nova no fim, nunca edição do que está aqui.

## Pergunta
Fase 3b (goals/estudo_fase3b.md, seção 4): em instruction tuning continual, o ganho de um método de CL sobre o
vanilla sobrevive quando o controle é um lr uniforme com a MESMA plasticidade de superfície E?

## O que roda
experiments/superni_cl.py no commit deste arquivo (branch exp/3b-superni-runner), Qwen2.5-0.5B + LoRA r 8 alpha 16 na
FFN, SuperNI-15 do SAPT, ordem 1, conjuntos completos (treino até 1.000, teste até 100 por tarefa), 1 época por
tarefa (escolha da calibração com a emenda post hoc declarada em goals/calibracao_3b.md), fp32, batch 2, eval batch 8,
lr 1e-4, geração gulosa até 50 tokens, --acquisition-margin 0.05 --min-vanilla-forgetting 0.05.

Seeds: 1000 a 1009 (banda nova; a calibração usou 950). Nenhuma seed é descartada nem trocada.

Braços, por seed (5):
- vanilla, exact (SlowHeat no LoRA), frozen_a_control (LoRA-FA): um processo, --arms vanilla exact frozen_a_control.
- lrc_exact: lr_control com escala = surface_plasticity do exact DA MESMA SEED, lida do manifest do processo acima
  (encadeado: só roda depois dele).
- lrc_lorafa: lr_control com escala 0.6148148148148148 = 2039808/3317760, a plasticidade de superfície do LoRA-FA, que
  é estrutural (parâmetros treináveis do LoRA-FA sobre os do vanilla) e não depende da seed.

Fila: scripts/confirm_3b_queue.py, um worker por GPU (0, 1, 2), CUDA_DEVICE_ORDER=PCI_BUS_ID,
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (só alocador, não muda conta). Saída results/3b_confirm/s<seed>/.

## Endpoints (AP e Fgt como no SAPT, ROUGE-L, escala fração)
Para X em {exact, frozen_a_control}, por seed:
  D_unmatched(X) = AP(X) - AP(vanilla)
  D_matched(X)   = AP(X) - AP(lr_control pareado a X)

Primário (o do estudo, seção 4): D_unmatched(X) - D_matched(X) > 0, pareado por seed. Note que isso é algebricamente
AP(lr_control_X) - AP(vanilla): o primário pergunta se o controle de lr sozinho já leva parte do ganho.
Teste: t pareado bilateral em 10 seeds, IC 95% t, Holm sobre os 2 métodos (alfa 0,05). Reportar n de seeds com sinal.

Secundários (sem correção de família, todos reportados, nenhum promovido depois):
  S1. D_matched(X) com IC 95% pareado, para cada X (a seleção bate o lr pareado?).
  S2. Os mesmos contrastes com Fgt no lugar de AP (sinal invertido: menos é melhor).
  S3. exact - frozen_a_control em AP e Fgt (E não pareado; só descritivo).
  S4. Ordenação dos braços por AP média, sem e com pareamento.

## Validade (checada antes de ler qualquer endpoint)
  V1. G-imp do exact passa em toda seed. Seed que falha fica no relatório, mas é marcada; se falhar em mais de 2
      seeds, o confirmatório inteiro é inválido.
  V2. Pareamento: |E(exact) - E(lrc_exact)| <= 0,01 e |E(LoRA-FA) - E(lrc_lorafa)| <= 0,01 em toda seed (por
      construção deveria dar 0; diferente disso é bug).
  V3. G-aq do vanilla >= 11/15 e Fgt do vanilla >= 0,05 (critério emendado da calibração), reportados por seed.
      Não excluem seed.
  V4. Processo que sai com código != 0 é relançado uma vez com a mesma seed e mesmos argumentos; o relançamento é
      registrado. Se falhar de novo, a seed fica faltando e o relatório diz isso.

## Leitura (declarada antes)
- Primário positivo e Holm significativo para X: o ganho de X sobre o vanilla é em parte plasticidade; reportar a
  fração D_matched/D_unmatched.
- S1 com IC acima de zero: a seleção de X ainda paga com E pareado. IC cruzando zero: não confirmado.
- Uma seed de calibração mostrou D_matched ~ +0,05 para os dois X; isso não entra em nenhuma conta.

## Custo esperado
Da calibração: principal ~2,0 h (vanilla 0,6 + exact 0,8 + LoRA-FA 0,6) e cada lr_control ~0,62 h, mais ~190 s de
zero-shot por processo. ~33 h-GPU, ~11 h nas 3 GPUs. Estimativa, não medida nesta banda.

## Limitações
Uma ordem de tarefas (ordem 1), um modelo (0.5B), 1 época. Qwen2.5 pode ter visto tarefas do SuperNI no pré-treino.
O critério de épocas teve emenda post hoc (goals/calibracao_3b.md).
