# Fase 3c: seleção do exact contra linhas aleatórias fixas (SuperNI-15, LoRA)

Status: PROTOCOLO CONGELADO antes de qualquer seed deste braço. Escrito em 11/11/2026, depois do confirmatório 3b
(goals/protocol_3b_confirmatory.md @ deea8d2, análise results/3b_confirm/analysis_preregistered.txt @ 0e56045) e
antes de lançar. Mudança depois do commit deste arquivo é desvio declarado, numa seção nova no fim.

## Por que este run existe
O 3b mediu plasticidade (lr_control - vanilla) e o ganho do exact sobre o lr pareado, mas não separou forma de
seleção. A única comparação com o exact foi o LoRA-FA (S3: +0,0028 AP, IC95 [-0,0026; +0,0082], 5/10), que congela
A inteiro e não é um sorteio das linhas que o exact escolhe. Este run fecha a célula de seleção da Tabela 1 para o
host generativo com o controle que o F16 usou: as mesmas unidades protegidas em quantidade e intensidade, em
linhas aleatórias fixas.

Fato que limita o que este run pode mostrar, medido nos manifests do 3b antes de escrever este protocolo: a máscara
do exact em B fica em média 0,91 (0,903 a 0,915 em todas as 15 fronteiras das 10 seeds). O exact tira só ~9% da
plasticidade de B; o resto do seu E (0,560 contra 0,615 do LoRA-FA) vem de congelar A. O termo de seleção aqui é,
por construção, a escolha de quais linhas recebem esses 9%. Um nulo é o resultado esperado e é informativo se
for limitado (ver Leitura).

## Pergunta
Com A congelado, mesmo orçamento, mesma força e a mesma distribuição de fatores de máscara por módulo, escolher as
linhas de B pela importância (exact) dá mais AP do que protegê-las em linhas aleatórias fixas (random_fixed)?

## O braço novo: random_fixed
experiments/superni_cl.py e src/dual_heater/lora_slowheat.py no commit deste arquivo (branch
exp/3c-random-fixed, implementação 929660e). random_fixed é o exact com uma única diferença: depois de cada
consolidação, os valores de heat de cada módulo são ordenados e entregues às linhas de UMA ordem aleatória, sorteada
uma vez por run com semente (seed + 500000) * 1009 + índice do módulo (mesmo offset do F16). O i-ésimo maior heat cai
sempre na linha order[i]: as linhas protegidas são order[:k] em toda fronteira (aninhadas se k cresce). Os valores
só mudam de linha, então o multiconjunto de fatores por módulo é o que a própria importância do braço daria.
O tracker continua medindo importância na trajetória do próprio braço (G-imp vale para ele).

Verificação antes do congelamento (CPU, Lab):
- 11 mutações do braço, 11 mortas pelos testes (tests/test_lora_slowheat.py, tests/test_superni_cl.py).
- 187 testes passam (lora_slowheat, superni_cl, frozen_a_control, lora, lora_confirmation_guard,
  plasticity_matched_runner, qwen_iso_plasticity) no venv do runner.
- Equivalência de código: vanilla, exact, frozen_a_control e lr_control rodados no experimento CPU minúsculo dos
  testes em deea8d2 e em 929660e dão pesos LoRA, slow_heat e cobertura de máscara bit a bit idênticos. É isso que
  permite parear o braço novo com os braços do 3b.
- Smoke CPU com o Qwen2.5-0.5B real (3 tarefas, 6 passos): resultado no fim deste arquivo.

## O que roda
Só o braço random_fixed, seeds 1000 a 1009, um processo por seed, com exatamente os argumentos do processo main do
3b: --order 1 --epochs-per-task 1 --batch-size 2 --eval-batch-size 8 --precision fp32
--acquisition-margin 0.05 --min-vanilla-forgetting 0.05 --arms random_fixed (lr 1e-4, LoRA r 8 alpha 16 na FFN,
budget 0,5, força 3, padrões do runner). Fila: scripts/queue_3c.py, um worker por GPU, CUDA_DEVICE_ORDER=PCI_BUS_ID,
mesmo venv e ambiente do 3b. Saída results/3c_random_fixed/s<seed>/. Reserva na planilha MOTOZ antes de lançar.

Desvio declarado da regra 2 da pré-registração (seeds de banda nova): as seeds são as do 3b, de propósito. vanilla,
exact, frozen_a_control e os dois lr_control NÃO rodam de novo; o braço novo é pareado por seed com eles. Os
resultados desses braços já são conhecidos; o do random_fixed não. Pareamento entre processos já era o caso do 3b
(lr_control em processos separados).

## Endpoints (AP e Fgt como no SAPT, ROUGE-L, escala fração)
Primário (teste único, sem família): seleção = AP(exact) - AP(random_fixed), por seed. t pareado bilateral nas 10
seeds, IC 95% t, alfa 0,05. Reportar n de seeds com sinal.

Secundários (sem correção, todos reportados, nenhum promovido depois):
  S1. Forma: AP(random_fixed) - AP(lrc_exact) (E pareado se V3 passar).
  S2. AP(random_fixed) - AP(frozen_a_control): custo de uma máscara aleatória de ~9% em B sobre o LoRA-FA
      (E não pareado, descritivo).
  S3. exact - frozen_a_control, repetido do 3b para ficar na mesma tabela.
  S4. Os contrastes do primário, S1 e S2 em Fgt (menos é melhor).
  S5. Sobreposição média entre as linhas protegidas pelo random_fixed e as que a importância dele mesmo escolheria
      (integridade; acaso = fração protegida = 0,5 com budget 0,5).

## Validade (checada antes de ler qualquer endpoint)
  V1. G-imp do random_fixed passa em toda seed; mais de 2 seeds falhando invalida o run.
  V2. Integridade da realocação em toda fronteira de toda seed: auditoria presente, linhas protegidas > 0, soma do
      heat igual antes e depois (tolerância relativa 1e-6), sobreposição média com a própria seleção < 1. Qualquer
      falha invalida.
  V3. Pareamento de E: |E(random_fixed) - E(exact)| <= 0,01 por seed, com E de superfície calculado sobre os
      3.317.760 parâmetros treináveis do vanilla do 3b. Mais de 2 seeds fora invalida o primário como contraste
      pareado.
  V4. commit do manifest de cada seed igual ao commit deste protocolo.
  V5. Processo com código != 0 é relançado uma vez com a mesma seed; registrado. Falhou de novo: seed faltando,
      reportada, e o primário roda nas seeds completas.

## Análise
scripts/analyze_3c.py, escrito ANTES do run e commitado com este arquivo. Validado em modo --dry-lorafa (usa o
frozen_a_control do 3b no lugar do random_fixed): reproduz exatamente o S3 do 3b (+0,0028, IC95 [-0,0026; +0,0082],
5/10, p 0,271) e marca o run como INVÁLIDO por V3 (|dE| 0,054 a 0,056 em 10/10 seeds), o que mostra que as checagens
de validade disparam.

## Leitura (declarada antes)
- Primário > 0 com p < 0,05: a seleção do exact paga sobre linhas aleatórias fixas neste host. Reportar o tamanho
  junto da ressalva de que a máscara tira só ~9% de B.
- Primário < 0 com p < 0,05: linhas aleatórias fixas batem a seleção. Reportar como resultado.
- IC95 do primário inteiro dentro de +-0,01 AP (1 pp): "seleção nula e limitada a 1 pp" neste host. É a frase que
  o manuscrito usa no lugar de "zero sob LoRA" para o stream generativo.
- IC cruzando zero e saindo de +-0,01: nulo não limitado; o manuscrito diz que o run não tem poder para limitar.
- Poder, estimado com o sd do S3 do 3b (0,0075, não medido neste braço): 80% para |efeito| = 0,0075 AP,
  85% para 0,008, 96% para 0,01; 47% para 0,005.

## Custo esperado
Da execução do 3b: o braço exact levou 0,73 a 0,76 h por seed (treino + geração), mais ~190 s de zero-shot por
processo. ~0,8 h por seed, ~8 h-GPU, ~3,2 h nas 3 GPUs (4 seeds na mais carregada). Estimativa, não medida para
este braço.

## Limitações
Uma ordem de tarefas, um modelo (0.5B), 1 época, a mesma banda de seeds do 3b. A máscara do exact é fraca (média
0,91 em B), então este run mede seleção num regime em que ela tem pouco a fazer; um resultado aqui não generaliza
para máscaras mais fortes. O orçamento e a força não foram ajustados e não serão.

## Smoke CPU com o modelo real (antes do congelamento)
tmp/3c_cpu_smoke (não versionado), commit 929660e, Qwen2.5-0.5B real, 3 tarefas, 16 exemplos de treino e 4 de
avaliação por tarefa, 6 passos, seed 1000, CPU, braços vanilla exact random_fixed. Serve para fiação, não para efeito.
- random_fixed: 2.039.808 parâmetros mascarados (igual ao exact), G-imp True, E 0,5576 contra 0,5577 do exact.
- Auditoria nas 3 fronteiras: 127.488 linhas protegidas, sobreposição com a própria seleção 0,501 / 0,500 / 0,500
  (acaso 0,5), soma do heat igual a 2e-8 relativo (ordem de soma em fp32).
- ~123 s por braço em CPU.
