# Fase 3b: estudo de desenho (instruction tuning como palco do instrumento)

Status: ESTUDO, não é protocolo. Nada aqui é pré-registro, nada foi rodado. Escrito em 09/11/2026 para
decidir com o Fachel como montar a fase. Fonte de verdade da fase continua sendo
goals/roadmap_icml_ijcnn.md (seção Fase 3b).

## 1. O que a fase precisa provar

Claim do roadmap (D6 = A): em instruction tuning, ganhos de métodos de CL não sobrevivem ao pareamento
por plasticidade efetiva, inclusive os nossos. Endpoint de fase (3b.4): a ordenação dos braços muda
quando E é pareado? Não é "o SlowHeat ganha?".

O que já existe em classificação (Qwen, CLINC150): bloco 12 (slice cai de 2º para 4º com E pareado,
E_slice = 0,0625) e bloco 14 (frozen_a_control - lr_control, plasticidade de superfície idêntica).
A 3b repete isso num host generativo.

## 2. Estado do código (medido no repo em 09/11)

Pronto:
- src/dual_heater/superni_data.py: prompt no formato SuperNI, truncamento, agrupamento por tarefa.
- src/dual_heater/superni_metrics.py: ROUGE-L e EM próprios, 7 mutações mortas. Não validado contra a
  implementação oficial (o docstring declara isso).
- src/dual_heater/lora_slowheat.py: build_lora_slowheat aceita CAUSAL_LM (aed2957).
- experiments/qwen_lora_slowheat.py: braços vanilla, exact, rank, leak, slice, lr_control,
  frozen_a_control, mas só como classificador (SEQ_CLS).

Falta:
- Runner generativo de CL: loss só na resposta, geração gulosa, matriz T x T de ROUGE-L, E medido por
  braço, manifest no formato dos outros runners.
- Dados: nem SuperNI nem TRACE estão no Lab (o cache HF só tem CLINC150, BERT-mini e Qwen2.5-0.5B).
  Baixar precisa do ok do Fachel.

## 3. Benchmark: proposta de inverter TRACE/SuperNI

O roadmap pôs TRACE no corpo e SuperNI como fallback. Os números do TRACE pesam contra isso no nosso
hardware:

| | TRACE | SuperNI (protocolo SAPT) |
|---|---|---|
| tarefas | 8 | 15 (5 tipos x 3), 2 ordens publicadas |
| treino por tarefa | 5.000 | 1.000 |
| teste por tarefa | ver repo | 100 |
| métrica | acc / ROUGE-L / SARI / edit sim., misturadas | ROUGE-L em tudo |
| entrada mais longa | MeetingBank, 2.853 palavras em média | curtas a médias (diálogo, QA, resumo) |
| épocas no paper | full FT 1-5, LoRA 3-7 por tarefa | ver repo SAPT |
| hardware do paper | 8x A100 80G, DeepSpeed | T5-large a LLaMA-2-13B |
| baselines publicadas | SeqFT, LoRA, replay, EWC, O-LoRA etc. | SeqLoRA, replay, L2P, LFPT5, O-LoRA, SAPT |

Fontes: TRACE arXiv 2310.06762 (tabela de datasets e seção de treino); SAPT ACL 2024 (seção 5.1,
apêndice B). O MeetingBank sozinho passa de 4k tokens por exemplo, o que não cabe com folga num 0.5B em
11 GB de Pascal sem truncar a tarefa até ela virar outra coisa.

Proposta: SuperNI-15 do SAPT no corpo (ordens 1 e 2), e TRACE só com as tarefas curtas (FOMC,
ScienceQA, NumGLUE-cm/ds, C-STANCE) como extensão se sobrar tempo. Ressalva honesta: o Qwen2.5 pode ter
visto tarefas do SuperNI no pré-treino. Isso afeta todos os braços igual e não derruba um contraste
pareado, mas precisa estar nas limitações.

## 4. Endpoint e gate (o buraco da Fase 3.5)

Lição da 3.5: o endpoint estava calculado certo e não tinha dispersão. Nenhum gate vigiava isso. Na 3b o
gate de quase-chance não existe (geração não tem chance uniforme). Proposta:

Endpoint primário (por método X com E_X < 1):
  D_unmatched(X) = AP(X) - AP(vanilla)
  D_matched(X)   = AP(X) - AP(lr_control em E_X)
  Primário: D_matched(X) < D_unmatched(X), pareado por seed, Holm sobre os métodos.
  Secundário: o mesmo com F.Ra (esquecimento) e a ordenação completa nas duas condições.
AP e F.Ra como no SAPT (ROUGE-L), para dar comparação externa.

Gates, checados numa seed de calibração FORA da banda, antes de gastar seeds:
  G-aq (aquisição): o vanilla tem que superar o zero-shot do modelo base em cada tarefa logo depois de
    treiná-la, por margem declarada. Substitui o gate de chance: o piso é o zero-shot, não 1/K.
  G-disp (dispersão): o esquecimento do vanilla tem que ser >= um mínimo declarado e o primário tem que
    variar entre braços. Se não variar, a fase para e o endpoint é re-registrado antes de qualquer seed.
  G-imp (importância viva): sob fp16 o estimador zerou 19 de 24 camadas no fine-tuning completo. No
    LoRA causal isso precisa ser medido de novo. Mesmo assert_importance_is_live do runner do Qwen.

## 5. Braços

Mínimo defensável:
- vanilla (SeqLoRA)
- exact (SlowHeat no LoRA)
- frozen_a_control (LoRA-FA, arXiv 2308.03303). Obrigatório pelo R-C/R7.
- lr_control em E_exact e em E_FA (um por método mascarado)
Opcionais, por custo:
- slice (repete o caso do bloco 12)
- replay ou O-LoRA (baseline publicada no SAPT e no TRACE). O-LoRA está no repo do TRACE.
- EWC no LoRA: só se D1 = sim, e a âncora (D4) precisa ser decidida antes.

## 6. Ordem de execução proposta

1. Ok do Fachel para baixar CL_Benchmark/SuperNI do repo circle-hit/SAPT (pequeno: 15 x 1.200
   exemplos) para /mnt/B-SSD/fachel.
2. Runner generativo com TDD e mutação (padrão do repo). Teste de wiring: E medido = E declarado por
   braço; loss mascarada no prompt; matriz T x T sem vazamento de tarefas futuras.
3. Validar superni_metrics contra o ROUGE-L oficial num lote fixo (pendência declarada no docstring).
4. Calibração 3b.2 numa seed fora da banda, uma GPU: memória de pico, tempo por tarefa, gates G-aq,
   G-disp e G-imp. Daqui sai o custo real e o número de seeds.
5. Protocolo congelado (padrão F11-F15), banda de seeds nova, análise validada no dado da calibração.
6. Confirmatório.

Custo: desconhecido até o passo 4. Não chuto número.

## 7. Decisões que dependem do Fachel

- Inverter TRACE/SuperNI (seção 3).
- Ok para o download do item 6.1.
- Quais braços opcionais entram (seção 5). Cada braço a mais custa uma run por seed.
- D1/D4 (EWC no host LoRA) continuam abertas e mudam a seção 5.
- Prazo: a 3b é a candidata natural a corte se a escrita apertar (R6 do roadmap). O plano B sem GPU
  (blocos 12 e 14 em classificação) continua valendo.
