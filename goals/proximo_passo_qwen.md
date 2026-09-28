# Próximo passo — Qwen2.5-0.5B (CLINC150: iso-plasticidade e LoRA)

> Um de quatro documentos por arquitetura. Índice e decisões transversais em
> [proximo_passo_artigo.md](proximo_passo_artigo.md).
>
> Escopo: `SlowHeatQwen2ForSequenceClassification` sobre `Qwen/Qwen2.5-0.5B`,
> FFN gated (SwiGLU). **Duas linhas experimentais distintas** que este documento
> mantém separadas: full fine-tuning sob iso-plasticidade (parte A) e adaptação
> LoRA (parte B). Fontes:
> [docs/architectures/arch_qwen.md](../docs/architectures/arch_qwen.md) e
> [docs/lora/](../docs/lora/).

---

## 0. Por que este host é o centro do artigo

É o único host onde o **protocolo** foi construído e exercitado: teorema do piso
de plasticidade (A1), construção iso-plasticidade por bisseção (A2), contrato de
otimizador (A3), braço `permuted` e braço `lr_control`.

E é o único onde o protocolo **rejeitou coisas**: três mecanismos novos e uma
hipótese central morreram sob ele.

Também é o host com a restrição mais dura: aqui **não existem** DER++, ER-ACE,
A-GEM, EWC, SI nem LwF. Esses baselines só estão implementados em MLP/CNN.
Qualquer plano que exija "vencer baselines estabelecidos" neste host está na
verdade exigindo implementá-los do zero para um LLM, o que é outro projeto.

---

# Parte A — Full fine-tuning sob iso-plasticidade

## A.1 O que está medido

Run de confirmação concluída em 23/09, 10 seeds (`confirm120_seed10..19`),
10 tarefas, 120 passos, `protocol_hash` idêntico nas 10.

**A pergunta:** com a mesma quantidade de plasticidade efetiva removida (`E`
fixo), importa *quais* unidades são protegidas? O braço `permuted` isola
exatamente isso.

| braço (E\*=0,75) | FAA | retenção t0 | forgetting |
|---|---|---|---|
| vanilla | 0,407 | 0,020 | 0,523 |
| permutado b=0,25 | 0,406 | 0,034 | 0,521 |
| reduced_lr | 0,424 | 0,032 | 0,494 |
| iso b=0,25 | 0,433 | 0,046 | 0,492 |
| **hard b=0,75** | **0,458** | **0,095** | **0,461** |

Sob Holm (24 comparações), **2 de 24 sobrevivem**, ambas `hard − vanilla` em
retenção t0, 10/10 seeds, `p_Holm = 0,047`.

## A.2 Os três achados que importam

**A hipótese central ficou sem suporte.** `iso − permutado` dá o sinal certo em
todos os endpoints mas nenhuma comparação sobrevive a Holm. Com 2 tarefas era
+0,080 em 3/3 seeds; com 10 tarefas caiu para +0,012 a +0,018. **A conclusão de
2 tarefas era otimista, e isso é reportável.**

**"Quanto" passou a importar mais que "quais".** `hard − iso` (+0,050 e +0,065
de retenção, 9/10) é maior que `iso − permutado`. Como ambos protegem as
**mesmas** unidades e diferem só na dureza, o ganho vem da forma da proteção,
não da escolha do alvo.

**O P6 explica.** A sobreposição de unidades dominantes entre tarefas é 0,679
(top-10) contra 0,002 de acaso. Se as tarefas disputam as mesmas unidades,
embaralhar com `E` fixo acaba protegendo um conjunto parecido — e a vantagem da
seleção aprendida encolhe conforme a sequência cresce.

## A.3 O limite de poder, que é estrutural

Com 10 seeds o menor `p` possível no teste exato é `2/1024 = 0,00195`. Sobre 24
comparações, Holm multiplica por 24 e dá 0,047. **Um efeito perfeito em 10/10
seeds mal passa de 0,05**, e uma família de 26 comparações não conseguiria
produzir nenhum resultado significativo.

Consequência prática: para afirmar algo sobre `iso − permutado` é preciso **mais
seeds, não mais braços**.

## A.4 Caminho crítico da parte A

### QA-1 — Mais seeds para `iso − permutado`, com família reduzida (GPU, médio)

É a hipótese central e está exatamente no limite do poder. Restringir a família
de 24 para 2–3 testes muda o limiar de Holm de 0,047 para ~0,006, mas **só é
legítimo se a redução for pré-registrada antes das seeds novas**, exatamente
como foi feito em `protocol_lora_confirmation.md` A6.

**Não reanalisar as 10 seeds existentes com a família reduzida.** Escolher a
correção depois de ver o resultado é o que ela existe para impedir.

### QA-2 — Escrever `docs/qwen_layer_anomaly.md` (CPU, barato, já medido)

Os dados estão em disco (`results/qwen_layer_anomaly/`, 7 manifestos, 2 ordens ×
3 seeds) e **a análise nunca foi escrita**. O documento é entregável prometido de
P6 e base do Gate 2.

Hipótese a testar com o que já existe: a queda de PR no fim da rede (L19 47,5%,
L20 45,3%, L22 27,7%, L23 10,8%) é efeito de profundidade, e L3/L21 (PR de 2,2%
e 0,9%) são fenômeno distinto. Custa CPU e fecha um gate aberto.

### QA-3 — Registrar Gate 1 e Gate 3 (CPU, texto)

O protocolo congelado exige registro de Gate 1, 2 e 3; **nenhum foi registrado**,
e a confirmação foi iniciada sem Gate 3. Isso é dívida de protocolo que um
revisor atento encontra. Registrar retroativamente é honesto desde que a linha
diga que foi retroativo.

### QA-4 — Sequências intermediárias de 4 e 6 tarefas (GPU, médio)

Mapeia onde a vantagem da seleção aprendida se dissolve entre 2 e 10 tarefas. É
o experimento que transforma "a conclusão de 2 tarefas era otimista" de ressalva
em **curva medida**, que é muito mais forte.

Prioridade abaixo de QA-1, acima de QA-5.

### QA-5 — Investigar o braço `hard` (GPU, médio)

É o efeito mais forte da parte A e o menos explorado: só foi testado em
`budget = E*`, porque sob máscara hard `E = budget` exatamente. Um sweep de
budget sob hard nunca foi feito.

### QA-6 — Desvio de pré-registro já registrado (verificação)

O braço `hard` aparece nos manifestos **sem constar da seção E do protocolo nem
ter linha na tabela K antes da run**. Isso precisa de uma linha honesta na
tabela K dizendo que foi detectado depois, e o resultado do `hard` precisa ser
reportado como exploratório, não confirmatório — o que enfraquece justamente as
2 comparações que sobreviveram a Holm.

**Este é o problema mais sério da parte A** e não foi tratado em nenhum
documento anterior.

---

# Parte B — LoRA

## B.1 Estado da run confirmatória: CONCLUÍDA E CONFIRMADA (28/09)

10/10 seeds da banda 700001+, `missing_seeds` vazio, commit `475d3c6`, 5.480 s
de orquestrador. Verificações H1–H5 do pré-registro todas passam.

### Endpoint primário (A3): `exact − lr_control`, forgetting

| grandeza | valor |
|---|---|
| diferença média | **−0,0921** |
| mediana pareada | −0,0998 |
| sinais | **9−/1+** |
| `p` (sinal exato bicaudal) | **0,02148** |
| critério A8 (`p < 0,025` e mediana < 0) | **ATENDIDO** |

> **Confirmado.** Na mesma plasticidade efetiva removida, a proteção seletiva
> reduz o esquecimento mais do que a redução uniforme de learning rate.

| braço | FAA | forgetting | E_eff |
|---|---|---|---|
| **exact** | **0,6384±0,0314** | **+0,3278** | 0,850 |
| lr_control | 0,5790±0,0468 | +0,4200 | 1,000 |
| vanilla | 0,5706±0,0273 | +0,4296 | 1,000 |

Sob Holm na família de 2 declarada em A6: forgetting sobrevive
(`p_Holm = 0,0430`), FAA **não** (`p_Holm = 0,1094`, 8/10). O ganho de FAA
contra `lr_control` **não deve ser reportado como positivo**.

Contra vanilla, ambos secundários: forgetting −0,1017 (10/10, p=0,00195) e FAA
+0,0678 (9/10, p=0,02148).

**O efeito encolheu do exploratório para o confirmatório** (−0,119 para −0,092,
e de 10/10 para 9/10). É regressão à média entre exploração e confirmação, que é
exatamente a razão de a confirmação existir. Detalhe completo em
[docs/lora/lora_confirmation_results.md](../docs/lora/lora_confirmation_results.md).

### Ressalva de registro aberta (R1)

A seção E do pré-registro descreve `alpha=32` e `max_length=64`; as 10 seeds
rodaram com **`alpha=16` e `max_length=48`** — o mesmo config da run
exploratória que motivou a confirmação, verificado campo a campo. Nenhum braço
foi favorecido e a comparação entre runs permanece válida. É erro de
transcrição no protocolo, já registrado na tabela K em 28/09.

Pré-registro congelado em
[protocol_lora_confirmation.md](protocol_lora_confirmation.md): três braços
(`vanilla`, `exact`, `lr_control`), 10 seeds na banda 700001+, endpoint primário
forgetting no contraste `exact − lr_control`, Holm sobre família de 2, critério
`p < 0,025`.

## B.2 O que as runs exploratórias produziram

**Três mecanismos novos, três falhas.** `rank`, `leak` e `slice` ficam todos
**abaixo do vanilla** sob plasticidade pareada em `E = 0,85` e não superam
`lr_control` significativamente (`p >= 0,11`). O `slice` tinha a defesa de que
`E_eff = 0,062` o penalizava na primeira run; com pareamento a defesa caiu.

**O `lr_control` é pior que o vanilla** (FAA 0,5684 contra 0,6016; forgetting
+0,4350 contra +0,3967). Reduzir LR uniformemente em 15% **piora** o
esquecimento. Isso torna o falsificador um teste não trivial, e mostra que
comparar só contra vanilla confunde "frear ajuda" com "frear seletivamente
ajuda".

**O efeito do `exact` cresce sob controle mais rigoroso:**

| contraste | sem pareamento | `E = 0,85` |
|---|---|---|
| forgetting vs vanilla | −0,0629 (p=0,002) | −0,0807 (p=0,002) |
| FAA vs vanilla | +0,0349 (p=0,11) | +0,0478 (p=0,02) |
| forgetting vs lr_control | — | **−0,1190 (p=0,002, 10/10)** |

Controles mais rigorosos normalmente encolhem efeitos. Este aumentou, o que
sugere que a primeira run **subestimava** o mecanismo.

## B.3 O buraco no único resultado positivo

`exact` treina **4.214.016** parâmetros contra **6.769.920** dos outros (`r=16`),
porque congela `A` — 62% da capacidade do adaptador. A leitura *ele esquece
menos porque aprendeu menos* **não está descartada**, e o pareamento em `E` não
resolve: `E` mede plasticidade retida da máscara, não capacidade do adaptador.

## B.4 Caminho crítico da parte B

### QB-1 — Confirmação fechada (CONCLUÍDA em 28/09)

Resultado em B.1: **confirmado**, `p = 0,02148 < 0,025`, mediana negativa, 9/10
seeds. Registrado em
[docs/lora/lora_confirmation_results.md](../docs/lora/lora_confirmation_results.md)
e na tabela K do pré-registro.

O que este resultado **habilita**: o artigo agora tem um resultado
confirmatório em Transformer, pré-registrado antes da primeira seed, com
critério declarado e controle não trivial. Era exatamente o que faltava.

O que ele **não** habilita, e a tentação a resistir: rodar mais seeds para
levar o FAA (`p_Holm = 0,1094`) até a significância. A família de 2 foi
declarada em A6 e o FAA não passou. Fim. Reabrir agora invalidaria
retroativamente o afrouxamento de limiar que o pré-registro comprou.

### QB-2 — Pré-registrar e rodar `exact r=32` vs `vanilla r=16` (~2h GPU)

**O experimento de maior valor por hora de GPU no projeto.** Iguala parâmetros
treináveis e decide se o único resultado positivo é mecanismo ou artefato de
capacidade.

Dois pontos que o pré-registro precisa declarar **antes**:

1. Igualar parâmetros treináveis **não** iguala a expressividade do subespaço.
   `exact` com `r=32` e `A` congelada tem 32 direções aleatórias fixas;
   `vanilla` com `r=16` tem 16 direções aprendíveis. Declarar como limitação, não
   descobrir depois.
2. O critério de ambos os desfechos. Se o efeito sobreviver, a explicação por
   capacidade cai. Se desaparecer, o único resultado positivo era artefato e
   **isso precisa ser reportado**, não engavetado.

Ordem: depois de QB-1 e do levantamento de literatura (ver hub).

#### ⚠ Achado do levantamento (28/09) que pode encolher QB-2

**LoRA-FA** ([arXiv 2308.03303](https://arxiv.org/abs/2308.03303), 2023) **já
congela `A` e treina só `B`** — é exatamente o mecanismo do braço `exact`,
proposto com motivação diferente (reduzir memória de ativação, não esquecimento).
LoRI (2025) faz o mesmo com múltiplos `B`.

Duas consequências:

1. **Redação.** O artigo não pode apresentar "congelar `A`" como mecanismo
   próprio. Precisa citar LoRA-FA e LoRI e posicionar `exact` como *avaliação de
   um mecanismo conhecido sob protocolo controlado*. Isso **não** derruba o
   resultado de 28/09 — a contribuição é o contraste `exact − lr_control` sob
   plasticidade pareada, que LoRA-FA não faz.
2. **Valor de QB-2.** LoRA-FA afirma revelar uma "estrutura assimétrica e
   colapsável" na atualização do LoRA, reformulável como regressão linear de
   camada única, implicando que **um dos fatores pode ser congelado sem sacrificar
   expressividade**. Se essa afirmação se sustentar, a explicação alternativa que
   QB-2 existe para eliminar (*"`exact` esquece menos porque tem menos
   capacidade"*) já tem resposta teórica publicada, e QB-2 cai de "necessário"
   para "confirmação empírica de algo já argumentado".

**Ação obrigatória antes de gastar as 2h de GPU:** ler LoRA-FA na íntegra e
decidir se QB-2 ainda se justifica. O resultado dessa leitura vai para a tabela K
de `protocol_lora_rank_matched.md` **antes** da run, qualquer que seja a decisão.

### QB-3 — Levantamento da família LoRA-CL — **CONCLUÍDO em 28/09**

Feito junto com o levantamento transversal; registro em
[../docs/related_work/protocol_prior_art.md](../docs/related_work/protocol_prior_art.md),
seção "Prioridade 2".

Resumo: **O-LoRA** ([2310.14152](https://arxiv.org/abs/2310.14152), EMNLP Findings
2023) restringe atualizações a subespaços ortogonais entre tarefas e colide com o
mecanismo `rank`. **InfLoRA** ([2404.00228](https://arxiv.org/abs/2404.00228),
CVPR 2024) reparametriza para eliminar interferência e colide com `rank` e
`slice`. Ambos **precisam ser citados** onde os três mecanismos negativos forem
reportados — sem isso, "o mecanismo não funciona" fica indefensável, porque
existem versões publicadas que funcionam. CURLoRA e GS-LoRA são vizinhos que não
colidem diretamente (o segundo resolve unlearning, problema diferente).

### QB-4 — Não medir `E` baixo sob pareamento (decisão de escopo)

`E = 0,85` foi escolhido por ser alcançável por todos os braços. O comportamento
em `E` baixo, onde o `slice` vivia na run 1, não foi medido sob pareamento.

Recomendo **não** medir: exigiria reabrir os três mecanismos falhados, e a
§5 do hub proíbe rodar mecanismo novo antes do levantamento. Declarar como
limitação de escopo.

---

## 5. O que NÃO fazer neste host

- **Não reabrir o pré-registro do LoRA em nenhum desfecho.** Nem para mais
  seeds, nem para família reduzida, nem para promover um endpoint secundário.
- **Não reanalisar as 10 seeds de iso-plasticidade com família reduzida.**
  Mesma regra, mesma razão.
- **Não rodar um quarto mecanismo novo** antes do levantamento de literatura.
  Três já falharam sob controle honesto.
- **Não comparar contra DER++/ER-ACE/EWC aqui.** Não existem neste host, e
  implementá-los para um LLM em 3 GPUs Pascal é um projeto separado.
- **Não citar as runs de calibração de 30 passos.** A tabela K revogou esse
  valor em favor de 120; com 30 passos a acurácia fica em 0,31 contra teto
  prático de ~0,94.
- **Não apagar `rank`, `leak` e `slice`.** Permanecem no código, nos testes e
  nos docs: são a evidência de que o protocolo tem poder de rejeição.

---

## 6. Resumo

| # | Ação | Custo | Bloqueia |
|---|---|---|---|
| ~~QB-1~~ | ~~Fechar a confirmação de LoRA~~ **CONCLUÍDA 28/09: confirmado, p=0,02148** | — | — |
| **QB-2** | **Pré-registrar e rodar `exact r=32` vs `vanilla r=16`** | **~2h GPU** | **o resultado confirmado agora depende disto** |
| QA-6 | Registrar o desvio do braço `hard` na tabela K | texto | integridade da parte A |
| QA-2 | Escrever `qwen_layer_anomaly.md` a partir dos dados em disco | CPU, horas | Gate 2 |
| QA-3 | Registrar Gate 1 e Gate 3 retroativamente | texto | dívida de protocolo |
| QB-3 | Levantamento LoRA-CL | CPU, meio dia | related work |
| QA-1 | Mais seeds para `iso − permutado`, família pré-registrada | GPU, médio | a hipótese central |
| QA-4 | Sequências de 4 e 6 tarefas | GPU, médio | a curva de dissolução |
| QA-5 | Sweep de budget sob `hard` | GPU, médio | o efeito mais forte, inexplorado |

**QB-2 subiu para o topo.** Antes da confirmação ele era um experimento de
saneamento; agora o artigo tem um resultado confirmado cuja única explicação
alternativa viva é capacidade do adaptador. As 3 GPUs estão livres.

## Referências

- [Evidência e limites do host](../docs/architectures/arch_qwen.md)
- [Resultados da confirmação iso-plasticidade](resultados_confirmacao.md)
- [Protocolo iso-plasticidade congelado](protocol_iso_plasticity.md)
- [Pré-registro da confirmação de LoRA](protocol_lora_confirmation.md)
- [Benchmark de LoRA](../docs/lora/lora_qwen_benchmark_results.md)
- [Os três mecanismos](../docs/lora/lora_slowheat_mechanisms.md)
- [Ablação iso-plasticidade e Anexo A](../docs/protocols/qwen_iso_plasticity_ablation.md)
- [Roadmap e gates](qwen_heat_roadmap.md)
- [Índice dos próximos passos](proximo_passo_artigo.md)
