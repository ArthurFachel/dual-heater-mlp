# TODO — ICML + IJCNN

**Atualizado:** 2026-10-05
**Substitui:** o roadmap narrativo (versão anterior deste arquivo, recuperável
com `git show 8b10bd8:goals/roadmap_icml_ijcnn.md`).
Este documento é **editável**; os protocolos em `goals/protocol_*.md` **não**.

| venue | deadline | páginas | estado |
|---|---|---|---|
| **ICML 2027** | ~22/01/2027 (estimado) | 8 + refs | ⚠️ **CFP não publicado** — `goals/venue_deadlines.md`, rechecar semanalmente |
| **IJCNN 2027** | **31/01/2027** confirmado | 6, IEEE, CMT | fonte oficial verificada 29/09 |

**~16 semanas.** O gargalo não é experimento, é escrita.

**Regra de não-sobreposição (inegociável):** R-A/R-B/R-D são corpo do IJCNN e
entram no ICML só como linha de tabela citando o companion. R-C/R-E/plasticidade
são corpo do ICML e **não** aparecem no IJCNN. Citar uma na outra como
"companion paper, under review".

---

## 🔴 Decisões travadas (precisam do Fachel)

- [ ] **D1. Host BERT entra no ICML?** EWC/SI/MAS não existem lá (verificado:
      0 ocorrências em `split_clinc150.py`). Reviewer cobra baseline no host
      transformer. Custo ~1 semana. **Fora do caminho crítico** — a Seção 4
      roda em MLP/CNN.
- [ ] **D2. Se a maioria dos métodos não sobreviver ao pareamento:** publicar o
      varrido completo (acusação ampla) ou 2–3 métodos com análise profunda
      (mais defensável)? Muda o tom do artigo inteiro.
- [ ] **D3. Plano B se o ICML for rejeitado** — ICLR 2028? NeurIPS 2027?
      Decidir agora, não em abril.
- [ ] **D4. Âncora do EWC sob LoRA** (só se D1 = sim): adaptador ou peso base?
      É decisão de método, exige pré-registro, não deve ser inferida.
- [ ] **D5. A passada 2 rodou e deu positivo num regime degenerado — replicar
      num regime são?** `mas − lr_control` foi significativo 12/12, mas os três
      arms ficaram perto do nível de chance (melhor arm 0,198 contra chance
      0,10). Separar "consolidação" de "mudou menos" exige um regime onde o
      `vanilla` de fato aprenda: mais épocas, lr maior, ou replay. **Isso é um
      experimento novo, com pré-registro novo.** A alternativa é reportar o
      resultado com a ressalva, que é o que o relatório já faz.
- [x] **D6. RESOLVIDO 05/10 — opção A.** O ICML passa a ser ancorado em
      **instruction tuning**, mantendo a tese do instrumento. **Rejeitada** a
      versão "método que diminui forgetting e aumenta acurácia": R-C dá 5+/5−
      contra LoRA-FA sob pareamento, nenhum contraste de FAA sobrevive a Holm em
      Qwen, e a conjunção se parte com significância nos dois sentidos em
      CIFAR-100. Benchmark: **TRACE**, SuperNI como fallback, **sem juiz LLM**.
      Fica aberto o que sai do plano — ver R6.

---

## ✅ Fechado

- [x] **R-A** proteção por ativação bate máscara aleatória em BERT: +12,23 pp,
      10/10 seeds, p_Holm = 0,0039 · `docs/results/criterion_ablation_results.md`
- [x] **R-B** o gradiente não compra nada: `|z·dL/dz|` empata com `|z|`
      (−0,17 pp, 5+/5, p = 1,00), com falsificador de variância · idem
- [x] **R-C** todo o ganho do `exact` vem de congelar `A` (= LoRA-FA);
      a máscara não adiciona nada (5+/5, p = 1,00) · `exact_decomposition_results.md`
- [x] **R-D** seletor de replay: perda alta **piora** (até −4,05 pp),
      representatividade melhora; unânime 20/20 em 4 learners · `replay_selector_results.md`
- [x] **R-E** `effective_plasticity()` media só sobre parâmetros com binding;
      o endpoint de 28/09 comparou E=0,532 contra E=0,850 · `lora_confirmation_results.md` §R3
- [x] **R-F** `lr_control < vanilla` **não replicou**; 7 ocorrências corrigidas em `45de59e`
- [x] **R-G** falsificador da §F da ablação de critério, **recuperado**:
      sobreposição top-k `magnitude` vs `aleatório` = 0,5748 < 0,80 em 10/10
      seeds. O braço `magnitude` foi genuinamente exercido; R-B é achado, não
      artefato · `docs/results/criterion_degeneracy_overlap.md`
- [x] Pareamento por superfície, 10 seeds, p_Holm = 0,0039 · `plasticity_matched_results.md`
- [x] EWC/SI/MAS extraídos para `src/dual_heater/ewc.py`, equivalência numérica pinada
- [x] **Instrumento de plasticidade** (`src/dual_heater/plasticity.py`):
      pré-registrado antes do código, lema pinado, 6 mutações mortas, ligado ao
      runner com guarda de wiring, validado CUDA ≡ CPU
- [x] **Passada 1 Split-MNIST**, 12 seeds, medição exaustiva ·
      `docs/results/penalty_pass1_plasticity.md`
- [x] **Passada 1 Split-CIFAR100**, 12 seeds, 3,44 h de GPU; G8 dispara nos três
      métodos · `docs/results/penalty_pass1_cifar_plasticity.md`
- [x] **L2 — `E > 1` é artefato de otimizador adaptativo.** Sob SGD puro os três
      métodos dão `E < 1`, 12/12 seeds, em três ordens de grandeza de lr.
      C1 falsificada, C2/C3/C5 confirmadas (13.533/13.533 passos), C4 falsificada ·
      `docs/results/sgd_plasticity_results.md`

---

## 🔵 Fase 2 — instrumento de plasticidade `[ICML §4]`

- [x] 2.1 Pré-registrar a métrica antes de implementar · `protocol_plasticity_generalization.md`
- [x] 2.2 TDD `plasticity_ratio` + diagnósticas
- [x] 2.3 Mutação (M1–M5 + M6 do `.grad`, todas mortas)
- [x] 2.4 Passo-sombra ligado ao runner, sem perturbar o treino
- [x] 2.5 Pré-registrar o piloto · `protocol_penalty_reevaluation.md`
- [x] 2.6 Forças publicadas adotadas (Hsu et al. 2018, class-IL) · `experiments/penalty_reevaluation.py`
- [x] 2.7 **Passada 1 MNIST** — SI e MAS **amplificam** o update, 12/12 seeds, p = 0,00049
- [x] 2.8 **Passada 1 CIFAR-100** — 12 seeds, 3 GPUs, 3,44 h · `5eed624`
- [x] 2.9 Agregação das 12 seeds, teste de sinal exato (`p = 0,00049` em todos os
      arms). O **cosseno do MAS < 0,9 se confirmou** (mínimo −0,88): o update
      inverte de direção, nenhum escalar de plasticidade o descreve
- [x] 2.10 `docs/results/penalty_pass1_cifar_plasticity.md`
- [x] 2.11 **Decidido: L2 primeiro** (§L do protocolo), e ele rodou. Resposta:
      `E ≈ 0,996` era artefato do AdamW. Sob SGD, `E` = 0,955 / 0,976 / 0,993
      (lr=3e-3) — os três passam no G8 e **a passada 2 volta a ter objeto**.
- [x] 2.12 **L3 — sweep de λ sob SGD**, 10 seeds × 5 pontos, 84,2 min de CPU.
      T-P1 a T-P4 **todas confirmadas**. De 15 células, **duas** têm poder, e as
      duas são do `mas`: 30× (`E` = 0,854) e 100× (`E` = 0,728), unânime 10/10.
      `ewc` e `si` passam de "fraco demais" (E > 0,91 a 3×) para "divergente"
      (10/10 seeds a 10×) **sem janela utilizável entre os dois regimes** ·
      `docs/results/lambda_sweep_results.md`
- [x] 2.13 **Passada 2 sob SGD** — 12 seeds, ~8 min de CPU. **Q1 FALSIFICADA**:
      `mas − lr_control` = −0,0400, 12/12 seeds, `p = 0,00049`. Q2 confirmada (o
      controle é um controle). Q3 falsificada na direção oposta (o `mas` ficou
      **acima** do vanilla em acurácia). ⚠️ **Mas os três arms terminaram em
      quase-chance** (0,149/0,198/0,160 contra chance 0,10): o contraste não
      separa consolidação de simples redução de mudança ·
      `docs/results/penalty_pass2_results.md`

> ⚠️ **G8 disparou sob AdamW, e o L2 explicou por quê.** SI e MAS tinham `E > 1`
> no MNIST, e os três no CIFAR-100: `lr_scale > 1` é aumento de LR, não controle
> de plasticidade. A família confirmatória caiu de **6 para 2** comparações, e
> depois para 1 operacionalmente útil. **O L2 mostrou que isso é interação com o
> otimizador adaptativo, não propriedade do método** — sob SGD os três ficam
> abaixo de 1. A redução **não foi pós-hoc**: G8 estava escrita no §C antes das
> seeds. Piso de significância com n=12, m=2: `0,00098`.

### Emendas aplicadas (todas mecanismo-only, sem ler acurácia)

| # | mudança | motivo medido |
|---|---|---|
| 1 | G1 → G1': primária vira razão de **normas** | razão por elemento deu `E > 1` (MAS 1,78); denominadores pequenos viram outliers ilimitados |
| 2 | forças de penalidade → Hsu et al. 2018 | `si_lambda=1.0` era **300× abaixo** do publicado; SI era vanilla com outro nome |
| 3 | amostragem 1/50 → **1/1** | dp entre passos 0,077 contra efeito 0,012 — ruído 14× o sinal; o veredito G8 invertia |

Runs descartadas ficam em `results/_abandoned_*` com `WHY_ABANDONED.md`.

---

## 🟢 Fase 3 — segundo host para `|z|` `[IJCNN]`

Limite declarado de R-A/R-B: um host, um benchmark, **duas tarefas por sequência**.

**Pré-registro:** `goals/protocol_long_sequence_criterion.md`, congelado 05/10,
`T = 5`, banda 12.000.017+, família de 2 com Holm, gate de quase-chance no §G.1.

- [x] 3.1 `task_limit` deixou de ser 2 fixo. Eram **três** pontos, e o terceiro
      era um bug: `condition_endpoints` lia `matrix[1]` e
      `parameter_drift_history[1]` — o **estágio 1**, não o último. Com `T = 5`
      reportaria "retenção" medida após a tarefa 2 enquanto a run foi até a 5,
      sem nada no artefato denunciando. Corrigido com não-regressão bit-idêntica
      para `T = 2`, guarda de wiring do CLI e recusa de agregar braços com `T`
      diferente. **19 mutações mortas.** Smoke em CPU prova `T = 3 / 5 / 10`.
- [x] 3.2 Pré-registrado antes de rodar; banda de seeds nova, pinada em teste.
      A escolha `T = 5` contra `T = 10` **leu FAA de smoke** e isso está
      declarado na §F do protocolo, não escondido.
- [x] 3.3 Falsificador repetido — e **a §F estava cumprida pela metade**. A
      sobreposição top-k entre braços nunca foi calculada na run de 28/09:
      é grandeza *entre runs* e o hook do runner roda dentro de um só.
      Recuperada dos buffers `slow_heat`: `magnitude` vs `aleatório` = **0,5748**
      (< 0,80, **o falsificador passa**), `magnitude` vs `funcional` = 0,8814 ·
      `docs/results/criterion_degeneracy_overlap.md`. Agora é checagem
      obrigatória pré-endpoint (`scripts/verify_long_sequence_criterion.py`, V6),
      ao lado do gate de quase-chance (V7). **16 mutações mortas.**
- [x] 3.4 **Calibração em GPU executada** (seed `11_900_003`, fora da banda,
      167 s). Custo OK: 0,46 h projetadas contra teto de 4 h. **E ela matou o
      endpoint primário:** `task1_retention` = 0,0000 nos quatro braços em
      `T = 5`. A família do S7 daria `p = 1,00` por construção. Não é ruído (em
      `T = 2` a amplitude entre braços é 0,354) nem regime degenerado (o gate
      passou com 14× a chance). É **efeito de piso** · `docs/results/retention_floor_effect.md`
- [ ] 3.5 **CONGELADA.** As 10 seeds da banda 12.000.017+ **não foram gastas** —
      rodá-las queimaria a banda num resultado vazio. Retomável em 28 min de GPU
      se o endpoint for re-registrado. **O endpoint não foi trocado de propósito:**
      `average_forgetting` ordena os braços, e promovê-lo depois de ver isso
      seria pesca de endpoint.

> ⚠️ **Nenhum gate pegou.** O §G.1 vigia FAA contra a chance (passou com 14×);
> V5–V7 vigiam ranking, sobreposição e regime. **Nenhum vigia se o primário tem
> dispersão.** Segunda ocorrência do padrão do item 5.1, e de espécie diferente:
> em R-E a métrica estava calculada *errada*; aqui está calculada *certa* e não
> responde nada neste regime.

---

## 🟣 Fase 3b — instruction tuning como palco do instrumento `[ICML §4, corpo]`

**Decidido com o Fachel em 05/10 (D6 = opção A).** O alvo é instruction tuning,
pelo motivo certo: não é onde o SlowHeat precisa ganhar, é onde o instrumento
precisa morder. Ainda **não pré-registrado, não orçado, não autorizado.**

**Claim desta fase:**

> Em instruction tuning, ganhos reportados de métodos de CL não sobrevivem ao
> pareamento por plasticidade efetiva — inclusive os nossos.

**Por que este enquadramento e não "nosso método ganha":** a evidência do próprio
repo refuta a versão forte, e ela está nos artefatos, não na minha memória:

| achado | artefato | o que diz |
|---|---|---|
| `exact − frozen_a_control` = −0,0092, **5+/5−**, p = 1,000 | `exact_decomposition_results.md` | sob pareamento, nossa máscara não acrescenta nada a **LoRA-FA** (arXiv 2308.03303, 2023), e custa 1,8× mais tempo |
| nenhum contraste de FAA sobrevive a Holm (2/24 sobrevivem, ambos retenção) | `qwen_confirmation_results.md` §8 | em Qwen com 10 tarefas, acurácia não é resultado |
| hard: esquecimento **−11,68 pp** (10/10) **e** acurácia **−1,41 pp** (10/10) | compêndio bloco 6, CIFAR-100 MLP | a conjunção "menos forget + mais acurácia" se parte, com significância nos dois lados |
| `er_ace`: menor esquecimento da tabela (0,0398), FAA 6 pp **abaixo** | compêndio bloco 2, 100 seeds | esquecer pouco por ter aprendido pouco |

**A tese sobrevive sem precisar vencer.** Se o SlowHeat ganhar em instruction
tuning, entra como linha de tabela. Se não ganhar, o artigo continua inteiro —
é o mesmo movimento da Seção 3, que já publica o próprio erro.

> ⚠️ **O que porta e o que não porta.** `E` e o pareamento por plasticidade
> portam para loss de geração **sem mudança de definição** (são razão de normas
> de update, indiferentes à loss). O endpoint de retenção **não** porta:
> `task1_retention` em geração não tem piso de chance definido, e o gate de
> quase-chance (§G.1) é definido contra chance uniforme de classes. Sem endpoint
> novo pré-registrado isto cai no buraco da Fase 3.5: medição certa, regime sem
> dispersão, seeds queimadas.

### Evidência que já existe e não precisa de GPU nova

O argumento central já está medido. Esta tabela é o coração da Seção 4:

| braço | posição **sem** pareamento | `E` medido | posição **com** `E` = 0,85 |
|---|---:|---:|---:|
| `slice` | **2º** | **0,0625** | **4º** |
| `exact` | 1º | 0,9234 | 1º |
| `vanilla` | 3º | 1,000 | 2º |

`results/qwen_lora_iso_10seed/`, compêndio bloco 12. O `slice` aparecia em
segundo porque tinha removido **94% da plasticidade**. Sem o instrumento isso é
invisível. **Este par de tabelas é o artigo.**

E o contraste pareado confirmatório (bloco 14, `plasticity_matched/analysis.json`,
10 seeds, Holm): `frozen_a_control − lr_control` dá esquecimento −0,1260 (0+/10−,
p_Holm = 0,00391) e FAA +0,0992 (10+/0−), com plasticidade de superfície
**idêntica** (0,62246) nos dois braços. Isola *como* contra *quanta* — mas o
vencedor é LoRA-FA, e isso vai dito no texto, não escondido.

### Tarefas

- [ ] 3b.1 **Benchmark: TRACE no corpo, SuperNI como fallback barato.**
      **Sem juiz LLM** — AlpacaEval/MT-Bench trazem custo de API e ruído que não
      dá para pré-registrar. LongBench não cabe em 11-12 GB.

| benchmark | o que mede | custo/risco |
|---|---|---|
| **TRACE** | CL para LLM alinhado, 8 tarefas, BWT + perda de alinhamento geral | desenhado para a pergunta; tarefas longas, caro em Pascal |
| **SuperNI** | generalização a instrução não vista, exact-match + ROUGE-L | métrica automática, barata; pode saturar em 0.5B |
| ~~LongBench/MTL5~~ | contexto longo | **descartado**: inviável em 11-12 GB fp16 |
| ~~AlpacaEval/MT-Bench~~ | juiz LLM | **descartado**: não pré-registrável |

- [ ] 3b.2 **Medir capacidade antes de prometer tabela.** Qwen2.5-0.5B + SlowHeat
      mediu 5,447 GiB de pico em CLINC150 (fp16) e 5,52-6,60 GiB na confirmação de
      10 tarefas (fp32). Instruction tuning tem sequência mais longa: refazer a
      medida **antes** de dimensionar o sweep. Pascal (CC 6.x): fp16, nunca bf16,
      uma GPU só, sem DDP (GPUs heterogêneas).
- [ ] 3b.3 **Endpoint novo, pré-registrado, com gate de regime.** Declarar antes
      de ver qualquer número: exact-match por tarefa, ROUGE-L, ou delta de
      perplexidade na tarefa 1. O gate equivalente ao §G.1 tem de ser redefinido
      junto — em geração, contra o quê se mede "quase-chance"? Sem resposta a
      isso, não há gate, e a Fase 3.5 provou o que acontece então.
- [ ] 3b.4 **Replicar a reordenação do bloco 12 em host generativo.** Este é o
      endpoint primário da fase: *a ordenação dos braços muda quando `E` é
      pareado?* Não é "o SlowHeat ganha?". Reportar a ordenação nas duas condições,
      como no bloco 12.
- [ ] 3b.5 **`lr_control` obrigatório**, e **LoRA-FA/`frozen_a_control` obrigatório**
      como baseline publicada. O R-C mostrou que ele é o verdadeiro vencedor no
      host Qwen; omiti-lo aqui seria a mesma pesca que o artigo denuncia.
- [ ] 3b.6 Rodar só depois de 3b.1-3b.3 fechados e com autorização explícita de GPU.

**Ameaça ao cronograma, não resolvida:** ~16 semanas, dois artigos, um autor, e o
gargalo declarado é escrita. Isto é experimento novo com protocolo novo. Se
apertar, a candidata a corte é a **Fase 3 do IJCNN**, que já está congelada por
endpoint morto (3.5) — não a escrita. E a Seção 4 tem plano B sem GPU nenhuma: o
bloco 12 e o bloco 14 já sustentam a tese em Qwen, só que em classificação.

---

## ✍️ Fase 4 — escrita IJCNN (6 pgs) `[31/01]`

**Título:** *Activation Magnitude Is Enough: Gradient-Free Neuron Selection for CL in Transformers*
**Claim:** `|z|` puro iguala `|z·dL/dz|` sob capacidade pareada; ambos batem
máscara aleatória por +12,23 pp. O componente caro não compra nada.

- [ ] 4.1 `article/ijcnn/manuscript.md` — cada seção com o arquivo de resultado que a sustenta
- [ ] 4.2 `scripts/build_ijcnn_tables.py --check` (tabelas dos JSON, nunca copiadas à mão)
- [ ] 4.3 Escrever na ordem **3 → 4 → 5 → 2 → 6 → 1** (introdução por último)
- [ ] 4.4 LaTeX IEEE, 6 pgs, via CMT — 3 dias de folga para estouro de páginas

| seção | pgs | fonte |
|---|---|---|
| 1 Introduction | 0,75 | — |
| 2 Related Work | 0,5 | AWARe (2608.11758) — o braço `magnitude` é o critério deles |
| 3 Method | 1,0 | `bert_slowheat_diagnostic.py:80` |
| 4 Experiments | 2,0 | `criterion_ablation_results.md` + Fase 3 |
| 5 Replay selection | 0,75 | `replay_selector_results.md` (corta se apertar) |
| 6 Limitations | 0,5 | — |

---

## ✍️ Fase 5 — escrita ICML (8 pgs) `[~22/01]`

**Título:** *Plasticity-Matched Controls: Re-evaluating Continual Learning for Instruction-Tuned LLMs*
**Claim:** comparações em CL não controlam quanta plasticidade o método removeu.
Damos o instrumento, aplicamos a instruction tuning, e ele muda conclusões —
inclusive as nossas, pré-registradas e já confirmadas.

> **Mudança de enquadramento, 05/10 (D6 = A).** O título anterior era
> *...Against What They Actually Constrain*, genérico quanto ao domínio. O alvo
> agora é declarado: **instruction tuning**. A tese não mudou; o palco mudou, e
> com ele a resposta à objeção "vocês só testam em benchmark dummy".
>
> **O que foi rejeitado, e por quê:** a alternativa era vender "método que
> diminui forgetting e aumenta acurácia, validado em instruction tuning". Três
> medições do próprio repo matam essa versão — `exact − frozen_a_control` = 5+/5−
> contra LoRA-FA (R-C); nenhum contraste de FAA sobrevivendo a Holm em Qwen
> (`qwen_confirmation_results.md` §8); e a conjunção se partindo com
> significância nos dois sentidos em CIFAR-100 (compêndio bloco 6). Detalhe na
> Fase 3b.

- [ ] 5.1 **Escrever a Seção 3 primeiro** (é o coração): um resultado
      confirmatório com Holm, sinal exato e 5 gates descreveu condições que não
      existiam, e nenhum gate podia detectar — todos verificavam a métrica, e era
      a métrica que estava errada. **Sem suavizar.** ⚠️ Agora há **dois** casos:
      R-E (métrica calculada errada) e o efeito de piso de 05/10 (métrica
      calculada certa que não responde nada no regime) ·
      `docs/results/retention_floor_effect.md`
- [x] 5.2 Canibalizado. Seções 3 e 4.4 salvas em
      `article/icml/salvaged_from_manuscript.md` com enquadramento de onde
      encaixam; `git mv` para `article/_archive/manuscript_functional_slowheat.md`
      com cabeçalho dizendo o que foi contraditado por evidência posterior.
      **Não apagado.** 29 referências ao caminho antigo atualizadas em 4 arquivos,
      e o teste `test_hard_vs_soft_documents.py` — que quebrou com o `git mv` —
      aponta para o arquivo, não foi removido.
- [ ] 5.3 Seção 4 com os resultados da Fase 2
- [ ] 5.4 Seções 1, 2, 5, 6, 7
- [ ] 5.5 LaTeX 8 pgs

| seção | pgs | conteúdo |
|---|---|---|
| 1 Introduction | 1,0 | o confundimento de plasticidade, **ancorado em instruction tuning** |
| 2 Effective plasticity | 1,5 | definição, lema `E_sup ≤ E_bind`, generalização para penalidade e para loss de geração |
| 3 Case study: nosso próprio resultado | 1,5 | R-E + pareamento. **O caso central.** |
| 4 Re-evaluation | 2,0 | **bloco 12 (reordenação sob pareamento) + Fase 3b** em host generativo; Fase 2 (amplificação, G8) comprimida |
| 5 Negative results | 1,0 | R-C (**nosso método não bate LoRA-FA sob pareamento**), R-F. R-A/R-B só como citação ao companion |
| 6 Limitations | 0,5 | não-unicidade, hosts, benchmarks |
| 7 Related Work | 0,5 | LoRA-FA (2308.03303), TRACE, O-LoRA/InfLoRA |

---

## 📦 Fase 6 — artefato

- [ ] 6.1 Suíte limpa: `868 passed, 1 skipped` (o skip precisa de CUDA, pré-existente)
- [ ] 6.2 `docs/reproducibility.md` — um comando por tabela dos dois artigos
- [ ] 6.3 Checklist de submissão, **honesto sobre os resultados negativos —
      eles são o conteúdo, não fraqueza a minimizar**

---

## ⚙️ Regras de execução

1. **TDD por tarefa de código.** Teste primeiro, **veja falhar**, implemente,
   veja passar, commit. Teste que nunca foi vermelho não provou nada.
2. **Mutação nos caminhos críticos.** Verde de primeira é sinal de alerta.
   Mutação sobrevivente = o teste que falta importa mais que a próxima tarefa.
3. **Nunca `pytest ... | tail`.** Use `> log 2>&1; echo EXIT=$?`. O pipe esconde
   exit 4 de erro de parsing de argumento.
4. **Suíte inteira antes de cada commit**, não só o arquivo tocado.
5. **Nunca atualize um hash de pré-registro para fazer um teste passar.**
   A correção é excluir o campo novo do payload quando o método não é usado.
6. **Runs longas destacadas:** `setsid nohup ... < /dev/null & disown`, e
   **verificar `PPID=1`**. Use `scripts/launch_*.sh` commitado, não redigite.
   Em GPU, `CUDA_DEVICE_ORDER=PCI_BUS_ID` — sem isso o índice não bate com o do
   `nvidia-smi` (verificado por UUID nesta máquina).
7. **Autorização explícita antes de qualquer GPU.**
8. **Emenda a protocolo congelado:** só com evidência mecanismo-only, documento
   commitado **antes** da run, e linha no change log dizendo o que mudou, por
   quê, e que nenhuma acurácia foi lida. Runs afetadas são **descartadas**, não
   reinterpretadas.

---

## ⚠️ Riscos vivos

| # | risco | mitigação |
|---|---|---|
| R1 | Deadline do ICML não confirmado | `venue_deadlines.md` semanal; planejar para 22/01 |
| R2 | Dois artigos, um autor, 16 semanas | IJCNN reusa evidência pronta; se apertar, cortar Fase 3, nunca a escrita |
| R3 | ~~Passada 2 pode dar contraste vazio (`E_ewc ≈ 0,996`)~~ **resolvido pelo L2**: sob SGD os `E` ficam entre 0,955 e 0,993 | decidir com o L3 (2.12) se há poder antes de gastar seeds |
| R4 | ~~Cosseno do MAS < 0,9 no CIFAR~~ **confirmou-se**, mínimo −0,88: o update inverte de direção | §E.1 manda reportar com ressalva; **proibido remover o arm** (seleção pós-hoc). Vira conteúdo da Seção 4 |
| R5 | Sobreposição vista como submissão dupla | regra de não-sobreposição; citar como companion |
| R6 | **Fase 3b não cabe no caminho crítico sem corte** — experimento novo, protocolo novo, e o gargalo é escrita | ordem de corte declarada: **Fase 3 do IJCNN primeiro** (já congelada por endpoint morto), nunca a escrita. Plano B sem GPU: bloco 12 + bloco 14 sustentam a Seção 4 em classificação |
| R7 | **Reviewer pergunta "e o método de vocês, ganha?"** e a resposta medida é 5+/5− contra LoRA-FA | responder na Seção 5, não evitar. É o mesmo movimento da Seção 3: o artigo ganha credibilidade publicando o próprio negativo. **Proibido omitir o `frozen_a_control` das tabelas** |

## 🚫 Fora de escopo

- **4× A6000** — sem acesso confirmado; tudo cabe nas 3 GPUs locais
- **Split-CIFAR100 com backbone CNN** — o loader entrega vetores achatados e
  `conv2d` recebe `[N, 3072]`. O §D do protocolo do piloto diz "(CNN)" e está
  **errado**; a passada 1 roda em MLP. Corrigir o §D ou implementar o reshape.
- **O-LoRA / InfLoRA / N-LoRA em LLaMA-7B** — upside, não caminho crítico
