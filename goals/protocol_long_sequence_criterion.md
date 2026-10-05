# Protocolo congelado — segundo host do `|z|`: sequências longas

**Congelado em:** 2026-10-05, antes da primeira seed da banda 12.000.017+.
**Item 3.1–3.3** de `goals/roadmap_icml_ijcnn.md`. Corpo do **IJCNN**.
**Estende, não substitui,** `goals/protocol_importance_criterion_ablation.md`,
que permanece congelado e cujos resultados (`docs/results/criterion_ablation_results.md`)
não são reinterpretados por este documento.

> ⚠️ **ESTA PASSADA LÊ ACURÁCIA.** O pré-registro é este arquivo. O resultado que
> ele produzir é o resultado, seja ele qual for.

---

## A. O limite que isto ataca

R-A e R-B são os dois achados que sustentam o IJCNN:

| # | achado | evidência |
|---|---|---|
| R-A | proteção por ativação bate máscara aleatória: **+12,23 pp**, 10/10 seeds, `p_Holm = 0,0039` | `criterion_ablation_results.md` |
| R-B | o gradiente não compra nada: `\|z·dL/dz\|` empata com `\|z\|` (−0,17 pp, 5+/5, `p = 1,00`) | idem |

Os dois carregam o mesmo limite declarado, escrito na §J do pré-registro
original e repetido na seção de limites do resultado:

> *Um host (BERT), um benchmark (Split-CLINC150), **duas tarefas por sequência**.*

Duas tarefas é o limite mais citável contra o artigo: um revisor pergunta se o
empate `|z|` vs `|z·dL/dz|` não é um artefato de uma sequência curta demais
para que a diferença de critério se acumule. É uma pergunta legítima, e a
resposta atual é "não sabemos".

**Esta passada não troca de host.** Ela estica a sequência no mesmo host, que é
o ataque de menor risco ao limite de maior retorno. Trocar de host ao mesmo
tempo confundiria as duas coisas.

## B. Correção de instrumento aplicada antes (mecanismo-only)

`condition_endpoints` lia `validation_accuracy_matrix[1]` e
`parameter_drift_history[1]`, isto é, **o estágio 1**, não o último. Com `T = 2`
os dois coincidem e os resultados publicados estão corretos. Com `T > 2` a
função reportaria "retenção da tarefa 1" medida logo após a tarefa 2, enquanto
o run seguiu até a tarefa 5 ou 10 — um número plausível, errado, e sem nada no
artefato que o denunciasse.

Corrigido em `experiments/bert_slowheat_diagnostic_common.py` com teste de
não-regressão para `T = 2` e 19 mutações mortas. **Nenhuma acurácia foi lida
para decidir a correção**, e os endpoints de `T = 2` são bit-idênticos aos
publicados — verificado por teste, não por inspeção.

Esta correção **não invalida** a run de 28/09: aquela run tem `T = 2`, onde
`matrix[1]` *é* o último estágio.

## C. Métrica faltante da §F, recuperada

A §F do pré-registro original declara **duas** diagnósticas obrigatórias:
a variância do ranking normalizado **e** a sobreposição top-k entre os braços.
A run de 28/09 emitiu só a primeira: a sobreposição é uma grandeza *entre
braços* (compara dois runs separados) e o hook do runner, que roda dentro de um
único run, não tinha acesso ao segundo braço. `top_k_overlap` nunca foi
calculado.

Sem ela, a regra de interpretação congelada — sobreposição(magnitude, aleatório)
> 0,8 declara o braço degenerado e torna qualquer empate **inconclusivo** — não
podia ser avaliada, e o empate de R-B apoiava-se só na variância.

Recuperada em `scripts/analyze_criterion_degeneracy.py` a partir dos buffers
`slow_heat` salvos, que **são** as máscaras que os runs usaram:

| par | média (10 seeds) | mínimo | máximo |
|---|---:|---:|---:|
| `magnitude` vs `aleatório` | **0,5748** | 0,5481 | 0,6731 |
| `magnitude` vs `funcional` | 0,8814 | 0,8758 | 0,8851 |

**0,5748 < 0,80: o falsificador da §F passa.** O braço `magnitude` protegeu
unidades próprias, não uma máscara quase-aleatória. O empate de R-B é achado, e
agora as duas diagnósticas obrigatórias existem. Capacidade pareada verificada
em 10/10 seeds. Artefato: `results/criterion_ablation/degeneracy_overlap.json`.

Os 0,8814 entre `magnitude` e `funcional` são, eles próprios, conteúdo: os dois
critérios selecionam quase o mesmo conjunto. **Isso é consistente com o empate e
não é um segundo achado independente dele** — é a mesma observação medida na
máscara em vez de na acurácia.

## D. Pergunta confirmatória

Em BERT/Split-CLINC150 com máscara hard e capacidade pareada, **em sequências de
5 tarefas**, o empate `funcional ≈ magnitude` e a vantagem `magnitude > aleatório`
se mantêm?

## E. Decisões congeladas

| # | decisão | valor | justificativa |
|---|---|---|---|
| S1 | Host | BERT-mini + Split-CLINC150, class-IL, cabeça 150-way | o mesmo de R-A/R-B; trocar host confundiria dois efeitos |
| S2 | **Comprimento da sequência** | **`T = 5`** | 5, não 10: ver §F |
| S3 | Braços | `vanilla`, `slowheat_random_hard`, `slowheat_magnitude_hard`, `slowheat_hard` | os quatro de I2/I3, sem alteração |
| S4 | Seeds | 10, banda **12.000.017+** (§E.1) | disjunta de todas as bandas gastas, verificado por teste |
| S5 | Endpoint primário | retenção da tarefa 1 **após a última tarefa** (`task1_retention`) | o mesmo endpoint de I5, agora corretamente definido para `T > 2` |
| S6 | Secundários | FAA, esquecimento, `last_task_acquisition`, variância do ranking, sobreposição top-k | reportados sempre, nunca promovidos |
| S7 | Família confirmatória | **2 contrastes**: `magnitude − aleatório` e `funcional − magnitude` | as mesmas duas arestas de I7 |
| S8 | Teste | sinal exato bilateral sobre as 10 diferenças pareadas por seed | o mesmo do projeto |
| S9 | Correção múltipla | Holm sobre a família de 2, α = 0,05 | o mesmo de I9 |
| S10 | Hiperparâmetros | **idênticos** a I11: `slow_strength = 3,0`, `epochs_per_task = 4`, `max_length = 128`, `batch_size = 2`, `plasticity_budget = 0,25` | re-tunar por comprimento destruiria a comparação com `T = 2` |
| S11 | Capacidade | `plasticity_budget` idêntico entre os braços, verificado nos artefatos | o mesmo de I10 |
| S12 | Diagnósticas da §F | **ambas obrigatórias**, incluindo a sobreposição entre braços | o que faltou em 28/09 |

### E.1 Banda de seeds

```
12000017, 12025031, 12050033, 12075059, 12100063,
12125083, 12150107, 12175117, 12200129, 12225149
```

Disjunta de 4.000.003+ (a ablação `T = 2`), de `[2, 9, 10, 28, 30, 32, 57, 67,
2005, 2012]` e `[11, 22, 33]` (BERT anterior), e de todas as outras bandas do
repositório. **Verificado por teste, não por inspeção.**

### E.1.1 A seed de calibração fica FORA da banda

```
11900003
```

A calibração do §I roda a matriz completa de 4 braços e portanto **escreve
acurácia**. Gastar nela uma seed da banda significaria ler a FAA de uma seed
confirmatória antes das outras nove existirem, e qualquer decisão posterior
sobre o regime (mais épocas, lr maior) teria sido tomada com informação vinda da
banda. Medir custo de parede não exige seed da banda.

**Isto diverge de `scripts/run_criterion_calibration.py`**, que calibrou em
`IMPORTANCE_CRITERION_ABLATION_SEEDS[0]` e depois rodou a mesma seed entre as
dez confirmatórias. Aquela run está publicada e não é reinterpretada aqui; a
divergência é declarada porque é uma mudança de prática, não um detalhe.

Pinado em `tests/test_long_sequence_criterion_seeds.py` e verificado em tempo de
execução por `scripts/run_long_sequence_calibration.py`, que se recusa a rodar
com uma seed da banda.

### E.2 Por que não reusar as seeds de 28/09

Seria tentador parear `T = 2` contra `T = 5` nas mesmas seeds para um contraste
dentro-de-seed. **Não é feito**, pelo mesmo motivo de I4: as seeds de 4.000.003+
já produziram um resultado publicado, e reusá-las acopla a passada nova ao ruído
particular daquela. O contraste entre comprimentos é **entre bandas** e está
declarado como descritivo, não confirmatório (§G).

## F. Por que `T = 5` e não `T = 10`

> ⚠️ **DECLARAÇÃO DE LEITURA DE ACURÁCIA.** Os smokes citados abaixo **leram
> FAA**. Eles rodaram numa configuração que **não é** a deste protocolo — 1
> época em vez de 4, 2 braços em vez de 4, CPU em vez de GPU, e a seed
> `4_000_003`, que pertence à banda já gasta da ablação `T = 2` e **não** à banda
> desta passada. Nenhum contraste confirmatório foi calculado e nenhuma seed da
> banda 12.000.017+ foi tocada. Mas a leitura aconteceu, ela informou a escolha
> de `T = 5` contra `T = 10`, e está escrita aqui em vez de escondida. Quem
> considerar isso contaminação deve tratar a escolha de `T` como decisão de
> desenho tomada com informação, não como pré-registro cego.

Declarado antes das seeds confirmatórias, com o custo medido:

1. **Custo medido, não estimado.** Smoke em CPU, 1 seed, 2 braços, 1 época:

   | `T` | tempo | FAA `vanilla` | FAA `magnitude` |
   |---:|---:|---:|---:|
   | 3 | 123 s | 0,0544 | 0,0400 |
   | 5 | 213 s | 0,0313 | 0,0360 |
   | 10 | 456 s | 0,0420 | 0,0167 |

   O custo é aproximadamente linear em `T`. Com 4 braços, 4 épocas e 10 seeds,
   `T = 5` projeta-se na mesma ordem dos 10,9 min medidos para `T = 2`, vezes o
   fator de comprimento. `T = 10` dobraria isso.

2. **O piso de chance cai com `T`, e o smoke mostra que o risco é real.** Numa
   cabeça 150-way class-IL, a FAA em `T = 10` caiu para 0,0167 e 0,0420 contra
   chance `1/150 ≈ 0,0067` — **2,5× e 6,3× a chance, com 1 época**. Quatro épocas
   melhoram isso, mas a margem em `T = 10` é estreita demais para que um
   contraste significativo signifique alguma coisa. Em `T = 5` a chance é
   `1/75 ≈ 0,0133` e o smoke de 1 época já entrega 2,4–2,7×.

   **Este é exatamente o modo de falha que degradou a passada 2**
   (`goals/protocol_penalty_pass2.md`, D5 do roadmap): `p = 0,00049` com os três
   arms perto da chance, e o resultado teve de sair com ressalva. A lição está
   aplicada, não é palpite.

3. **5 já é 2,5× o limite declarado.** O limite citável é "duas tarefas". Cinco
   o quebra. Dez o quebra mais, mas num regime onde a medida pode não significar
   nada.

**Se o resultado em `T = 5` vier em quase-chance** — definido no §G.1 — ele é
reportado como inconclusivo, e `T = 10` **não** é a resposta.

## G. Predições declaradas antes da run

| # | predição | o que falsifica |
|---|---|---|
| **W1** | `magnitude − aleatório` continua **positivo e significativo** em `T = 5` | que a vantagem da proteção seletiva era artefato de sequência curta |
| **W2** | `funcional − magnitude` continua **não significativo** (`p >= 0,05` após Holm) | que o gradiente passa a comprar algo quando a sequência estica |
| **W3** | a sobreposição top-k `magnitude` vs `aleatório` permanece **abaixo de 0,8** | que o braço `magnitude` degenera com mais consolidações |

**W2 é declarada como nula de propósito**, na mesma direção de R-B. Dizer isso
antes de gastar as seeds é o oposto de escolher o resultado.

### G.1 Gate de quase-chance, declarado antes

Chance numa cabeça 150-way com 5 tarefas vistas é `1/75 ≈ 0,0133` sob o
mascaramento de logits não vistos; sem mascaramento, `1/150 ≈ 0,0067`.

**Regra congelada:** se a **FAA do melhor braço** ficar abaixo de **4× a chance
correspondente**, toda a família confirmatória é reportada como **inconclusiva
por regime degenerado**, independentemente do valor de `p`. Um contraste
significativo entre três arms que mal aprenderam não separa "o critério funciona"
de "um arm mudou menos".

Esta regra existe porque o projeto já pagou por não tê-la: a passada 2
(`6970361`) produziu `p = 0,00049` com os três arms entre 0,149 e 0,198 contra
chance 0,10, e o resultado teve de ser reportado com ressalva em vez de como
achado. **O gate está escrito antes das seeds, não depois.**

**Alerta honesto sobre o próprio gate:** o smoke de 1 época em `T = 5` entregou
FAA 0,0313 e 0,0360 contra chance 0,0133 — **2,4× e 2,7×, abaixo do limiar de
4×**. As quatro épocas de S10 devem melhorar isso (em `T = 2` a FAA com 4 épocas
foi 0,5723, contra chance 0,0333, ou seja 17×), mas **não há garantia**. É
plenamente possível que esta passada dispare o próprio gate e saia inconclusiva.

Isso é aceito de propósito. **O limiar não será afrouxado depois de ver o
resultado** — essa é a única coisa que torna o gate útil. Se ele disparar, o
próximo passo é melhorar o regime (mais épocas, lr maior, replay), com protocolo
novo, e não baixar o 4× para 3×.

### G.2 As saídas, com consequência escrita antes

- **W1 e W2 confirmadas:** o limite "duas tarefas" cai. A Seção 4 do IJCNN passa
  a reportar `T = 2` e `T = 5`, e o claim do artigo ganha uma dimensão sem mudar
  de forma. **Não autoriza dizer "generaliza para sequências longas"** — cinco
  não é longo.
- **W2 falsificada (o gradiente passa a comprar algo em `T = 5`):** é o resultado
  mais interessante possível e **muda o claim do artigo**. O título
  *"Activation Magnitude Is Enough"* deixa de ser defensável como está e vira
  algo como *"enough for short sequences"*. Exigiria replicação antes de
  qualquer afirmação forte, e a replicação é um protocolo novo.
- **W1 falsificada:** a vantagem da proteção seletiva era um efeito de sequência
  curta. **Isso derrubaria R-A**, o achado central do IJCNN, e a consequência
  seria repensar o artigo, não enterrar o resultado.
- **Gate do §G.1 disparado:** inconclusivo, reportado como tal, e o próximo passo
  é melhorar o regime (mais épocas, lr, replay), **não** esticar mais a
  sequência.

**Os quatro desfechos são publicáveis.**

## H. Implementação exigida antes da run

1. ✅ `condition_endpoints` lendo o último estágio, com teste de não-regressão
   para `T = 2` e `task_count` nos artefatos.
2. ✅ `run_diagnostic(task_limit=...)` propagando de verdade, com guarda de
   wiring do CLI (um flag que parseia e não chega no runner é pior que nenhum
   flag).
3. ✅ `summarize_diagnostic` recusando-se a agregar braços com `T` diferente.
4. ✅ Sobreposição top-k entre braços (`experiments/criterion_degeneracy.py`),
   com a regra da §F implementada e 16 mutações mortas.
5. ✅ Smoke end-to-end em CPU provando que `T = 3`, `T = 5` e `T = 10` rodam
   (1 época, 2 braços, seed da banda antiga; leitura de FAA declarada na §F).
6. ✅ Banda de seeds da §E.1 registrada em código e pinada em teste
   (`tests/test_long_sequence_criterion_seeds.py`).
7. ✅ Verificador pré-endpoint com o falsificador de variância do item 3.3 e o
   gate da §G.1 (`scripts/verify_long_sequence_criterion.py`, 16 mutações
   mortas).
8. ⬜ Calibração de **1 seed** medindo o custo real antes de lançar as 10 (§I).
9. ⬜ Suíte verde e este arquivo commitado **antes** da primeira seed.

## I. Custo

A §I do pré-registro original exige custo **medido**, porque estimativa por
analogia já falhou duas vezes neste projeto. A regra se mantém: **rodar 1 seed,
medir, e só então projetar.** Se a projeção passar de 4 h, reavaliar o escopo.

Referência conhecida: `T = 2`, 4 braços, 10 seeds, 1080 Ti = **10,9 min**
(66,4 s por seed, medidos).

**Esta passada exige GPU e portanto autorização explícita do Fachel.**

## J. O que este protocolo NÃO autoriza

- **Não autoriza trocar de host.** Um host. Qwen, LLaMA e LoRA estão fora.
- **Não autoriza `T = 10`** sem um documento novo. A §F explica por quê.
- **Não autoriza reinterpretar a run de 28/09.** Ela é `T = 2` e continua válida
  como está.
- **Não autoriza promover a sobreposição top-k a endpoint.** Ela é diagnóstica
  da §F, declarada como tal desde 28/09.
- **Não autoriza dizer "o resultado generaliza".** Cinco tarefas num host.

## K. Registro de alterações

| data | alteração | antes da run? |
|---|---|---|
| 05/10/2026 | criação e congelamento. S1–S12 fechados, gate de quase-chance do §G.1 declarado, banda de seeds fixada. Correções de instrumento das §B e §C aplicadas antes e mecanismo-only. | sim — nenhuma seed executada |
