# Resultado — confirmação do LoRA exato produtor-only

**Estado:** run confirmatória **concluída em 28/09/2026**, 10/10 seeds.
**Protocolo:** [`goals/protocol_lora_confirmation.md`](../../goals/protocol_lora_confirmation.md), congelado em 25/09 antes da primeira seed.
**Dados:** `results/qwen_lora_confirmation/seed_{700001..925097}/manifest.json`, agregado em `sweep.json`.
**Commit:** `475d3c6ed39ff1de13625519dcbae52764ad8667`.
**Custo:** 5.480 s de wall-clock do orquestrador, 3 GPUs, 30 runs.

---

## A pergunta confirmatória

Seção B do pré-registro:

> Dada a mesma quantidade de plasticidade efetiva removida, proteger
> **seletivamente** as unidades importantes reduz o esquecimento mais do que
> reduzir o learning rate **uniformemente**?

Não é "proteger ajuda?" — isso é compatível com "qualquer freio ajuda". É "a
**distribuição** da proteção importa?".

---

## Resultado

| braço | FAA (média±sd) | forgetting | E_eff | params treináveis | s/seed |
|---|---|---|---|---|---|
| **exact** | **0,6384±0,0314** | **+0,3278±0,0365** | 0,850 | 4.214.016 | 624,8 |
| lr_control | 0,5790±0,0468 | +0,4200±0,0549 | 1,000 | 6.769.920 | 355,8 |
| vanilla | 0,5706±0,0273 | +0,4296±0,0283 | 1,000 | 6.769.920 | 353,6 |

### Endpoint primário (A3): `exact − lr_control`, forgetting

| grandeza | valor |
|---|---|
| diferença média | **−0,0921** |
| mediana pareada | −0,0998 |
| sinais | **9−/1+** |
| `p` (sinal exato bicaudal) | **0,02148** |
| critério A8 (`p < 0,025` **e** mediana < 0) | **ATENDIDO** |

> **Confirmado.** A proteção seletiva reduz o esquecimento mais do que a redução
> uniforme de learning rate na mesma plasticidade efetiva.

### Família de Holm declarada em A6 (2 comparações)

| comparação | diferença | `p` | `p_Holm` | veredito |
|---|---|---|---|---|
| forgetting vs `lr_control` | −0,0921 | 0,02148 | **0,0430** | sobrevive |
| FAA vs `lr_control` | +0,0594 | 0,10938 | 0,1094 | não sobrevive |

### Endpoints secundários contra vanilla (A4)

| comparação | diferença | sinais | `p` |
|---|---|---|---|
| forgetting vs vanilla | −0,1017 | **10−/0+** | 0,00195 |
| FAA vs vanilla | +0,0678 | 9+/1− | 0,02148 |

Reportados como secundários e **não promovidos**, conforme A4.

---

## Verificação da seção H

| # | Exigência | Medido | Veredito |
|---|---|---|---|
| H1 | `E_eff = 0,8500` (tol. 1e-3) para `exact` em todas as seeds | `0,849999997917 ± 6,9e-09` | **passa** |
| H2 | `lr_control` sem máscara, `effective_learning_rate = 8,5e-5` | `8,5e-05` nas 10 seeds | **passa** |
| H3 | `vanilla` com `E = 1,0` e `lr = 1e-4` | `E=1,0`, `lr=1e-4` | **passa** |
| H4 | tokens idênticos entre braços dentro de cada seed | 0 seeds divergentes | **passa** |
| H5 | `claim_scope` declara pareamento e controle | presente em `sweep.json` | **passa** |

Config constante nas 10 seeds em todos os 17 campos não relacionados a seed e
device. `missing_seeds` vazio, `completed_seeds` = 10.

---

## Duas ressalvas de registro

### R1 — A seção E do pré-registro descreve dois valores que não foram os executados

| item | seção E diz | executado (confirmatória **e** exploratória) |
|---|---|---|
| `alpha` | 32 | **16** |
| `max_length` | 64 | **48** |

A confirmatória usou **exatamente o mesmo config da exploratória** que a
motivou, então a comparação entre runs permanece válida e nenhum braço foi
favorecido. É erro de transcrição no documento do protocolo, não desvio de
execução. Exige linha na tabela K e correção da seção E.

### R2 — O confundimento de aquisição permanece, como previsto em J

`exact` treina **4.214.016** parâmetros contra **6.769.920** dos outros, porque
congela `A` — 62% da capacidade do adaptador. A leitura *ele esquece menos
porque aprendeu menos* **não está descartada por esta run**. `E` mede
plasticidade retida da máscara, não capacidade do adaptador.

Resolver isso é [QB-2](../../goals/proximo_passo_qwen.md): `exact r=32` contra
`vanilla r=16`, que iguala parâmetros treináveis. Até lá, o resultado acima é
reportável **com esta ressalva anexada**, nunca sem ela.

### R3 — O pareamento por plasticidade efetiva não valia (registrado em 29/09/2026)

R2 acima trata capacidade do adaptador e plasticidade efetiva como dois
problemas distintos. **São o mesmo**, e a verificação H1 mediu só metade dele.

`effective_plasticity()` promedia a máscara **sobre os parâmetros que carregam
binding de máscara**. No `exact` o único binding é `lora_b.weight`; os 2.555.904
parâmetros de `A` têm fator de update exatamente zero e **não entram na média**.
H1 verificou `E = 0,8500` e passou — sobre a superfície errada.

Medido sobre a superfície treinável do braço de referência:

| braço | `E` reportado | `E` de superfície |
|---|---:|---:|
| `vanilla` | 1,000 | 1,000 |
| `lr_control` | 1,000 | 0,850 |
| `exact` | **0,850** | **0,532** |

**O endpoint primário comparou 0,532 contra 0,850.** O contraste
`exact − lr_control` não era pareado em plasticidade, e a diferença de 32 pontos
está na direção que favorece o braço tratado.

Isto **não** anula o que foi medido; muda a descrição das condições. A
afirmação "sob plasticidade efetiva pareada", que aparece neste documento e em
`proximo_passo_qwen.md` B.1, **não pode ser reportada** até que a run pareada
sobre a superfície feche.

Combinado com a [decomposição](../results/exact_decomposition_results.md) — que
mostrou que todo o efeito vem de congelar `A` e nada da máscara —, o quadro é
coerente: a máscara, que é o que `E` media, não faz nada; o congelamento, que
`E` não via, faz tudo.

Instrumento da correção: `surface_plasticity()` em
`src/dual_heater/lora_slowheat.py`, com o lema `E_superfície <= E_bindings`
travado em `tests/test_surface_plasticity.py`. A run que re-testa o contraste
sob pareamento correto está pré-registrada em
[`goals/protocol_plasticity_matched.md`](../../goals/protocol_plasticity_matched.md).

---

## O que este resultado sustenta e o que não sustenta

**Sustenta:**

- Na mesma plasticidade efetiva removida, a distribuição da proteção importa: a
  máscara seletiva bate o escalar uniforme de learning rate em forgetting, com
  pré-registro congelado antes da primeira seed e critério declarado.
- O `lr_control` **não** é um alvo fácil: ele é pior que o vanilla nos dois
  endpoints (FAA 0,5790 contra 0,5706; forgetting +0,4200 contra +0,4296),
  replicando o achado exploratório de que reduzir LR uniformemente atrapalha.

> **Correção de 29/09/2026 — esta afirmação estava errada neste documento.**
>
> As médias citadas acima estão certas, mas a leitura não: **a diferença não é
> significativa e o sinal é o oposto do que o texto diz.** Recomputado dos dez
> manifestos desta própria run:
>
> | contraste | diferença pareada | sinais | p |
> |---|---:|---:|---:|
> | FAA `lr_control − vanilla` | **+0,0084** | 6+/4− | 0,754 |
> | forgetting `lr_control − vanilla` | **−0,0096** | 3+/7− | 0,344 |
>
> FAA maior e forgetting menor significam `lr_control` **melhor** que vanilla na
> média, não pior — e nenhum dos dois passa longe da significância. A frase
> "replicando o achado exploratório" é falsa: o achado exploratório **não
> replicou**.
>
> Leitura correta: **nula.** Isso não afeta o endpoint primário, que é
> `exact − lr_control` e foi pré-registrado. Fortalece o controle, removendo a
> objeção de que o falsificador foi escolhido por ser ruim.

**Não sustenta:**

- Ganho de FAA contra `lr_control`: `p_Holm = 0,1094`, 8/10 seeds. Não é
  significativo e não deve ser reportado como positivo.
- Que o efeito seja mecanismo e não capacidade do adaptador (R2).
- Qualquer alegação de prioridade. O-LoRA, InfLoRA e a família LoRA-CL seguem
  não levantados (seção J do pré-registro, e QB-3).
- Generalização: um modelo, um `r`, um benchmark, um ponto de plasticidade.

---

## Comparação com o exploratório

| contraste | exploratório (10 seeds, múltiplos de 11) | **confirmatório (10 seeds, banda 700001+)** |
|---|---|---|
| forgetting vs `lr_control` | −0,1190 (p=0,0020, 10/10) | **−0,0921 (p=0,0215, 9/10)** |
| FAA vs `lr_control` | +0,0810 (p=0,0215, 9/10) | +0,0594 (p=0,1094, 8/10) |
| forgetting vs vanilla | −0,0807 (p=0,0020, 10/10) | −0,1017 (p=0,0020, 10/10) |
| FAA vs vanilla | +0,0478 (p=0,0215, 9/10) | +0,0678 (p=0,0215, 9/10) |

O efeito primário **encolheu** do exploratório para o confirmatório (−0,119 para
−0,092) e perdeu uma seed de concordância. Isso é o padrão esperado de regressão
à média entre exploração e confirmação, e é a razão de a confirmação existir. O
contraste contra vanilla, que não era o endpoint primário, ficou mais forte.

## Referências

- [Pré-registro congelado](../../goals/protocol_lora_confirmation.md)
- [Benchmark exploratório de LoRA](../lora/lora_qwen_benchmark_results.md)
- [Os três mecanismos falhados](../lora/lora_slowheat_mechanisms.md)
- [Próximos passos do host Qwen](../../goals/proximo_passo_qwen.md)
