# Protocolo congelado — capacidade pareada do braço `exact`

> Pré-registro. Escrito **antes** de qualquer seed deste experimento. Não é
> editado depois de ver acurácia; alterações exigem commit anterior à run e
> linha nova na tabela K.

**Estado: CONGELADO em 28/09/2026**, antes da primeira seed.

> ## ARQUIVADO EM 29/09/2026 — SUPERADO, NÃO EXECUTADO
>
> Este protocolo (QB-2) existe para decidir se o efeito confirmado do `exact` é
> mecanismo ou capacidade do adaptador, parcando contagem de parâmetros
> treináveis. **A decomposição do `exact` respondeu antes, e melhor.**
>
> Três razões para não gastar as ~2h de GPU:
>
> 1. **O braço de tratamento morreu.** `exact − frozen_a_control` deu 5+/5,
>    p = 1,00 ([registro](../docs/results/exact_decomposition_results.md)): a
>    máscara SlowHeat não contribui nada. O que resta do `exact` é LoRA-FA, e
>    testá-lo com rank elevado mede LoRA-FA sob capacidade pareada, não o
>    mecanismo da casa.
> 2. **A pergunta foi reformulada, não abandonada.** Capacidade do adaptador e
>    plasticidade efetiva são a **mesma grandeza** medida sobre escopos
>    diferentes (ver R3 em
>    [`docs/lora/lora_confirmation_results.md`](../docs/lora/lora_confirmation_results.md)).
>    O experimento que a responde agora é
>    [`protocol_plasticity_matched.md`](protocol_plasticity_matched.md), que
>    pareia por **plasticidade de superfície** em vez de por contagem de
>    parâmetros — mais geral, e ~1h10 em vez de ~2h.
> 3. **A limitação declarada em §J tornou-se decisiva.** `exact_r26` teria 26
>    direções aleatórias fixas contra 16 aprendíveis, então mesmo um resultado
>    positivo deixaria a explicação "mais direções, ainda que fixas, ajudam" em
>    aberto. O pareamento por superfície não tem esse problema porque não mexe
>    no rank.
>
> **Nada aqui foi desperdiçado.** A aritmética da seção F (rank pareado é 26,
> não 32) e o bug de `scaling = alpha/r` corrigido na seção G continuam válidos e
> estão no código, com teste. O documento permanece como registro de uma decisão
> tomada por aritmética antes de qualquer seed.

---

## A. O que motiva

A confirmação de
[protocol_lora_confirmation.md](protocol_lora_confirmation.md) fechou em 28/09
com o endpoint primário **confirmado**: `exact − lr_control` em forgetting,
diferença −0,0921, `p = 0,02148 < 0,025`, mediana −0,0998, 9/10 seeds.

A seção J daquele protocolo declarou, antes da run, um confundimento que ele
não resolve:

> `exact` treina 4.214.016 parâmetros contra 6.769.920 (r=16), porque congela
> `A`. `E` mede plasticidade retida da máscara, não capacidade do adaptador,
> então a leitura *esquece menos porque aprendeu menos* permanece possível.

Este protocolo existe para decidir essa leitura.

## B. Pergunta

Com o **número de parâmetros treináveis pareado**, o braço `exact` ainda reduz
o esquecimento em relação ao `vanilla`?

## C. Hipótese declarada antes da run

*`exact` com rank elevado reduz o forgetting médio em relação a `vanilla` com
`r = 16`, com a diferença pareada por seed negativa na maioria das seeds.*

Hipótese nula: as diferenças pareadas se distribuem simetricamente em torno de
zero. **Resultado nulo encerra a linha e deve ser reportado**: significaria que
o efeito confirmado em 28/09 é explicável por capacidade do adaptador.

## D. Decisões congeladas

| # | Decisão | Valor congelado | Justificativa |
|---|---|---|---|
| B1 | Braços | `exact_r26` (r=26, alpha=26, `A` congelada) e `vanilla_r16` (r=16, alpha=16) — exatamente dois | ver seção F: `r=26` é o rank que **de fato** pareia os parâmetros; `r=32` erra por 22,5% |
| B2 | Seeds | banda nova `1_000_003, 1_025_011, 1_050_017, 1_075_037, 1_100_043, 1_125_059, 1_150_061, 1_175_087, 1_200_089, 1_225_097` | disjuntas das confirmatórias do LoRA (700001–925097), das do Split-MNIST (104729–586429), das exploratórias (múltiplos de 11) e da banda de M1 (2000003+) |
| B3 | Endpoint primário | forgetting médio, contraste `exact_r26 − vanilla_r16` | mesmo endpoint da confirmação que este protocolo defende |
| B4 | Endpoints secundários | FAA (mesmo contraste); parâmetros treináveis por braço (verificação, não resultado) | reportados sempre, nunca promovidos |
| B5 | Teste | sinal exato bicaudal sobre as 10 diferenças pareadas por seed | mesmo teste da confirmação; sem suposição de normalidade em n=10 |
| B6 | Correção múltipla | Holm sobre a família de 2 (forgetting e FAA no contraste primário), α=0,05 | família declarada **antes** da run |
| B7 | Plasticidade alvo | `E = 0,85` para `exact_r26`, re-resolvida por bisseção em cada fronteira | mesmo valor da confirmação, para que os dois experimentos sejam comparáveis |
| B8 | Critério de sucesso | `p < 0,025` no endpoint primário **e** mediana negativa | idêntico ao A8 da confirmação |

## E. Cenário

Idêntico ao executado na confirmação de 28/09 — que **não** é idêntico ao
descrito na seção E daquele documento; ver a tabela K dele, linha de 28/09.

| Item | Valor |
|---|---|
| Modelo | `Qwen/Qwen2.5-0.5B`, `Qwen2ForSequenceClassification`, 150 rótulos |
| Precisão | fp32 (Pascal CC 6.x não tem bf16 nativo) |
| Alvos LoRA | `gate_proj`, `up_proj`, `down_proj` |
| **`exact_r26`** | **`r = 26`, `alpha = 26,0`** → `scaling = 1,0` |
| **`vanilla_r16`** | **`r = 16`, `alpha = 16,0`** → `scaling = 1,0` |
| Dataset | `clinc/clinc_oos:plus`, Class-IL por domínio, 10 tarefas |
| Dados por tarefa | 750 treino / 300 validação |
| Épocas | 3 |
| `batch_size` | 8 |
| `max_length` | 48 |
| `lr` | 1e-4 |
| Hardware | uma seed inteira por GPU, sem DDP |

## F. Por que `r = 26` e não `r = 32`

O plano original deste experimento pedia `exact r=32` contra `vanilla r=16`. A
contagem de parâmetros foi derivada **antes da run** a partir dos dois valores
medidos em `r = 16` (4.214.016 e 6.769.920) e da estrutura do adaptador:

```text
por camada LoRA:  A tem r*in_features,  B tem out_features*r
vanilla(r) = r * sum(in + out) + head
exact(r)   = r * sum(out)      + head        (A congelada)

head = 896 * 150 = 134.400        (hidden_size x num_labels)
sum(in + out)    = 414.720
sum(out)         = 254.976
```

Disto sai:

| configuração | parâmetros treináveis | razão vs `vanilla r=16` |
|---|---:|---:|
| `vanilla r=16` | 6.769.920 | 1,0000 |
| `exact r=16` | 4.214.016 | 0,6225 |
| `exact r=24` | 6.254.784 | 0,9239 |
| **`exact r=26`** | **6.763.776** | **0,9991** |
| `exact r=32` | 8.293.632 | 1,2251 |

**`r = 32` erra o pareamento por 22,5%** e teria falhado o gate da seção G. Como
`exact` congela `A`, dobrar o rank não dobra a parte treinável na mesma
proporção que no `vanilla`; o rank que pareia é 26, com 0,09% de diferença.

Esta decisão foi tomada **por aritmética, antes de qualquer seed** — nenhum dado
de acurácia foi consultado.

## G. Por que `alpha` acompanha o rank

`src/dual_heater/lora.py:101` define `self.scaling = lora_alpha / r`. Manter
`alpha = 16` em `r = 26` daria `scaling = 0,615` contra `1,0` do braço de
referência, introduzindo uma diferença de escala efetiva **dentro do experimento
que existe para remover um confundimento**. Com `alpha = 26` em `r = 26`, os dois
braços têm `scaling = 1,0`.

Coberto por `tests/test_lora.py::test_scaling_is_constant_when_alpha_tracks_rank`,
que também trava o modo de falha (`scaling == 0,5` quando `alpha` fica para trás).

O flag `--alpha` não existia; foi acrescentado em `experiments/qwen_lora_slowheat.py`
(commit `d528485`), **antes** deste pré-registro, por TDD.

## H. Verificação obrigatória antes da run

1. `tests/test_lora.py::test_scaling_is_constant_when_alpha_tracks_rank` passa;
2. `tests/test_lora_slowheat.py::test_alpha_is_configurable_and_defaults_to_sixteen` passa;
3. suíte de LoRA completa verde (67 testes em 28/09);
4. os parâmetros treináveis dos dois braços diferem em **menos de 5%** —
   verificado nos manifestos de uma run de fumaça **antes** das 10 seeds;
5. `E_eff = 0,8500` (tol. 1e-3) para `exact_r26` em todas as fronteiras;
6. as seeds de B2 nunca foram executadas com nenhum braço de LoRA.

## I. Análise declarada

```text
para cada seed s: d_s = forgetting(exact_r26, s) − forgetting(vanilla_r16, s)
teste: sinal exato bicaudal sobre {d_s}
confirma se: p < 0,025 E mediana(d_s) < 0
```

Sem análise intermediária, sem parada antecipada, sem inspeção seed a seed
antes das 10 terminarem.

## J. O que este protocolo não resolve

- **Expressividade do subespaço.** Igualar parâmetros treináveis **não** iguala
  expressividade: `exact_r26` tem 26 direções aleatórias fixas em `A`,
  `vanilla_r16` tem 16 direções aprendíveis. Se `exact_r26` vencer, a explicação
  "mais direções, ainda que fixas, ajudam" permanece aberta. Declarado aqui,
  antes da run, para não ser descoberto depois.
- **A premissa pode já ter resposta na literatura.** Ver a tabela K, linha de
  28/09: LoRA-FA argumenta que congelar um fator não sacrifica expressividade.
  Se esse argumento se sustentar, este experimento é confirmação empírica de
  algo já argumentado, não a remoção de um confundimento vivo.
- **Generalização.** Um modelo, um benchmark, um ponto de plasticidade.

## K. Registro de alterações

| Data | Alteração | Antes da run? |
|---|---|---|
| 28/09/2026 | criação e congelamento. B1 a B8 fechados. | sim — nenhuma seed executada |
| 28/09/2026 | **B1 alterado de `exact r=32` para `exact r=26`.** A contagem analítica de parâmetros (seção F) mostrou que `r=32` dá razão 1,2251 contra `vanilla r=16`, falhando o gate de 5% da seção H; `r=26` dá 0,9991. Decisão por aritmética, sem consultar acurácia. | sim — nenhuma seed executada |
| 28/09/2026 | **Achado de literatura registrado antes da run.** O levantamento ([../docs/related_work/protocol_prior_art.md](../docs/related_work/protocol_prior_art.md)) encontrou **LoRA-FA** (arXiv 2308.03303, 2023), que já congela `A` e treina só `B` — o mecanismo do braço `exact` — e afirma que a estrutura assimétrica do LoRA permite congelar um fator **sem sacrificar expressividade**. Consequência declarada: (a) o artigo precisa citar LoRA-FA e LoRI e posicionar `exact` como avaliação de mecanismo conhecido, não como proposta; (b) se a afirmação de LoRA-FA se sustentar sob leitura integral, o valor deste experimento cai e a execução deve ser reavaliada antes de gastar GPU. | sim — nenhuma seed executada |
