# Protocolo congelado — plasticidade pareada sobre a superfície treinável

> Pré-registro. Escrito **antes** de qualquer seed deste experimento e antes de
> qualquer implementação do braço de controle. Não é editado depois de ver
> acurácia; alterações exigem commit anterior à run e linha nova na tabela K.

**Estado: CONGELADO em 29/09/2026**, antes da primeira seed.

---

## A. O defeito de medição que motiva

A confirmação de 28/09 ([`protocol_lora_confirmation.md`](protocol_lora_confirmation.md))
reportou o contraste `exact − lr_control` como **pareado em plasticidade
efetiva**, com a verificação H1 exigindo `E_eff = 0,8500 ± 1e-3` em todas as
seeds. A verificação passou. A afirmação de pareamento, mesmo assim, é falsa.

`effective_plasticity()` (`src/dual_heater/lora_slowheat.py`) calcula a média da
máscara **sobre os parâmetros que carregam um binding de máscara**. No braço
`exact` o único binding é `lora_b.weight`, porque `A` é congelada pelo builder
(`FROZEN_A_METHODS`). Os 2.555.904 parâmetros de `A` têm fator de update
exatamente zero e **não entram na média**.

Medindo sobre a superfície treinável do braço de referência
(`vanilla r=16`, 6.769.920 parâmetros):

| braço | `E` reportado (bindings) | `E` de superfície | diferença |
|---|---:|---:|---:|
| `vanilla` | 1,000 | 1,000 | — |
| `lr_control` (lr×0,85) | 1,000 | 0,850 | — |
| `frozen_a_control` | 1,000 | **0,622** | `A` congelada |
| `exact` | **0,850** | **0,532** | `A` congelada + máscara em `B` |

**O endpoint primário confirmado comparou 0,532 contra 0,850.** A diferença de
32 pontos de plasticidade removida está na direção que favorece o braço tratado.

### A.1 O lema, que independe de dados

Para qualquer modificação multiplicativa diagonal do update,

```text
E_superfície = lr_scale * (média_máscara * P_mascarado + P_não_mascarado) / P_referência
E_superfície <= E_bindings
```

com igualdade se e somente se nada fora do suporte dos bindings foi congelado ou
reescalado. Congelar é `m_i = 0` em parâmetros que a métrica não enumera, logo a
diferença é exatamente a fração congelada. Implementado em
`surface_plasticity()`, travado em `tests/test_surface_plasticity.py`.

### A.2 Por que isto não é apenas uma correção interna

A decomposição do `exact` ([`docs/results/exact_decomposition_results.md`](../docs/results/exact_decomposition_results.md))
mostrou que **todo** o efeito vem de congelar `A` (`exact − frozen_a_control`
deu 5+/5, p = 1,00) e **nenhum** da máscara. Os dois achados se encaixam: a
máscara, que é o que `E` media, não faz nada; o congelamento, que `E` não via,
faz tudo.

O confundimento de capacidade registrado em R2/§J da confirmação e a plasticidade
efetiva **não são dois problemas** — são a mesma grandeza, medida sobre escopos
diferentes. Este protocolo os trata como um só.

## B. Pergunta

Removida a **mesma quantidade média** de plasticidade da superfície treinável,
reduzir a superfície (congelar `A`, que é LoRA-FA) reduz o esquecimento mais do
que reduzir o learning rate uniformemente?

Não é "congelar ajuda?" — isso é compatível com "qualquer freio ajuda". É
"a **forma** da remoção importa, sob medição correta?".

## C. Hipótese declarada antes da run

*`frozen_a_control` reduz o forgetting médio em relação a `lr_control_062`, com
a diferença pareada por seed negativa na maioria das seeds.*

Hipótese nula: as diferenças pareadas se distribuem simetricamente em torno de
zero.

**Os dois desfechos são reportáveis e nenhum encerra o artigo:**

- **Sobrevive:** reduzir a superfície bate reduzir o lr uniformemente na mesma
  plasticidade de superfície. O resultado do host continua de pé sob pareamento
  correto, e o artigo de protocolo ganha um caso em que a correção **preserva**
  a conclusão.
- **Morre:** o único resultado positivo do host era o pareamento defeituoso. O
  artigo de protocolo ganha o caso mais forte possível — o instrumento derruba
  o resultado dos próprios autores — e a linha de mecanismo em LoRA fecha.

Nenhum desfecho autoriza seeds extras, família reduzida ou promoção de endpoint.

## D. Decisões congeladas

| # | Decisão | Valor congelado | Justificativa |
|---|---|---|---|
| P1 | Braços | `vanilla`, `lr_control_062`, `frozen_a_control` — exatamente três | `exact` sai: a decomposição mostrou que a máscara não contribui (5+/5), e mantê-lo custa 1,8× o tempo para medir um empate já medido |
| P2 | Seeds | banda nova `6_000_003, 6_025_019, 6_050_029, 6_075_041, 6_100_057, 6_125_069, 6_150_077, 6_175_091, 6_200_107, 6_225_119` | disjuntas de todas as bandas já gastas; verificado por teste |
| P3 | Endpoint primário | forgetting médio, contraste `frozen_a_control − lr_control_062` | mesmo endpoint da confirmação que este protocolo re-testa |
| P4 | Endpoints secundários | FAA (mesmo contraste); `surface_plasticity` por braço (verificação, não resultado) | reportados sempre, nunca promovidos |
| P5 | Teste | sinal exato bicaudal sobre as 10 diferenças pareadas por seed | mesmo teste da confirmação; sem suposição de normalidade em n=10 |
| P6 | Correção múltipla | Holm sobre a família de **2** (forgetting e FAA no contraste primário), α=0,05 | família declarada **antes** da run. Piso de Holm em n=10: 2×2/1024 = 0,0039 |
| P7 | Plasticidade alvo | `E_superfície = 0,622462` para os dois braços tratados | é a fração que o congelamento de `A` impõe; não é escolhida, é medida |
| P8 | `lr` do controle | `1e-4 × 0,6224617129892229 = 6,224617e-5` | derivado de 4.214.016/6.769.920, aritmética pura, sem consultar acurácia |
| P9 | Critério de sucesso | `p < 0,025` no endpoint primário **e** mediana negativa | idêntico ao A8 da confirmação e ao B8 do rank-matched |
| P10 | `vanilla` | descritivo, **fora** da família confirmatória | serve de âncora de superfície e de sanidade; não entra em Holm |

## E. Cenário

Copiado verbatim do manifest da confirmação original
(`results/qwen_lora_confirmation/seed_700001/manifest.json`), para que este
experimento meça sob as mesmas condições do resultado que re-testa.

| Item | Valor |
|---|---|
| Modelo | `Qwen/Qwen2.5-0.5B`, `Qwen2ForSequenceClassification`, 150 rótulos |
| Precisão | fp32 (Pascal CC 6.x não tem bf16 nativo) |
| Alvos LoRA | `gate_proj`, `up_proj`, `down_proj` |
| `rank` / `alpha` | 16 / 16,0 → `scaling = 1,0` nos três braços |
| Dataset | `clinc/clinc_oos:plus`, Class-IL por domínio, 10 tarefas |
| Dados por tarefa | 750 treino / 300 validação |
| Épocas | 3 |
| `batch_size` | 8 |
| `max_length` | 48 |
| `lr` base | 1e-4 |
| Hardware | uma seed inteira por GPU, sem DDP |

## F. Os três braços

| braço | `A` | máscara | lr | treináveis | `E` superfície |
|---|---|---|---|---:|---:|
| `vanilla` | treinável | não | 1e-4 | 6.769.920 | 1,000 |
| `lr_control_062` | treinável | não | **6,2246e-5** | 6.769.920 | **0,622462** |
| `frozen_a_control` | **congelada** | não | 1e-4 | 4.214.016 | **0,622462** |

Os dois braços tratados são pareados em `E` de superfície **por construção
aritmética**, não por bisseção: `0,622462` é simultaneamente a fração de
parâmetros que sobrevive ao congelamento e o fator de lr do controle.

`lr_control` a 0,85 — o executado na confirmação — **não** é este braço. O braço
existente nunca mediu o controle na plasticidade certa, porque a plasticidade
certa nunca foi calculada.

## G. Família confirmatória

| contraste | isola |
|---|---|
| `frozen_a_control − lr_control_062` (forgetting) | **primário**: a forma da remoção, na mesma quantidade |
| `frozen_a_control − lr_control_062` (FAA) | secundário da mesma família |

`frozen_a_control − vanilla` e `lr_control_062 − vanilla` são **descritivos,
fora da família**, e não recebem correção nem veredito.

## H. Verificação obrigatória antes da run

1. `tests/test_surface_plasticity.py` verde (9 testes);
2. `tests/test_plasticity_matched_runner.py` verde (11 testes);
3. teste de mutação: as seis mutações declaradas em §H.1 são todas mortas;
4. os manifestos registram `masked_mean`, `masked_parameters` e
   `learning_rate_scale` em **todos** os braços, de modo que
   `surface_plasticity` seja recomputável offline;
5. `surface_plasticity` medido difere de `0,622462` em menos de `1e-6` nos dois
   braços tratados, **em todas as seeds**;
6. `effective_learning_rate` = `6,224617e-5` para `lr_control_062` e `1e-4` para
   os outros dois, em todas as seeds;
7. tokens idênticos entre braços dentro de cada seed;
8. as seeds de P2 nunca foram executadas com nenhum braço de LoRA.

### H.1 Mutações que devem morrer

O modo de falha silenciosa deste desenho é o controle rodar **sem tratamento**:
o falsificador a `lr × 1,0` produz um contraste contra baseline não tratado, com
manifest de aparência perfeitamente normal. Seis mutações cobrem a cadeia:

| # | Mutação | Efeito se sobrevivesse |
|---|---|---|
| A | `learning_rate_scale` ignorado em `resolve_effective_lr` | controle sem tratamento |
| B | escala aplicada a todos os braços | contraste destruído |
| C | CLI não repassa o campo ao `RunConfig` | controle sem tratamento |
| D | `target_plasticity` com precedência sobre a escala | controle a 0,85, não 0,622 |
| E | `masked_parameters` fixo em 0 | pareamento não auditável |
| F | `learning_rate_scale` fixo em 1,0 | pareamento não auditável |

Verificado em 29/09: **6/6 mortas**, com baseline limpo verde antes.

## I. Análise declarada

```text
para cada seed s:
    d_s = forgetting(frozen_a_control, s) − forgetting(lr_control_062, s)
teste: sinal exato bicaudal sobre {d_s}
confirma se: p < 0,025 E mediana(d_s) < 0
```

Sem análise intermediária, sem parada antecipada, sem inspeção seed a seed antes
das 10 terminarem. A tabela de braços com FAA, forgetting e retenção absolutas é
reportada **ao lado** de cada contraste, nunca substituída por ele.

## J. O que este protocolo NÃO resolve

- **Expressividade do subespaço.** Congelar `A` remove 37,75% dos parâmetros
  treináveis *e* fixa as direções da down-projection. Um controle que removesse
  a mesma fração de parâmetros **treináveis aleatórios** separaria as duas
  coisas; não está neste desenho. Declarado antes, não descoberto depois.
- **Não decide o SlowHeat.** A máscara já foi decidida pela decomposição
  (5+/5, p = 1,00) e não está aqui.
- **Não re-mede o `exact`.** Nenhum contraste deste protocolo autoriza
  reinterpretar a confirmação de 28/09 além do que §A estabelece: ela não era
  pareada. O número medido continua sendo o que foi medido; muda a descrição
  das condições.
- **Generalização.** Um modelo, um benchmark, um `r`, um ponto de plasticidade.

## K. Registro de alterações

| Data | Alteração | Antes da run? |
|---|---|---|
| 29/09/2026 | criação e congelamento. P1 a P10 fechados. Braço `lr_control_062`, medida `surface_plasticity`, flag `--learning-rate-scale` e os 20 testes implementados por TDD **antes** deste congelamento; 6/6 mutações mortas. | sim — nenhuma seed executada |
