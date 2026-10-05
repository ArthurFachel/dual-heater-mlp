# Protocolo congelado — Passada 2: o MAS sobrevive a um controle pareado?

**Congelado em:** 2026-10-05, antes da primeira seed da banda 9.500.011+.
**Item 2.13** de `goals/roadmap_icml_ijcnn.md`. Autorizado pelo §E.1 de
`goals/protocol_lambda_sweep.md` (L3), confirmado em `ac00084`.
**Substitui**, para efeito de execução, o §C.1 passo 3 de
`goals/protocol_penalty_reevaluation.md`, que foi escrito para AdamW e cujo
desenho de 6 comparações foi eliminado por G8.

> ⚠️ **ESTA PASSADA LÊ ACURÁCIA.** É a primeira da Fase 2 que o faz. Todas as
> anteriores (passadas 1, L2, L3) foram mecanismo-only, e era isso que permitia
> redesenhar sem queimar pré-registro. **A partir daqui não é mais o caso:**
> este documento é o pré-registro, e o resultado que ele produzir é o resultado,
> seja ele qual for.

---

## A. A pergunta, e o quanto ela encolheu

O desenho original tinha 6 comparações, 3 métodos × 2 hosts. Sobrou **uma**:

| etapa | o que restou |
|---|---|
| desenho original (§F da re-avaliação) | 6 comparações |
| após passada 1 (AdamW, G8) | 1, com `lr_scale = 0,9958` — vazia |
| após L2 (SGD) | 3 pareáveis, mas `lr_scale` entre 0,955 e 0,993 — fracas |
| após L3 (sweep de λ) | **1 com poder: `mas` a 30×, `lr_scale = 0,854`** |

> **O MAS, a 30 vezes a força publicada e sob SGD puro, reduz a norma do update
> em 14,6%. Essa redução compra alguma coisa em esquecimento, além do que a
> mera perda de plasticidade já explicaria?**

## B. O que esta passada NÃO pode responder

Declarado antes das seeds porque é a limitação central e ela não some depois:

1. **Não fala sobre o MAS publicado.** `mas_lambda = 30` não é o `mas_lambda = 1`
   de Hsu et al. É outro ponto de operação, escolhido pelo L3 por ser o menor da
   grade com poder. Nenhum resultado aqui transfere para a configuração
   publicada.
2. **Não fala sobre EWC nem SI.** O L3 mostrou que para esses dois não existe λ
   onde o controle seja simultaneamente útil e estável. Eles não estão nesta
   passada, e sua ausência é um resultado do L3, não uma omissão desta.
3. **Não fala sobre AdamW.** Sob AdamW nenhum controle é construível (G8).
4. **Um host.** Split-MNIST, MLP (256, 128), class-IL.

## C. Arms

| arm | o que é | `E` |
|---|---|---|
| `vanilla` | fine-tuning sequencial, sem penalidade | 1,0 por construção |
| `mas` | MAS a `mas_lambda = 30` (30× Hsu) | 0,8541 (medido no L3) |
| `lr_control` | **sem penalidade**, `lr × 0,8541056558955461` | 0,8541 por construção |

O `lr_control` é o ponto inteiro do desenho: ele remove exatamente a mesma
quantidade de plasticidade que o MAS remove, mas **sem nenhum mecanismo de
consolidação**. Se o MAS não bater esse controle, o que ele faz é indistinguível
de treinar mais devagar.

### C.1 O escalar congelado

```
lr_scale = 0.8541056558955461
```

Média das 10 seeds do `mas` a 30× em `results/lambda_sweep/mult_30/`, lida do
agregado do L3 (`aggregate.json`), **antes de qualquer acurácia desta passada**.
Mínimo entre seeds 0,8286, máximo 0,8795, dp 0,0161.

**É um escalar único, não um por seed.** Um `lr_scale` por seed faria o controle
perseguir o ruído do próprio método e tornaria o contraste não interpretável: o
tratamento mudaria de seed para seed. O custo é que o pareamento é exato na
média e aproximado em cada seed, e isso está no §G como limite.

**Este número não pode ser recalculado depois de ver acurácia.** Está pinado em
`tests/test_penalty_pass2.py`.

## D. Decisões congeladas

| # | decisão | valor | justificativa |
|---|---|---|---|
| P1 | Otimizador | SGD puro, sem momentum, sem weight decay | o único regime onde o controle existe (L2) |
| P2 | Learning rate base | `3e-3` | o mesmo do L3; mudá-lo moveria o `E` e invalidaria o escalar |
| P3 | Força do MAS | `mas_lambda = 30,0` (30× Hsu) | o menor λ com poder unânime (L3, §E.2) |
| P4 | `lr_scale` do controle | `0,8541056558955461`, congelado no §C.1 | medido no L3, antes desta passada |
| P5 | Arms | `vanilla`, `mas`, `lr_control` | três, não sete: os outros não têm controle construível |
| P6 | Seeds | banda nova **9.500.011+**, 12 seeds, disjunta de todas | 12 e não 10: aqui o endpoint é ruidoso (acurácia), ao contrário do `E` |
| P7 | Endpoint primário | **esquecimento médio** (`average_forgetting`), diferença pareada por seed | o mesmo do §F da re-avaliação; não muda |
| P8 | Contraste primário | **`mas − lr_control`** | o §F original; `mas − vanilla` é secundário e não entra na família |
| P9 | Família | **1 comparação** | sem Holm: com `m=1` a correção é a identidade. Declarado aqui para que ninguém a aplique depois para afrouxar |
| P10 | Teste | sinal exato bilateral sobre as 12 diferenças pareadas | o mesmo das passadas anteriores |
| P11 | Limiar | `p < 0,05` | com n=12 o piso é `0,00049`, folga de duas ordens |
| P12 | Secundários | `mas − vanilla`, `lr_control − vanilla`, acurácia média final dos três arms | reportados sempre, fora da família |

### D.1 Banda de seeds

```
9500011, 9525013, 9550037, 9575041, 9600053,
9625061, 9650069, 9675077, 9700081, 9725089,
9750097, 9775103
```

Disjunta de 700.001+, 4.000.003+, 6.000.003+, 7.000.003+, 8.000.011+ e
9.000.011+ (a do L3). Verificado por teste, não por inspeção.

### D.2 A acurácia média final é obrigatória na tabela

O §F da re-avaliação já exigia isso e a razão continua valendo: **um método pode
reduzir esquecimento colapsando a aquisição**. Se o `mas` a 30× simplesmente
aprende menos, seu esquecimento cai sem que nada tenha sido preservado. A
acurácia final vai na tabela principal ao lado do esquecimento, sempre, mesmo
que o primário dê nulo.

## E. Predições declaradas antes da run

| # | predição | o que falsifica |
|---|---|---|
| **Q1** | `mas − lr_control` em esquecimento **não é significativo** (`p >= 0,05`) | que o MAS faz algo além de remover plasticidade |
| **Q2** | `lr_control − vanilla` **reduz** esquecimento | que a redução de lr sozinha já compra retenção — se falso, o controle não é um controle |
| **Q3** | `mas` a 30× tem acurácia média final **menor** que `vanilla` | que a força inflada não custa aquisição |

**Q1 é declarada como nula de propósito.** Os três resultados negativos já
estabelecidos pelo projeto (R-B, R-C, R-F) e o próprio L3 apontam nessa direção,
e dizer isso antes de gastar as seeds é o oposto de escolher o resultado.

### E.1 As duas saídas, com consequência escrita antes

- **Q1 falsificada (o MAS bate o controle):** o método faz algo que a perda de
  plasticidade não explica. A frase do artigo passa a ser *"sob pareamento
  correto, um dos três métodos sobrevive — mas apenas a 30× a força publicada,
  num regime que a literatura não usa"*. Seria o primeiro resultado positivo da
  Fase 2, e exigiria replicação antes de qualquer afirmação forte.
- **Q1 confirmada (empate):** *"quando comparado contra um controle que remove a
  mesma plasticidade sem nenhum mecanismo de consolidação, o efeito do MAS
  desaparece — e essa comparação só foi construível fora das forças
  publicadas"*. Fecha a Seção 4 do ICML com o mesmo eixo das outras: a comparação
  que a literatura faz não é a comparação que o instrumento diz que deveria ser
  feita.

**Os dois desfechos são publicáveis e nenhum muda o plano do artigo.**

## F. Custo

**Medido, não estimado.** Cada célula-seed do L3 custou ~100 s de CPU com 4 arms
e passo-sombra ativo a 1/1. Esta passada tem 3 arms e **não precisa do
passo-sombra** (o `E` já foi medido; medi-lo de novo custaria 3 passos por
passo). Estimativa conservadora: ~75 s por seed, **~15 min de CPU para as 12
seeds**, rodáveis em paralelo.

CPU, sem GPU, dentro da faixa pré-aprovada do projeto.

## G. Limites, declarados antes

1. **O pareamento é exato na média, aproximado por seed.** O `lr_scale` é um
   escalar único (§C.1); o `E` do MAS varia entre 0,829 e 0,879 entre seeds. Uma
   seed cujo `E` real foi 0,829 recebe um controle a 0,854, ligeiramente menos
   restritivo. O efeito é simétrico em torno da média e não tem sinal preferido.
2. **O `E` foi medido com `plasticity_sampling_interval=1` sobre 128 passos com
   âncora por seed, no L3** — não nesta passada. Se o regime de treino diferir
   entre as duas, o escalar descreve outra coisa. Os hiperparâmetros são os
   mesmos por construção (P1–P3), e isso é verificado por teste.
3. **Um host, um λ, um otimizador, um lr.** Nada aqui generaliza sem replicação.
4. **12 seeds detectam efeitos grandes.** Com sinal exato e n=12, um efeito que
   aparece em 9 de 12 seeds dá `p = 0,146` e seria reportado como nulo. Esta
   passada não distingue "sem efeito" de "efeito pequeno".

## H. O que este protocolo NÃO autoriza

- **Não autoriza o CIFAR-100.** Um host.
- **Não autoriza outros λ.** 30×, o escolhido pelo L3. Rodar 100× também seria
  duas comparações e exigiria Holm, com a família declarada antes — não depois.
- **Não autoriza reler o `lr_scale`.** Ele está congelado no §C.1 e pinado em
  teste. Recalculá-lo após ver acurácia invalidaria a passada inteira.
- **Não reabre EWC nem SI.** O L3 fechou isso.

## I. Registro de alterações

| data | alteração |
|---|---|
| 2026-10-05 | documento congelado, antes de qualquer seed da banda 9.500.011+. O `lr_scale` do §C.1 vem do agregado do L3, lido antes de qualquer acurácia desta passada. |
