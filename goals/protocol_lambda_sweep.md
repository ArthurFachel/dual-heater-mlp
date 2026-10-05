# Protocolo congelado — L3: existe λ onde o pareamento tem poder?

**Congelado em:** 2026-10-05, antes da primeira seed da banda.
**Opção L3** de `goals/protocol_penalty_reevaluation.md` §L.
**Item 2.12** de `goals/roadmap_icml_ijcnn.md`.
**Instrumento:** `goals/protocol_plasticity_generalization.md`, primária G1'
(razão de normas).
**Antecedente direto:** `docs/results/sgd_plasticity_results.md` (L2).

> **Passada mecanismo-only. Nenhum endpoint de acurácia é lido**, em nenhuma
> circunstância (S9 do L2, reafirmado aqui como T9). É isso que permite que o
> resultado redesenhe a passada 2 sem queimar pré-registro.

---

## A. A pergunta

O L2 mostrou que, sob SGD puro, os três métodos de penalidade reduzem a norma do
update: `E` = 0,955 (`ewc`), 0,976 (`si`), 0,993 (`mas`) em `lr = 3e-3`, unânime
em 12 seeds. Isso reabriu a passada 2 — os três passam no G8.

**Mas reabrir não é o mesmo que ter poder.** Um `lr_control` a `lr × 0,993` é
operacionalmente indistinguível do vanilla, e foi exatamente esse argumento
(`E_ewc ≈ 0,996` sob AdamW) que suspendeu a passada 2 no §L. Trocar 0,996 por
0,993 não resolve nada.

A pergunta do L3 é portanto:

> **Existe uma força de penalidade λ para a qual `E` fica longe o bastante de 1
> para que um controle pareado seja um controle de verdade — e o método continua
> numericamente são nessa força?**

As duas metades importam. Um λ que produz `E = 0,5` mas faz o otimizador
divergir não serve, porque o `E` medido num arm que explodiu não mede
plasticidade.

## B. Por que esta pergunta precede a passada 2

A passada 2 lê acurácia. Gastar seeds confirmatórias num contraste cujo
tratamento é uma redução de 0,7% no learning rate produziria um nulo
não-informativo: não se saberia se o método não tem efeito ou se o controle não
tinha força para revelá-lo.

O L3 é barato (CPU, minutos), mecanismo-only, e responde isso antes. Se a
resposta for "não existe λ com poder", **a passada 2 morre definitivamente** e
isso é um resultado publicável sobre o instrumento, não um fracasso.

## C. A calibração que informou a grade

Rodada **antes** deste congelamento, em `scripts/calibrate_lambda_sweep.py`,
seed `8.999.977` (fora de toda banda), `lr = 3e-3`, SGD puro, 8,7 min de CPU,
`n=1`. Multiplicador sobre as forças de Hsu et al. (2018):

| mult | `ewc` | `si` | `mas` |
|---:|---|---|---|
| 1 | 0,9606 | 0,9777 | 0,9928 |
| 3 | 0,9262 | 0,9614 | 0,9796 |
| 10 | **divergiu** (29 não-fin.) | **divergiu** (58) | 0,9428 |
| 30 | **divergiu** (65) | **divergiu** (65) | 0,8738 |
| 100 | **divergiu** (111) | **divergiu** (92) | 0,7507 |
| 300 | **divergiu** (115) | **divergiu** (101) | **divergiu** |

**Isto é `n=1` e não tem valor inferencial.** Está aqui porque a grade do §D foi
escolhida com estes números à vista, e omitir isso tornaria a escolha menos
auditável, não mais. Nenhuma acurácia foi lida ao produzi-los.

### C.1 O que a calibração já sugere

1. **Os três arms têm fronteiras de estabilidade muito diferentes.** `ewc` e
   `si` divergem entre 3× e 10×; `mas` sobrevive até 100× e só quebra em 300×.
   Isso não é propriedade da "força" em abstrato — os λ de Hsu já estão em
   escalas distintas por método (`mas_lambda = 1` contra `si_lambda = 600`),
   e o multiplicador preserva essa diferença de propósito (T3).
2. **Só o `mas` oferece a faixa útil.** Entre 10× e 100× ele percorre `E` de
   0,943 a 0,751 sem uma única amostra não-finita, com cosseno ≥ 0,92.
3. **A predição que isso gera está no §E**, declarada antes das seeds.

## D. Decisões congeladas

| # | decisão | valor | justificativa |
|---|---|---|---|
| T1 | Otimizador | `torch.optim.SGD`, puro (sem momentum, sem weight decay) | herdado do L2/S1; o AdamW já foi medido e produz `E > 1` |
| T2 | Learning rate | **`3e-3`, fixo** | o ponto são do eixo do L2: em `1e-2` o `si` diverge, em `1e-3` os `E` colam em 1. Fixar o lr é o que torna λ o único eixo |
| T3 | Eixo de λ | **multiplicador** sobre as forças de Hsu: `1, 3, 10, 30, 100` | multiplicador, não valor absoluto, preserva a razão entre métodos que Hsu publicou. `300` foi **excluído** porque a calibração mostrou os três arms divergindo lá |
| T4 | Arms | `vanilla`, `ewc`, `si`, `mas` | os mesmos quatro, por pareamento com L2 e com as passadas 1 |
| T5 | Seeds | banda nova **9.000.011+**, 10 seeds, disjunta de todas as gastas | verificado por teste, não por inspeção |
| T6 | Amostragem | exaustiva (1/1) | a série amostrada já inverteu uma classificação G8 uma vez |
| T7 | Métricas | `norm_ratio` (primária), `direction_cosine`, `penalty_scale`, `penalty_alignment`, contagem de não-finitos | as mesmas do L2; nenhuma métrica nova |
| T8 | Divergência | arm com **qualquer** amostra não-finita é marcado `divergente` e **excluído dos julgamentos T1–T4 do §E**, nunca apagado | um `E` de 1e19 não mede plasticidade, mede um otimizador explodindo. A contagem vai no manifest |
| T9 | Endpoint | nenhum de acurácia, em nenhuma circunstância | mantém a passada mecanismo-only |
| T10 | Teste | sinal exato bilateral sobre `E − 1` por seed, por célula (arm × λ) | o mesmo das passadas 1 e do L2 |

### D.1 Banda de seeds

```
9000011, 9025013, 9050033, 9075041, 9100051,
9125059, 9150067, 9175073, 9200089, 9225091
```

Disjunta de 700.001+, 4.000.003+, 6.000.003+, 7.000.003+ e 8.000.011+.
**Dez** seeds, não doze: o piso de significância do sinal exato com n=10 é
`2^-9 = 0,00195`, e a família do §E tem no máximo 5 comparações por arm, cujo
limiar de Holm mais estrito é `0,05/5 = 0,01`. Com 10 seeds unânimes o `p` é
0,00195 < 0,01, logo o desenho limpa a própria barra. Doze seeds custariam 20% a
mais para ganhar margem que não muda nenhum veredito.

### D.2 O limiar de "poder"

Declarado **antes** das seeds, porque escolhê-lo depois seria escolher o
resultado:

> Uma célula (arm, λ) tem **poder** se `E ≤ 0,90` e o arm não é divergente ali.

A justificativa do 0,90: um `lr_control` a `lr × 0,90` é uma redução de 10% no
learning rate, que é a menor diferença que o projeto já tratou como
operacionalmente real em outro contexto — o `lr_control_062` do
`protocol_plasticity_matched.md` usou 0,622. Dez por cento é uma barra
deliberadamente generosa; se nem ela for atingida, o veredito é claro.

## E. Predições declaradas antes da run

| # | predição | falsifica o quê |
|---|---|---|
| **T-P1** | `E` é **monótono decrescente** em λ dentro de cada arm não-divergente | a leitura de que λ modula a anti-relação entre `g` e `p` |
| **T-P2** | `ewc` e `si` **divergem** em λ ≥ 10× em pelo menos metade das seeds | a fronteira de estabilidade da calibração, se ela for artefato de `n=1` |
| **T-P3** | O `mas` atinge `E ≤ 0,90` (§D.2) em λ = 30× ou 100×, sem divergir | que existe pelo menos uma célula com poder |
| **T-P4** | **Nenhuma** célula de `ewc` ou `si` tem poder sem divergir | que os três métodos são igualmente pareáveis |

### E.1 As duas saídas, com consequência escrita antes

- **T-P3 confirmada:** a passada 2 fica autorizada a existir — **só para o
  `mas`, só no λ identificado**, e com pré-registro próprio (item 2.13). A frase
  do artigo passa a ser *"o pareamento é construível, mas apenas fora das forças
  publicadas, e apenas para um dos três métodos"*. Isso é um resultado sobre o
  instrumento e sobre o custo de usá-lo, não uma vitória do método.
- **T-P3 falsificada:** **a passada 2 morre, e o item 2.13 fecha sem rodar.** A
  frase passa a ser *"não existe força de penalidade que torne o controle
  pareado simultaneamente poderoso e numericamente são; a inconstrutibilidade é
  estrutural, não um acidente das forças publicadas"*. Para um artigo de
  instrumentação **esta é a saída mais forte das duas**, e dizer isso antes de
  gastar as seeds é o oposto de escolher o resultado.

### E.2 A ressalva que vale nos dois casos

Mudar λ muda o método. Um EWC a 100× a força publicada não é "o EWC de Hsu com
mais poder de controle" — é outro ponto de operação, provavelmente pior em
acurácia, e **este protocolo não mede acurácia e portanto não pode dizer o
quanto**. Qualquer célula com poder encontrada aqui carrega essa ressalva para a
passada 2 e para o artigo, obrigatoriamente.

## F. Custo

**Medido, não estimado.** A calibração rodou 6 pontos × 1 seed em 8,7 min de
CPU, ou ~87 s por célula-seed. A run declarada é 5 pontos × 10 seeds = 50
células-seed, logo **~72 min de CPU**, single-thread, sem GPU.

Como é CPU e está dentro da faixa pré-aprovada do projeto para testes leves, não
exige o portão de autorização de GPU — mas o lançamento é reportado ao Fachel
antes de começar.

## G. O que este protocolo NÃO decide

- **Não autoriza a passada 2.** Ela exige o seu próprio pré-registro, mesmo que
  T-P3 seja confirmada, e a escolha do λ tem de estar declarada lá.
- **Não mede acurácia, esquecimento ou eficácia.** Nada aqui sustenta afirmação
  sobre qualidade de método.
- **Não estende ao CIFAR-100.** Um host, Split-MNIST. O L2 já estabeleceu que o
  S5 do protocolo dele não autoriza o CIFAR sob SGD.
- **Não mede momentum.** Terceiro regime, outro protocolo.

## H. Registro de alterações

| data | alteração |
|---|---|
| 2026-10-05 | documento congelado, antes de qualquer seed da banda 9.000.011+. A calibração do §C (seed 8.999.977, fora da banda, `n=1`, mecanismo-only) é anterior ao congelamento e informou a grade do T3 e o limiar do §D.2. |
