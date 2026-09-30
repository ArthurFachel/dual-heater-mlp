# Protocolo congelado — L2: `E > 1` é fenômeno de otimizador adaptativo ou aritmética?

**Congelado em:** 2026-09-29, antes da primeira seed.
**Opção L2** de `goals/protocol_penalty_reevaluation.md` §L, escolhida pelo Fachel.
**Instrumento:** `goals/protocol_plasticity_generalization.md`, primária G1'
(razão de normas).
**Antecedentes:** `docs/results/penalty_pass1_plasticity.md` (MNIST) e
`docs/results/penalty_pass1_cifar_plasticity.md` (CIFAR-100).

> **Esta é uma passada mecanismo-only. Nenhum endpoint de acurácia é lido.**
> Como as passadas 1, isso é o que permite que o resultado altere o desenho sem
> queimar pré-registro.

---

## A. A pergunta, e por que ela mudou de forma

As duas passadas 1 mediram `E > 1` para cinco das seis combinações método ×
host, e **os dois relatórios levantaram a mesma hipótese não testada**: que o
mecanismo seria o estado do otimizador adaptativo — `exp_avg_sq` do AdamW
normalizando um gradiente sistematicamente maior.

**Ao desenhar este protocolo, essa hipótese ficou provavelmente errada.** O
motivo é aritmético e não precisa de run nenhuma para ser enunciado. Ele está
no §B. A consequência é que o L2 deixa de ser "vamos ver o que o SGD dá" e passa
a ser **um teste de predição pontual declarada antes da run**, que é um
experimento muito mais forte.

## B. A derivação, feita ANTES de qualquer seed

Sob SGD puro com learning rate `η`, seja `g` o gradiente da loss e `p` o
gradiente do termo de penalidade. Então:

```
Δ_unpenalized = −η g
Δ_native      = −η (g + p)
E  =  ‖Δ_native‖ / ‖Δ_unpenalized‖  =  ‖g + p‖ / ‖g‖
```

Elevando ao quadrado:

```
E²  =  (‖g‖² + 2 g·p + ‖p‖²) / ‖g‖²
```

Portanto:

> **`E > 1`  ⟺  `2 g·p + ‖p‖² > 0`  ⟺  `g·p > −‖p‖²/2`**

### B.1 O que isso significa

`E < 1` **não é o caso genérico.** Exige que o gradiente da penalidade seja
anti-alinhado ao gradiente da loss por mais do que metade da própria norma. Uma
perturbação aleatória, independente de `g`, produz `E > 1` na maioria das vezes:
medido numericamente sobre 2.000 sorteios com `p` gaussiano independente,
**85,3% deram `E > 1`**.

Ou seja: **somar qualquer coisa a um vetor tende a aumentar sua norma.** Um
método de penalidade só reduz a norma do update se empurrar especificamente
*contra* o gradiente da loss.

Isso é verdade sob SGD e não depende de otimizador adaptativo, de arquitetura,
nem de dataset.

### B.2 A identidade que liga as duas métricas já medidas

Sob SGD vale exatamente (verificado numericamente com erro < 1e-5):

```
E · cos(Δ_native, Δ_unpenalized)  =  1 + (g·p) / ‖g‖²
```

As passadas 1 já reportam `E` e `cos` por arm. **Sob AdamW essa identidade não
vale**, porque o update não é `−η g`. Mas aplicá-la aos números medidos como
heurística sugere a escala do termo de penalidade:

| host | arm | `E` | `cos` | `g·p/‖g‖²` implícito | `‖p‖/‖g‖` implícito |
|---|---|---:|---:|---:|---:|
| MNIST | `ewc` | 0,9958 | 0,9990 | −0,0052 | 0,045 |
| MNIST | `si` | 1,0116 | 0,9927 | +0,0042 | 0,122 |
| MNIST | `mas` | 1,0103 | 0,9729 | −0,0171 | 0,234 |
| CIFAR | `ewc` | 1,0013 | 0,9968 | −0,0019 | 0,080 |
| CIFAR | `si` | 1,0124 | 0,9357 | −0,0527 | 0,361 |
| CIFAR | `mas` | 1,0194 | **0,4489** | **−0,5424** | **1,060** |

**São valores implícitos sob uma identidade que não se aplica ao otimizador que
os gerou.** Não são medições. Estão aqui porque geram as predições do §C, e
porque o último caso é gritante: se a escala estiver certa, o gradiente da
penalidade do MAS no CIFAR é **maior que o gradiente da loss** — o que
explicaria o cosseno de 0,449 e o mínimo de −0,876 sem invocar otimizador
nenhum.

## C. Predições declaradas antes da run

| # | predição | falsifica o quê |
|---|---|---|
| **C1** | Sob SGD, `E > 1` **persiste** para pelo menos `si` e `mas` nos dois hosts | a hipótese do otimizador adaptativo dos dois relatórios |
| **C2** | Sob SGD, a identidade `E·cos = 1 + g·p/‖g‖²` vale numericamente, erro < 1e-4 | o instrumento, se falhar |
| **C3** | O sinal de `E − 1` casa com o sinal de `2 g·p + ‖p‖²` em **todos** os passos medidos | a derivação do §B |
| **C4** | `‖p‖/‖g‖` do `mas` no CIFAR é > 0,5 sob SGD | a leitura do §B.2 de que a penalidade domina ali |

**O desfecho que falsificaria a derivação:** `E < 1` unânime sob SGD. Isso
mostraria que o `E > 1` é de fato criado pelo otimizador adaptativo, a hipótese
original dos relatórios sobreviveria, e o §B estaria errado sobre o regime real
(ainda que correto como álgebra).

**Os dois desfechos são publicáveis e ambos já têm consequência escrita:**

- **C1 confirmada:** a frase do artigo passa a ser *"o controle pareado é
  inconstruível por um motivo aritmético, não por um artefato de otimizador"*.
  É mais forte, porque generaliza para qualquer otimizador de primeira ordem e
  para qualquer método que **some** um termo ao gradiente.
- **C1 falsificada:** a frase passa a ser *"`E > 1` é um artefato de otimizador
  adaptativo; sob SGD o pareamento é construível"*, e a passada 2 volta a ter
  objeto — com um pré-registro novo, sobre SGD.

## D. Decisões congeladas

| # | decisão | valor | justificativa |
|---|---|---|---|
| S1 | Otimizador | `torch.optim.SGD`, sem momentum, sem weight decay | momentum reintroduz estado; weight decay soma outro termo ao update e confunde `p` |
| S2 | Arms | `vanilla`, `ewc`, `si`, `mas` — os mesmos quatro | pareamento com as passadas 1 |
| S3 | Forças | **idênticas às das passadas 1**: `ewc_lambda=200`, `si_lambda=600`, `mas_lambda=1` | mudar força junto com otimizador confundiria os dois eixos |
| S4 | Learning rate | `1e-2`, declarado aqui | o `1e-3` do AdamW não transfere para SGD; valor típico, escolhido sem consultar acurácia |
| S5 | Hosts | Split-MNIST primeiro; CIFAR-100 **apenas se** o MNIST não falsificar C1 | o MNIST é CPU e responde a pergunta central; o CIFAR custa GPU |
| S6 | Seeds | banda nova **8.000.011+**, 12 seeds, disjunta de todas as gastas | verificado por teste antes da run |
| S7 | Amostragem | exaustiva (1/1), como emendado nas passadas 1 | a série amostrada já inverteu uma classificação G8 |
| S8 | Métricas | `norm_ratio` (primária), `plasticity_ratio`, `direction_cosine`, **mais** `g·p/‖g‖²` e `‖p‖/‖g‖` por passo | os dois últimos são novos e existem para testar C2/C3/C4 |
| S9 | Endpoint | nenhum de acurácia, em nenhuma circunstância | mantém a passada mecanismo-only |
| S10 | Teste | sinal exato bilateral sobre `E − 1` por seed, por arm | mesmo teste das passadas 1 |

### D.1 Banda de seeds

```
8000011, 8025013, 8050021, 8075027, 8100037, 8125043,
8150053, 8175059, 8200063, 8225069, 8250077, 8275081
```

Disjunta de 700.001+, 4.000.003+, 6.000.003+ e 7.000.003+. A verificação é por
teste, não por inspeção visual.

## E. Instrumentação nova

`measure_shadow_step()` já devolve `norm_ratio`, `plasticity_ratio` e
`direction_cosine`. Para C2–C4 é preciso registrar também, no mesmo passo:

| campo | definição | testa |
|---|---|---|
| `penalty_alignment` | `g·p / ‖g‖²` | C2, C3 |
| `penalty_scale` | `‖p‖ / ‖g‖` | C4 |
| `predicted_above_one` | `2 g·p + ‖p‖² > 0` | C3 |

`g` é o gradiente da loss não penalizada e `p` o gradiente **apenas** do termo
de penalidade, ambos no mesmo ponto — ou seja `p = g_penalizado − g`. Nenhum
dos dois exige passo extra: o passo-sombra já computa os dois gradientes.

## F. Custo

**A medir antes de pedir autorização, não a estimar.** As passadas 1 no MNIST
custaram 11,4 min de CPU para 12 seeds a 1/1. O SGD não muda a contagem de
passos, então a ordem esperada é a mesma; medir uma seed fora da banda e
reportar antes de lançar.

O CIFAR-100 custou 3,44 h de GPU e **só será lançado se o MNIST não falsificar
C1** (S5).

## G. Portão de autorização

**Este protocolo não autoriza nada.** Depois de congelado e commitado, o
lançamento exige "pode rodar" explícito do Fachel, com o custo medido em mãos.

Lançar destacado:

```bash
setsid nohup <cmd> < /dev/null > results/<dir>/run.log 2>&1 & disown
```

e verificar `PPID=1` com `ps -eo pid,ppid,sess,cmd`. Uma run já morreu por
SIGTERM aos 34 min por não estar destacada; verificar a detenção **é** parte do
lançamento.

## H. O que este protocolo NÃO decide

- Não reabre a passada 2. Se C1 for falsificada, a passada 2 sob SGD exige seu
  próprio pré-registro, com banda de seeds nova.
- Não afirma nada sobre acurácia, esquecimento ou eficácia de nenhum método.
- Não mede o efeito de momentum. Um SGD com momentum é um terceiro regime e
  ficaria entre este e o AdamW; se for interessante, é outro protocolo.

## I. Registro de alterações

| data | alteração |
|---|---|
| 2026-09-29 | documento congelado, antes de qualquer seed. Inclui a derivação do §B, feita durante o desenho e antes de qualquer medição sob SGD. |
