# L3 — existe um λ com poder, e ele é de um só método

**Data:** 2026-10-05
**Protocolo:** `goals/protocol_lambda_sweep.md`, congelado em `4a28768`,
**antes** da primeira seed da banda 9.000.011+.
**Instrumento:** `src/dual_heater/plasticity.py`, primária G1' (razão de normas).
**Artefatos:** `results/lambda_sweep/` — 50 manifests, agregado em
`aggregate.json`.
**Agregador:** `scripts/analyze_lambda_sweep.py` (15 testes, 7 mutações mortas).
**Commits:** protocolo `4a28768`, agregador `3ac7d5a`.
**Antecedente direto:** `docs/results/sgd_plasticity_results.md` (L2).

> **Nenhum endpoint de acurácia foi lido.** O runner não referencia
> `accuracy_matrix`, `final_average_accuracy`, `average_forgetting` nem
> `backward_transfer`, verificado por
> `tests/test_lambda_sweep.py::test_runner_reads_no_accuracy_field`.

---

## Pergunta

O L2 reabriu a passada 2: sob SGD puro os três métodos de penalidade passam no
G8. Mas passar no G8 não é ter poder. Nas forças publicadas por Hsu et al., os
`E` medidos foram 0,955 / 0,976 / 0,993 — um `lr_control` a `lr × 0,993` é
indistinguível do vanilla, e foi exatamente esse argumento que suspendeu a
passada 2 no §L.

> **Existe uma força de penalidade λ para a qual `E` fica longe o bastante de 1
> para que o controle pareado seja um controle de verdade — e o método continua
> numericamente são nessa força?**

O limiar de "longe o bastante" (`E ≤ 0,90`) foi declarado no §D.2 antes das
seeds, e a grade do eixo veio de uma calibração `n=1` fora da banda, também
anterior ao congelamento.

## Desenho

| item | valor |
|---|---|
| host | Split-MNIST class-IL, MLP (256, 128) |
| otimizador | SGD puro, sem momentum, sem weight decay (T1) |
| learning rate | `3e-3`, **fixo** (T2) — o ponto são do eixo do L2 |
| eixo de λ | multiplicador sobre Hsu et al.: `1, 3, 10, 30, 100` (T3) |
| seeds | 10, banda 9.000.011+ (T5), disjunta de todas as gastas |
| arms | `vanilla`, `ewc`, `si`, `mas` |
| amostragem | exaustiva (1/1): 128 passos com âncora por arm por seed |
| custo | **84,2 min de CPU**, 50 células, rodadas em 5 processos paralelos (~17 min de relógio) |

## Resultado

### A métrica primária, por célula

| arm | λ | `E` | dp entre seeds | seeds divergentes | poder (§D.2) |
|---|---:|---:|---:|---:|---|
| `ewc` | 1× | 0,9517 | 0,0032 | 0/10 | 0/10 |
| `ewc` | 3× | 0,9124 | 0,0053 | 0/10 | 0/10 |
| `ewc` | 10× | — | — | **10/10** | 0/10 |
| `ewc` | 30× | — | — | **10/10** | 0/10 |
| `ewc` | 100× | — | — | **10/10** | 0/10 |
| `si` | 1× | 0,9730 | 0,0024 | 0/10 | 0/10 |
| `si` | 3× | 0,9561 | 0,0036 | 0/10 | 0/10 |
| `si` | 10× | *2,98e17* | — | **9/10** | 0/10 |
| `si` | 30× | — | — | **10/10** | 0/10 |
| `si` | 100× | — | — | **10/10** | 0/10 |
| `mas` | 1× | 0,9906 | 0,0018 | 0/10 | 0/10 |
| `mas` | 3× | 0,9741 | 0,0045 | 0/10 | 0/10 |
| `mas` | 10× | 0,9303 | 0,0100 | 0/10 | 0/10 |
| `mas` | **30×** | **0,8541** | 0,0161 | 0/10 | **10/10** |
| `mas` | **100×** | **0,7279** | 0,0214 | 0/10 | **10/10** |

### Veredito das quatro predições

| # | predição | veredito | evidência |
|---|---|---|---|
| **T-P1** | `E` monótono decrescente em λ | **CONFIRMADA** | 0 inversões nos três arms. O `mas` percorre os 5 pontos sem pular nenhum; `ewc` e `si` entregam 2 pontos cada, com 3 pulados por divergência |
| **T-P2** | `ewc` e `si` divergem em λ ≥ 10× | **CONFIRMADA** | 10/10 seeds no `ewc` nos três pontos; 9/10 no `si` a 10× e 10/10 a 30× e 100× |
| **T-P3** | `mas` atinge `E ≤ 0,90` sem divergir | **CONFIRMADA** | 0,8541 a 30× e 0,7279 a 100×, **unânime em 10/10 seeds**, zero amostras não-finitas |
| **T-P4** | nenhuma célula de `ewc`/`si` tem poder | **CONFIRMADA** | 0/10 em todas as 10 células; as únicas sãs (1× e 3×) ficam acima do limiar |

### O ponto escolhido

Pelo §E.2 o agregador seleciona o **menor** λ com poder unânime:

> **`mas` a 30× as forças publicadas, `E = 0,8541`**, dp entre seeds 0,0161,
> cosseno médio 0,9645 (mínimo 0,9522), `‖p‖/‖g‖` = 0,2517.

O 100× dá um `E` mais baixo (0,7279), mas também afasta mais o ponto de operação
das forças publicadas, e o cosseno já cai para 0,9005 (mínimo 0,8766). A 30× o
update ainda é essencialmente um escalonamento; a 100× ele começa a girar.

## Interpretação

**Existe um λ com poder, e a resposta é mais estreita do que "sim".** De 15
células, **duas** têm poder, e as duas são do mesmo método. Isso não é um
detalhe de execução — é a estrutura do resultado:

**Os três métodos têm fronteiras de estabilidade incompatíveis com a faixa onde
o controle seria construível.** `ewc` e `si` precisariam de λ ≥ 10× para
aproximar `E = 0,90`, e é exatamente aí que explodem: a 3× ainda estão em 0,912
e 0,956, acima do limiar, e a 10× não produzem uma única seed sã. **Não existe
janela entre "fraco demais para controlar" e "instável demais para medir"** — os
dois regimes são adjacentes, sem sobreposição.

O `mas` é a exceção, e por um motivo mecanístico visível nos dados: seu
`‖p‖/‖g‖` cresce devagar (0,014 em 1×, 0,252 em 30×, 0,429 em 100×), enquanto
nos outros dois o termo de penalidade ultrapassa o gradiente da loss e leva o
otimizador junto. O `mas_lambda = 1` de Hsu está muito abaixo da fronteira de
estabilidade do método; o `si_lambda = 600`, muito perto dela.

**A monotonicidade do T-P1 confirma a leitura do L2.** Se `E` cai de forma
regular conforme λ cresce, em três ordens de grandeza, então o gradiente da
penalidade é sistematicamente anti-alinhado ao da loss e λ modula essa relação
de forma previsível. O `mas` dá os 5 pontos sem uma inversão.

## Consequência para o protocolo de re-avaliação

**A passada 2 fica autorizada a existir — mas só para um arm, num ponto de
operação que ninguém publicou.** O §E.1 previu as duas saídas e esta é a
confirmada:

| o que estava em jogo | antes do L3 | depois do L3 |
|---|---|---|
| arms pareáveis | 3 (por G8, sob SGD) | **1** (`mas`) |
| λ utilizável | o publicado | **30× o publicado** |
| contraste do controle | 0,7% a 4,5% | **14,6%** |

### As ressalvas que vão junto, obrigatoriamente

1. **Mudar λ muda o método (§E.2).** Um MAS a 30× a força publicada não é "o MAS
   de Hsu com mais poder de controle" — é outro ponto de operação, e **este
   protocolo não mede acurácia e portanto não pode dizer o quanto ele é pior**.
   A passada 2 vai medir a acurácia de um método que ninguém propôs.
2. **Isso enfraquece a generalidade, não a fortalece.** Um resultado da passada 2
   falaria sobre `mas` a 30× em Split-MNIST sob SGD. Nenhuma das três condições
   é a configuração em que o método foi publicado.
3. **Um host, um otimizador.** O CIFAR-100 sob SGD não está autorizado (S5 do
   L2), e o AdamW já foi medido e não tem controle construível para nenhum arm.

## Limites

1. **Um host.** Split-MNIST com MLP.
2. **Um learning rate.** `3e-3` fixo. A interação λ × lr não foi medida; o L2
   mostrou que `E` depende dos dois, logo a grade de poder provavelmente se move
   com o lr. Não é caro de medir e continua não medido.
3. **Grade logarítmica grossa.** Entre 10× e 30× o `mas` cruza o limiar em algum
   ponto que este desenho não localiza. Para a passada 2 isso não importa — 30×
   já tem poder — mas significa que "30×" não é o λ mínimo com poder, apenas o
   menor **da grade**.
4. **`E` é média sobre passos com âncora.** O primeiro passo após cada
   consolidação tem drift zero e `E = 1` exato, o que atenua a média em direção
   a 1. Como todos os efeitos são `E < 1`, a atenuação só pode ter subestimado a
   magnitude.
5. **Divergência de execução em relação ao §G.** O protocolo pede `setsid` com
   verificação de `PPID = 1`. A política de segurança do ambiente bloqueia
   `setsid`/`nohup` e exige o gerenciador de processos do agente; as runs foram
   lançadas com `persist_on_release`. O efeito é equivalente, a letra do
   protocolo não. O progresso foi verificado por contagem de artefatos.
6. **O eixo foi rodado em 5 processos paralelos**, um por ponto. Cada célula é
   independente e single-thread (`OMP_NUM_THREADS=1`), então o paralelismo não
   altera nenhum resultado — só o relógio (84,2 min de CPU em ~17 min de
   relógio). Verificado: a seed 9.000.011 a 1× deu `ewc = 0,9564` tanto na run
   serial abortada quanto na paralela.

## O que isto significa para o artigo

O resultado afia a Seção 4 do ICML num ponto específico:

> *Sob SGD, os três métodos de penalidade reduzem a norma do update, e a redução
> cresce monotonicamente com a força da penalidade. Mas a faixa de forças em que
> a redução é grande o bastante para construir um controle pareado útil é
> disjunta da faixa em que o método permanece numericamente estável, para dois
> dos três métodos testados. EWC e SI passam de "fraco demais para controlar"
> (E > 0,91 a 3×) para "divergente" (10/10 seeds a 10×) sem janela utilizável
> entre os dois regimes. Apenas o MAS admite um controle pareado, e apenas a 30
> vezes a força publicada — um ponto de operação que a literatura não propõe e
> cuja acurácia este desenho não mede.*

Isso é mais específico do que "o pareamento é inconstruível": diz **por que** é
inconstruível, e mostra que a exceção existe mas custa caro.

**O que NÃO sustenta:** nada sobre acurácia, esquecimento ou eficácia. Nenhum
desses números foi medido, aqui, no L2 ou nas passadas 1.

## Nota de método: as quatro predições foram declaradas antes

As quatro foram confirmadas, o que é um desfecho mais fraco em termos de
informação do que uma falsificação — um protocolo que acerta tudo aprendeu menos
do que um que erra algo. Vale registrar que o §E.1 declarava a saída oposta
(T-P3 falsificada) como **a mais forte das duas para um artigo de
instrumentação**, e que ela não ocorreu. O resultado foi o menos interessante
dos dois possíveis, e está reportado como tal.

A calibração do §C (`n=1`, seed 8.999.977, fora da banda) já apontava para esta
direção, e informou a grade. Isso está no pré-registro, não foi acrescentado
depois.

## Registro de alterações

| data | alteração |
|---|---|
| 2026-10-05 | resultado da run L3, 10 seeds × 5 pontos do eixo, 84,2 min de CPU, medição exaustiva. T-P1 a T-P4 todas confirmadas. Passada 2 autorizada a existir para `mas` a 30×, com as ressalvas do §E.2. |
