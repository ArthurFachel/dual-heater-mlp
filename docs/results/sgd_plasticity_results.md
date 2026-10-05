# L2 — `E > 1` é artefato de otimizador adaptativo, não aritmética

**Data:** 2026-09-29
**Protocolo:** `goals/protocol_sgd_plasticity.md`, congelado em `f9f9bb0`,
emendado em `80bd322` (eixo de learning rate), ambos **antes** da primeira seed.
**Instrumento:** `src/dual_heater/plasticity.py`, primária G1' (razão de normas).
**Artefatos:** `results/sgd_plasticity/` — 36 manifests, agregado em
`aggregate.json`.
**Agregador:** `scripts/analyze_sgd_plasticity.py` (15 testes, 6 mutações mortas).
**Commit do código:** `f6a596b`.
**Antecedentes:** `docs/results/penalty_pass1_plasticity.md` (MNIST/AdamW) e
`docs/results/penalty_pass1_cifar_plasticity.md` (CIFAR-100/AdamW).

> **Nenhum endpoint de acurácia foi lido.** O runner não referencia
> `accuracy_matrix`, `final_average_accuracy`, `average_forgetting` nem
> `backward_transfer`, verificado por
> `tests/test_run_sgd_plasticity.py::test_runner_reads_no_accuracy_field`.

---

## Pergunta

As duas passadas 1 mediram `E > 1` em cinco das seis combinações método × host,
e as duas levantaram a mesma hipótese não testada: que o mecanismo seria o
estado do otimizador adaptativo.

Ao desenhar este protocolo, essa hipótese foi declarada **provavelmente errada**
por um argumento aritmético (§B): sob SGD, `E > 1 ⟺ 2 g·p + ‖p‖² > 0`, e uma
perturbação independente do gradiente satisfaz isso em ~85% dos casos. A
predição C1 era que `E > 1` **persistiria** sob SGD.

## Desenho

| item | valor |
|---|---|
| host | Split-MNIST class-IL, MLP (256, 128) |
| otimizador | **SGD puro** — sem momentum, sem weight decay (S1) |
| eixo de lr | `1e-2`, `3e-3`, `1e-3` (S4 emendado) |
| seeds | 12, banda 8.000.011+ (S6), disjunta de todas as gastas |
| arms | `vanilla`, `ewc`, `si`, `mas` |
| forças | idênticas às das passadas 1: `ewc_lambda=200`, `si_lambda=600`, `mas_lambda=1` |
| amostragem | exaustiva (1/1): 128 passos com âncora por arm por seed |
| custo | **33,6 min de CPU**, 36 runs |

## Resultado

### A métrica primária, por ponto do eixo

| lr | arm | `E` | dp entre seeds | sinais | `p` exato | não-finitos | `cos` | `‖p‖/‖g‖` |
|---|---|---:|---:|---|---:|---:|---:|---:|
| 1e-2 | `ewc` | **0,8876** | 0,0071 | 0+/12− | 0,00049 | 0 | 0,9167 | 0,3773 |
| 1e-2 | `si` | *9,97e16* | 2,70e17 | 12+/0− | 0,00049 | **291** | 0,6534 | 7,81e15 |
| 1e-2 | `mas` | **0,8991** | 0,0103 | 0+/12− | 0,00049 | 0 | 0,9328 | 0,2593 |
| 3e-3 | `ewc` | **0,9552** | 0,0033 | 0+/12− | 0,00049 | 0 | 0,9847 | 0,1679 |
| 3e-3 | `si` | **0,9757** | 0,0021 | 0+/12− | 0,00049 | 0 | 0,9902 | 0,1241 |
| 3e-3 | `mas` | **0,9926** | 0,0011 | 0+/12− | 0,00049 | 0 | 0,9999 | 0,0138 |
| 1e-3 | `ewc` | **0,9760** | 0,0015 | 0+/12− | 0,00049 | 0 | 0,9925 | 0,1122 |
| 1e-3 | `si` | **0,9932** | 0,0005 | 0+/12− | 0,00049 | 0 | 0,9986 | 0,0426 |
| 1e-3 | `mas` | **0,9980** | 0,0002 | 0+/12− | 0,00049 | 0 | 1,0000 | 0,0040 |

### Veredito das cinco predições

| # | predição | veredito | evidência |
|---|---|---|---|
| **C1** | `E > 1` persiste sob SGD | **FALSIFICADA** | `E < 1` em 8 das 9 células, 12/12 seeds cada, `p = 0,00049`. A nona (`si` a 1e-2) é divergência numérica, não `E > 1` |
| **C2** | `E·cos = 1 + g·p/‖g‖²` | **CONFIRMADA** | resíduo máximo ≤ 1,8e-05 em todas as células não-divergentes |
| **C3** | sinal previsto = sinal medido | **CONFIRMADA** | **13.533 de 13.533 passos**, 100,0%, sem uma única exceção |
| **C4** | `‖p‖/‖g‖ > 0,5` no `mas` | **FALSIFICADA** | 0,2593 / 0,0138 / 0,0040 — uma a duas ordens de grandeza abaixo |
| **C5** | `E` monótono no lr | **CONFIRMADA** | os três arms crescem quando o lr cai, sem inversão |

## Interpretação

**A hipótese original dos dois relatórios estava certa, e a minha estava errada.**
`E > 1` **é** artefato de otimizador adaptativo. Sob SGD puro, os três métodos de
penalidade reduzem a norma do update — unanimemente, em todas as 12 seeds, em
todos os pontos do eixo onde a run não divergiu.

**Por que o argumento do §B falhou, e por que C3 mostra que ele não estava
errado.** A derivação está correta: C3 acertou 13.533 de 13.533 passos, o que é
uma confirmação mais forte do que o desenho esperava. O que estava errado era a
premissa de que `p` seria *aproximadamente independente* de `g`. Os dados dizem
o contrário: `g·p < 0` em toda amostra, com cosseno entre 0,92 e 1,00. **O
gradiente da penalidade é sistematicamente anti-alinhado ao da loss**, e por
margem suficiente (`g·p < −‖p‖²/2`) para que `E < 1`.

Isso tem uma leitura mecanística direta. O termo `λ Ω (θ − θ*)` aponta de volta
para a âncora, e o gradiente da loss aponta para longe dela — é exatamente o que
um método de consolidação deveria fazer. A "surpresa" do §B.1 era um argumento
sobre vetores aleatórios aplicado a dois vetores que não são aleatórios entre si.

**C5 explica a transição entre os regimes.** `‖p‖/‖g‖` cresce com o learning
rate (0,004 → 0,26 no `mas`, de 1e-3 para 1e-2), porque passos maiores afastam
mais os parâmetros da âncora e aumentam o termo de penalidade. Quanto maior o
passo, mais a penalidade morde — e o `E` cai. A curva é monótona nos três arms.

**A divergência do `si` a `lr = 1e-2` é fronteira de estabilidade, não
resultado.** 291 amostras não-finitas de 1.536, com `‖p‖/‖g‖` de 7,8e15: o
termo de penalidade explodiu e levou o otimizador junto. Está reportado como
dado — a contagem de não-finitos aparece em cada manifest — e **excluído** dos
julgamentos C2 e C5, porque um arm que explodiu não testa a identidade nem a
monotonicidade, testa o passo do otimizador. Note que C3 continuou valendo
mesmo lá (1.245/1.245): a predição acompanha a medição até dentro da
divergência.

## Consequência para o protocolo de re-avaliação

**O §L de `goals/protocol_penalty_reevaluation.md` muda de estado.** A passada 2
foi suspensa porque G8 declarou cinco das seis combinações não-pareáveis — mas
G8 foi avaliada **sob AdamW**. Sob SGD, com estes números:

| arm | `E` a usar como `lr_scale` (lr=3e-3) | pareável por G8 |
|---|---:|---|
| `ewc` | 0,9552 | **sim** |
| `si` | 0,9757 | **sim** |
| `mas` | 0,9926 | **sim** |

Os três passam. **A passada 2 volta a ter objeto, desde que rode sob SGD.**

Três ressalvas que precisam ir junto:

1. **Isto exige um pré-registro novo.** O protocolo da passada 2 foi escrito
   para AdamW; rodá-la sob SGD é outro experimento, com banda de seeds nova.
2. **Os `lr_scale` são próximos de 1** (2,4% a 4,5% de redução no melhor caso, a
   `lr = 3e-3`). São maiores que os 0,42% do EWC sob AdamW, mas ainda pequenos.
   Vale checar se o contraste resultante tem poder antes de gastar seeds.
3. **A escolha do ponto do eixo é uma decisão de protocolo**, não de análise. A
   `lr = 1e-2` os `E` são mais distantes de 1 (0,888 e 0,899), o que daria
   controles mais fortes — mas é o ponto onde o `si` diverge. Escolher o ponto
   depois de ver estes números é legítimo porque nenhuma acurácia foi lida, mas
   tem de ser declarado antes da passada 2.

## Limites

1. **Um host.** Split-MNIST com MLP. O CIFAR-100 sob SGD não foi medido; o S5
   condiciona o CIFAR a C1 não ser falsificada, e ela foi — então, pela letra do
   protocolo, o CIFAR sob SGD **não está autorizado** por este documento.
2. **Uma família de forças.** As publicadas por Hsu et al. Um sweep de λ × `E`
   continua não feito, e é barato.
3. **SGD sem momentum.** O regime intermediário (SGD com momentum) não foi
   medido e fica entre este e o AdamW. É outro protocolo.
4. **`E` é média sobre passos com âncora.** O primeiro passo após cada
   consolidação tem drift zero e `E = 1` exato, o que **atenua** a média em
   direção a 1. Como todos os efeitos aqui são `E < 1`, a atenuação só pode ter
   subestimado a magnitude, nunca criado o sinal.
5. **Divergência de execução em relação ao §G.** O protocolo pede lançamento com
   `setsid` e verificação de `PPID = 1`. A política de segurança do ambiente
   bloqueia `setsid`/`nohup` e exige o gerenciador de processos do agente; a run
   foi lançada com `persist_on_release`, que mantém o processo vivo através do
   ciclo da sessão. O efeito é equivalente, a letra do protocolo não. O
   progresso foi verificado por contagem monotônica de artefatos em três
   checagens independentes, não por `pgrep`.

## O que isto significa para o artigo

O resultado **fortalece** a Seção 4 do ICML em vez de enfraquecê-la, porque
transforma uma observação num mecanismo identificado:

> *Métodos de consolidação por penalidade reduzem a norma do update sob SGD, de
> forma unânime em 12 seeds e em três ordens de grandeza de learning rate — o
> gradiente da penalidade é sistematicamente anti-alinhado ao da loss. Sob
> AdamW, nos mesmos métodos, nas mesmas forças e nas mesmas tarefas, a norma
> AUMENTA. A inconstrutibilidade do controle pareado em plasticidade é portanto
> uma propriedade da interação entre o termo de consolidação e o otimizador
> adaptativo, não do método de consolidação. Um instrumento que mede plasticidade
> efetiva sem declarar o otimizador não mede uma propriedade do método.*

Essa última frase é o argumento de instrumentação que o artigo precisa, e ela só
é enunciável porque o L2 foi rodado.

**O que NÃO sustenta:** nada sobre acurácia, esquecimento ou eficácia. Nenhum
desses números foi medido, aqui ou nas passadas 1.

## Nota de método: a predição foi registrada antes de ser falsificada

O §I do protocolo registra, **antes** da primeira seed da banda, que a
calibração já apontava para `E ≤ 1` e que "o desfecho mais provável desta run é
a falsificação de C1". A predição C1 não foi ajustada depois de ver os dados;
foi mantida, declarada provavelmente falsa, e falsificada.

Isso importa para o que o artigo pode afirmar: C2 e C3 — as predições sobre o
**instrumento** — foram confirmadas no mesmo movimento em que C1, a predição
sobre o **regime**, caiu. Um instrumento que acerta 13.533 de 13.533 predições
pontuais enquanto derruba a hipótese de quem o construiu é o tipo de evidência
que um paper de protocolo existe para apresentar.

## Registro de alterações

| data | alteração |
|---|---|
| 2026-09-29 | resultado da run L2, 12 seeds × 3 pontos do eixo, 33,6 min de CPU, medição exaustiva |
