# Passada 1 — plasticidade de métodos de penalidade sob pareamento

**Data:** 2026-09-29
**Protocolo:** `goals/protocol_penalty_reevaluation.md` (§C.1, passada 1)
**Instrumento:** `goals/protocol_plasticity_generalization.md`, primária G1'
**Artefatos:** `results/penalty_pass1/`
**Commit do código:** `e90efd1`

> **Nenhum endpoint de acurácia foi lido nesta passada.** É o que mantém o
> pré-registro válido: o `lr_scale` dos controles da passada 2 sairia destes
> números, e escolhê-los depois de ver acurácia seria seleção pós-hoc.

---

## Pergunta

Quanta plasticidade cada método de penalidade remove do update, medido pelo
passo-sombra (`‖Δ_native‖ / ‖Δ_unpenalized‖`) a partir do mesmo estado?

A resposta define o `lr_scale` do `lr_control` pareado de cada método. Um
método com `E > 1` não tem controle construível (G8): `lr × E` seria um
**aumento** de learning rate, não uma remoção de plasticidade.

## Desenho

| item | valor |
|---|---|
| host | Split-MNIST class-IL, MLP (256, 128), AdamW `lr=1e-3` |
| seeds | 12, banda 7.000.003+ (§G), disjunta de todas as já gastas |
| arms | `vanilla`, `ewc`, `si`, `mas` |
| forças | Hsu et al. 2018 class-IL: `k_ewc=100`, `k_si=600`, `k_mas=1` |
| amostragem | **todos** os 128 passos com âncora por arm por seed (§E emendado) |
| custo | 11,4 min, CPU |

## Resultado

| arm | `E` (norma) | dp entre seeds | faixa | `cos` | pareável (G8) |
|---|---:|---:|---|---:|---|
| `ewc` | **0,9958** | 0,0007 | [0,9944, 0,9973] | 0,9990 | **sim** |
| `si` | **1,0116** | 0,0022 | [1,0090, 1,0156] | 0,9927 | **não** |
| `mas` | **1,0103** | 0,0020 | [1,0069, 1,0130] | 0,9729 | **não** |

**Unanimidade perfeita, teste de sinal exato bilateral:**

| arm | sinais | `p` exato |
|---|---|---:|
| `ewc` | 0+ / 12− (todas abaixo de 1) | 0,00049 |
| `si` | 12+ / 0− (todas acima de 1) | 0,00049 |
| `mas` | 12+ / 0− (todas acima de 1) | 0,00049 |

## Interpretação

**SI e MAS aumentam a magnitude do update em vez de reduzi-la.** O efeito é
pequeno (~1%) mas perfeitamente consistente: 12 de 12 seeds, com dp entre
seeds de 0,002 — uma ordem de grandeza menor que o efeito.

O cosseno alto (0,97–0,999) elimina a leitura alternativa: a penalidade **não**
está girando o update para uma direção nova. Ela o escala, e o escala **para
cima**. Um método cujo propósito declarado é restringir o movimento dos
parâmetros está, sob AdamW, movendo-os mais do que o treino não penalizado.

O mecanismo plausível é o estado do otimizador: o termo quadrático
`λ Ω (θ − θ*)` entra no gradiente, e `exp_avg_sq` do AdamW normaliza o passo
pela magnitude histórica do gradiente. Um gradiente sistematicamente maior não
produz um passo proporcionalmente maior — mas a interação entre o termo de
penalidade e os momentos não é uma contração garantida. **Esta explicação é
uma hipótese não testada** e está listada como trabalho futuro; o dado aqui é
a medição, não o mecanismo.

EWC fica logo abaixo de 1 (0,9958), também unânime. A diferença entre EWC e os
outros dois não é de força — `k_ewc = 100` contra `k_si = 600` — e sim de como
o Fisher empírico distribui a importância comparado a Ω do SI e do MAS. Também
não testado.

## Consequência para o protocolo

**A regra G8 dispara para SI e MAS.** A família confirmatória do §F cai de 6
para 2 comparações:

| # | contraste | host | estado |
|---|---|---|---|
| 1 | `ewc − lr_control_ewc` | Split-MNIST | vale |
| 2 | `ewc − lr_control_ewc` | Split-CIFAR100 | vale, pendente da passada 1 do CIFAR |
| ~~3–6~~ | ~~`si`/`mas` − controle~~ | — | **removidos: controle não construível** |

O piso de significância com `n = 12` e `m = 2` é `2 × 2 / 2^12 = 0,00098`,
ainda muito abaixo de 0,05 — o desenho reduzido continua capaz de limpar a
própria barra.

**A redução da família não é seleção pós-hoc:** a regra G8 foi escrita no §C
antes das seeds, com a consequência declarada, e a decisão usou apenas a
plasticidade medida. Nenhuma acurácia foi consultada.

## Limites

1. **Um host.** Tudo aqui é Split-MNIST com MLP e AdamW. O §D declara que sob
   AdamW o contrafactual é local; `E > 1` pode ser específico de otimizador
   adaptativo. Split-CIFAR100 ainda não foi medido, e SGD não foi testado.
2. **Uma configuração de força.** Os λ são os publicados por Hsu et al. para
   este benchmark. Não sabemos se `E > 1` persiste em outras forças; um
   sweep de λ × `E` responderia, e é barato (~1 min por ponto).
3. **O mecanismo não foi isolado.** A hipótese do estado do otimizador é
   plausível e não testada. Um arm com SGD puro seria o teste direto.
4. **`E` é a média sobre os passos com âncora.** O primeiro passo após cada
   consolidação tem `E = 1` exatamente (drift zero ⇒ gradiente da penalidade
   zero), o que puxa a média para 1. Medido: excluir esses 4 passos de 128
   move `E_si` de 1,0117 para 1,0121 e `E_mas` de 1,0124 para 1,0128 — a
   atenuação existe mas é desprezível, e não muda nenhum veredito.

## O que isto significa para o artigo

Este é um resultado de instrumentação, que é o eixo do ICML. A frase que ele
sustenta:

> *Métodos de consolidação por penalidade não removem plasticidade de forma
> monotônica sob otimizadores adaptativos. Em Split-MNIST com AdamW, SI e MAS
> aumentam a norma do update em relação ao treino não penalizado, de forma
> unânime em 12 seeds. Um controle pareado em plasticidade, no sentido de
> learning rate reduzido, é portanto inconstruível para esses métodos — e a
> literatura que os compara contra fine-tuning sequencial está comparando
> contra um baseline que remove MENOS plasticidade, não mais.*

O que ele **não** sustenta: nenhuma afirmação sobre acurácia, esquecimento ou
eficácia dos métodos. Nada disso foi medido.

## Registro de alterações

| data | alteração |
|---|---|
| 2026-09-29 | resultado da passada 1, 12 seeds, medição exaustiva |
