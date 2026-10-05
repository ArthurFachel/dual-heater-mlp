# Passada 1 no CIFAR-100 — plasticidade de métodos de penalidade

**Data:** 2026-09-29
**Protocolo:** `goals/protocol_penalty_reevaluation.md` (§C.1, passada 1)
**Instrumento:** `goals/protocol_plasticity_generalization.md`, primária G1'
(razão de normas)
**Artefatos:** `results/penalty_pass1_cifar/` (12 manifests + 3 logs de shard)
**Commit do código:** `447f969`
**Companheiro:** `docs/results/penalty_pass1_plasticity.md` (mesmo desenho, host MLP)

> **Nenhum endpoint de acurácia foi lido nesta passada.** É o que mantém o
> pré-registro válido: o `lr_scale` dos controles da passada 2 sairia destes
> números, e escolhê-los depois de ver acurácia seria seleção pós-hoc.

---

## Pergunta

A mesma da passada 1 no MNIST, no segundo host declarado no §D: quanta
plasticidade cada método de penalidade remove do update, medido pelo
passo-sombra a partir do mesmo estado?

A resposta decide, por G8, quais métodos têm `lr_control` construível e portanto
quais contrastes sobrevivem na família confirmatória.

## Desenho

| item | valor |
|---|---|
| host | Split-CIFAR100 class-IL, MLP (1024, 512) sobre vetor achatado de 3072, 10 tarefas de 10 classes |
| device | CUDA, 3 GPUs, sharding por seed |
| seeds | 12, banda 7.000.003+ (§G), as **mesmas** do MNIST |
| arms | `vanilla`, `ewc`, `si`, `mas` |
| forças | `ewc_lambda=200`, `si_lambda=600`, `mas_lambda=1` |
| amostragem | **exaustiva**, 1/1 — 2.880 passos com âncora por arm por seed |
| custo | 3,44 h de GPU somadas; 50,8 / 67,2 / 70,1 min por shard |

**As forças são idênticas às do MNIST.** Isto importa para a interpretação e é
fácil de ler errado: `ewc_lambda = 200` corresponde a `k_ewc = 100` de Hsu et
al., porque nosso termo é `0.5 · ewc_lambda · Σ Ω (θ−θ*)²` enquanto Hsu soma
`k · Σ Ω (θ−θ*)²`. Ver `experiments/penalty_reevaluation.py`. Portanto **toda
diferença entre este relatório e o do MNIST é efeito de host, não de força.**

## Resultado

Métrica primária G1' (razão de normas), média por seed e agregada sobre as 12:

| arm | `E` (norma) | dp entre seeds | faixa | `cos` médio | sinais | `p` exato | pareável (G8) |
|---|---:|---:|---|---:|---|---:|---|
| `ewc` | **1,0013** | 0,0001 | [1,0011, 1,0014] | 0,9968 | 12+/0− | 0,00049 | **não** |
| `si` | **1,0124** | 0,0002 | [1,0121, 1,0128] | 0,9357 | 12+/0− | 0,00049 | **não** |
| `mas` | **1,0194** | 0,0036 | [1,0150, 1,0268] | 0,4489 | 12+/0− | 0,00049 | **não** |

Diagnósticas, agregadas sobre as 12 seeds:

| arm | D1 razão por elemento | `cos` mínimo por passo | `E` por passo, faixa |
|---|---:|---:|---|
| `ewc` | 1,3919 | 0,8979 | [0,9452, 1,0692] |
| `si` | 3,2512 | 0,0884 | [0,4625, 1,9397] |
| `mas` | 6,1850 | **−0,8758** | [0,6610, 1,9469] |

## Interpretação

**Os três métodos aumentam a norma do update. G8 dispara para todos.** A
unanimidade é perfeita — 12 de 12 seeds acima de 1 nos três arms — e o desvio
entre seeds é uma a duas ordens de grandeza menor que o efeito.

**EWC inverte de sinal entre os hosts.** No MNIST mediu `E = 0,9958` (abaixo de
1, pareável); aqui mede `E = 1,0013` (acima, não pareável). Como as forças são
idênticas, a diferença é do host — 10 tarefas de 10 classes em
vez de 5 de 2, e escala de gradiente diferente. O efeito é pequeno (0,13%) mas
tem dp entre seeds de 0,0001, ou seja treze vezes menor que o efeito.

**MAS gira o update, não só o escala.** O cosseno médio de 0,449 e o mínimo de
**−0,8758** dizem que em alguns passos o update penalizado aponta na direção
aproximadamente oposta ao update não penalizado. Isso é qualitativamente
diferente do MNIST, onde `cos = 0,9729` e a leitura "escala para cima" bastava.
Aqui o número que descreve o MAS não é um escalar de plasticidade: é uma
rotação. **Nenhuma noção de "mesma plasticidade por learning rate reduzido" se
aplica a um update que mudou de direção** — G8 já o excluía pela norma, e a
diagnóstica D2 mostra que a exclusão é ainda mais justificada do que a norma
sozinha indicaria.

A divergência entre a primária (razão de normas, ~1,02) e a diagnóstica D1
(razão por elemento, 6,19 no MAS) é exatamente o motivo da emenda G1' de 29/09:
a razão por elemento é dominada por elementos cujo update não penalizado é
quase zero, onde a razão explode. A norma não tem esse defeito. As duas são
reportadas, e a primária é a que foi declarada antes.

## Consequência para o protocolo

**A família confirmatória do §F fica vazia.**

| # | contraste | host | estado |
|---|---|---|---|
| ~~1~~ | ~~`ewc − lr_control_ewc`~~ | Split-MNIST | vale (E = 0,9958 < 1) |
| ~~2~~ | ~~`ewc − lr_control_ewc`~~ | Split-CIFAR100 | **removido: E = 1,0013 > 1** |
| ~~3–4~~ | ~~`si` − controle~~ | ambos | removidos na passada 1 anterior |
| ~~5–6~~ | ~~`mas` − controle~~ | ambos | removidos na passada 1 anterior |

Sobra **uma** comparação construível em todo o piloto: `ewc − lr_control_ewc`
em Split-MNIST, com `lr_scale = 0,9958` — uma redução de learning rate de
**0,42%**. Vale registrar o que isso significa na prática: um controle que
remove 0,42% da plasticidade é indistinguível do `vanilla` para qualquer
propósito, e um contraste contra ele mede essencialmente `ewc − vanilla`.

**Isto não é seleção pós-hoc.** A regra G8 foi escrita no §C antes de qualquer
seed, com a consequência declarada, e a decisão usou somente a plasticidade
medida. Nenhuma acurácia foi consultada em nenhum dos dois hosts.

**A passada 2, como desenhada, perdeu o objeto.** Ela existe para rodar os
`lr_control` pareados; cinco dos seis não são construíveis e o sexto é uma
redução de 0,42%. Executá-la gastaria GPU para produzir um controle vazio. A
decisão sobre o que fazer no lugar é do Fachel e está aberta abaixo.

## Limites

1. **Dois hosts, uma família de otimizador, uma arquitetura.** MNIST e CIFAR-100,
   ambos AdamW e **ambos MLP** — o loader entrega vetores achatados e
   `generalization_configs()['split_cifar100'].backbone == 'mlp'`. O §D do
   protocolo diz "(CNN)" e está errado; nenhuma convolução foi treinada. O §D do
   protocolo do instrumento já declara que sob otimizador
   adaptativo o contrafactual é local. **`E > 1` pode ser um fenômeno de
   otimizador adaptativo, e isso não foi testado.** Um arm com SGD puro é o
   teste direto e continua não feito.
2. **Uma configuração de força por método.** São as publicadas por Hsu et al.
   para class-IL. Não sabemos se `E > 1` persiste em outras forças. Um sweep de
   λ × `E` é barato e responderia.
3. **O mecanismo não foi isolado.** A hipótese do estado do otimizador
   (`exp_avg_sq` normalizando um gradiente sistematicamente maior) continua
   plausível e não testada nos dois hosts.
4. **`E` é média sobre passos com âncora.** O primeiro passo após cada
   consolidação tem drift zero e portanto `E = 1` exato, o que atenua a média
   em direção a 1. No MNIST a correção foi medida e é desprezível; aqui não foi
   recomputada, e como todos os efeitos são de mesmo sinal a atenuação só pode
   ter **subestimado** `E`, nunca produzido um `E > 1` espúrio.

## O que isto significa para o artigo

O resultado do CIFAR fortalece a frase do MNIST em vez de complicá-la, porque
estende o fenômeno a um segundo benchmark — **não** a uma segunda arquitetura,
já que os dois hosts rodam MLP:

> *Métodos de consolidação por penalidade não removem plasticidade de forma
> monotônica sob otimizadores adaptativos. Em Split-MNIST e Split-CIFAR100, ambos
> com MLP e AdamW, EWC, SI e MAS aumentam a norma do update em relação ao
> treino não penalizado — unânime em 12 seeds por host. Para o MAS no CIFAR-100
> o update chega a inverter de direção (cosseno mínimo −0,88), de modo que
> nenhum escalar de plasticidade o descreve. Um controle pareado em plasticidade,
> no sentido de learning rate reduzido, é portanto inconstruível para cinco das
> seis combinações método × host testadas.*

Isso é mais forte que o plano original previa. A Seção 4 do ICML deixa de ser
"re-avaliamos métodos publicados sob pareamento correto" e passa a ser "o
pareamento correto **não é construível** para esta família de métodos, e o
instrumento mostra por quê" — que é um resultado de instrumentação, exatamente o
eixo do artigo.

O que **não** sustenta: nada sobre acurácia, esquecimento ou eficácia. Nenhum
desses números foi medido.

## Registro de alterações

| data | alteração |
|---|---|
| 2026-09-29 | resultado da passada 1 no CIFAR-100, 12 seeds, medição exaustiva, 3,44 h de GPU |
| 2026-10-05 | **Correção factual: o host é MLP, não CNN.** Cinco ocorrências de "CNN" descreviam uma arquitetura que não foi treinada — `generalization_configs()['split_cifar100'].backbone == 'mlp'`, sobre vetor achatado de 3072. O §D do protocolo e o "Fora de escopo" do roadmap já registravam isso; este relatório não. **Nenhum número muda**: as medições são do que de fato rodou, só a descrição estava errada. A frase destinada ao artigo foi enfraquecida de "segunda arquitetura" para "segundo benchmark". |
