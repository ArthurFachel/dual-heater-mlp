# Sweep de seleção de replay — 50 seeds

**Fonte:** `results/cache_derpp_10seeds/replay_selection_sweep/sweep_report.json`
**Status declarado no artefato:** `exploratory_not_independent_confirmation`
**Escopo:** 50 seeds × 6 learners × 4 seletores × 5 datasets = **4.500 runs de learner**
**Endpoint primário:** `final_average_accuracy`
**Multiplicidade declarada:** Holm sobre os contrastes de `loss`, `representative` e
`hybrid` dentro de cada família dataset/backbone/learner

> **Este é o maior `n` do repositório** (50 seeds) e nenhuma tabela publicada o usava.
>
> **Análise exploratória por construção.** Os dados já existiam em disco antes desta
> pergunta ser feita. Não é confirmação independente e não deve ser apresentada como tal
> — o próprio artefato declara `exploratory_not_independent_confirmation`.
>
> **Ressalva do artefato, verbatim:** *"Every ranked learner selects its own memory.
> Replay/SlowHeat+Replay and DER++/SlowHeat+DER++ contrasts are algorithm-level
> comparisons, not isolated SlowHeat effects."* Cada learner ranqueia as próprias
> imagens de treino (`ranking_ownership: each_learner_ranks_its_own_training_images`),
> então os contrastes abaixo medem **algoritmo completo contra algoritmo completo**, não
> o efeito isolado do SlowHeat.
>
> **Nota de artefato (28/09/2026):** os 4.000 checkpoints `.pt` deste diretório foram
> removidos (98,78 GB liberados). Os agregados JSON/CSV estão intactos e foram
> verificados após a remoção. Nenhuma análise de endpoint depende dos pesos.

## Método desta extração

Os `p` de Holm não estão persistidos no artefato — só a nota de multiplicidade. A tabela
abaixo reporta o que **está** nos dados: média da diferença pareada, meia-largura do IC95
normal, contagem de sinais e um teste de sinal exato bicaudal recomputado sobre as 50
diferenças pareadas. `forget` é a média da diferença pareada em `average_forgetting`
(negativo = menos esquecimento), reportada ao lado da acurácia por regra editorial.

**Os `p` de sinal abaixo são brutos, sem correção.** Com 40 contrastes na tabela, um
Holm sobre essa família mataria tudo com `p` acima de ~0,00125. Os contrastes marcados
como grandes abaixo sobrevivem com folga; os de `p` entre 0,03 e 0,12 não sobreviveriam.
Não recalculei a família declarada no artefato porque escolher família depois de ver o
resultado é exatamente o que a correção existe para impedir.

## Contrastes

```text
========================================================================================================
SLOWHEAT_VS_DERPP
dataset              seletor            media FAA      IC95    sinais   p_sinal    forget
--------------------------------------------------------------------------------------------------------
permuted_mnist       first                +0.0012    0.0003    41+/9- 5.614e-06   -0.0057
permuted_mnist       loss                 +0.0028    0.0004    49+/1- 9.059e-14   -0.0079
permuted_mnist       representative       +0.0006    0.0004   31+/19-    0.1189   -0.0051
permuted_mnist       hybrid               +0.0011    0.0004   37+/13- 0.0009362   -0.0057
split_cifar10        first                -0.0078    0.0019    6+/44- 3.244e-08   +0.0078
split_cifar10        loss                 -0.0004    0.0022   25+/25-         1   +0.0005
split_cifar10        representative       -0.0037    0.0019   17+/33-   0.03284   +0.0015
split_cifar10        hybrid               +0.0006    0.0022   29+/21-    0.3222   -0.0022
split_cifar100       first                -0.0035    0.0013   13+/37- 0.0009362   -0.0006
split_cifar100       loss                 +0.0035    0.0009    44+/6- 3.244e-08   -0.0045
split_cifar100       representative       -0.0050    0.0013    7+/43- 2.099e-07   +0.0008
split_cifar100       hybrid               +0.0011    0.0008   32+/18-   0.06491   -0.0022
split_cifar10_cnn    first                -0.0224    0.0026    0+/50- 1.776e-15   -0.0048
split_cifar10_cnn    loss                 -0.0325    0.0040    1+/49- 9.059e-14   +0.0174
split_cifar10_cnn    representative       -0.0210    0.0028    0+/50- 1.776e-15   -0.0082
split_cifar10_cnn    hybrid               -0.0135    0.0040    9+/41- 5.614e-06   -0.0146
split_mnist          first                +0.0273    0.0030    50+/0- 1.776e-15   -0.0452
split_mnist          loss                 +0.0388    0.0064    48+/2- 2.267e-12   -0.0556
split_mnist          representative       +0.0228    0.0035    48+/2- 2.267e-12   -0.0401
split_mnist          hybrid               +0.0609    0.0102    49+/1- 9.059e-14   -0.0836

========================================================================================================
SLOWHEAT_VS_REPLAY
dataset              seletor            media FAA      IC95    sinais   p_sinal    forget
--------------------------------------------------------------------------------------------------------
permuted_mnist       first                +0.0042    0.0007    47+/3- 3.708e-11   -0.0110
permuted_mnist       loss                 +0.0099    0.0028    44+/6- 3.244e-08   -0.0182
permuted_mnist       representative       +0.0024    0.0007    42+/8- 1.164e-06   -0.0088
permuted_mnist       hybrid               +0.0055    0.0018   38+/12- 0.0003059   -0.0129
split_cifar10        first                +0.0012    0.0017   28+/22-    0.4799   +0.0015
split_cifar10        loss                 +0.0015    0.0011   32+/18-   0.06491   -0.0002
split_cifar10        representative       +0.0059    0.0023   38+/12- 0.0003059   -0.0047
split_cifar10        hybrid               +0.0041    0.0016   38+/12- 0.0003059   -0.0043
split_cifar100       first                +0.0057    0.0009    48+/2- 2.267e-12   +0.0069
split_cifar100       loss                 +0.0071    0.0009    48+/2- 2.267e-12   +0.0084
split_cifar100       representative       +0.0102    0.0011    50+/0- 1.776e-15   +0.0030
split_cifar100       hybrid               +0.0073    0.0006    50+/0- 1.776e-15   +0.0084
split_cifar10_cnn    first                -0.0126    0.0022    2+/48- 2.267e-12   -0.0203
split_cifar10_cnn    loss                 -0.0196    0.0030    1+/49- 9.059e-14   -0.0042
split_cifar10_cnn    representative       -0.0199    0.0029    2+/48- 2.267e-12   -0.0107
split_cifar10_cnn    hybrid               -0.0149    0.0040    8+/42- 1.164e-06   -0.0172
split_mnist          first                +0.0008    0.0036   27+/23-    0.6718   -0.0049
split_mnist          loss                 -0.0164    0.0081   14+/36-  0.002602   +0.0163
split_mnist          representative       +0.0056    0.0031   33+/17-   0.03284   -0.0112
split_mnist          hybrid               -0.0037    0.0080   26+/24-    0.8877   +0.0002
```

## Leitura

### 1. As 50 seeds confirmam a inversão de sinal, e a tornam mais específica

`docs/architectures/arch_cnn.md` (C1) reporta, com 10 seeds, que SlowHeat acoplado a
ER-ACE melhora e acoplado a DER++ piora. As 50 seeds **replicam a direção negativa
contra DER++** onde ela aparecia, e acrescentam uma precisão que 10 seeds não davam: o
sinal **não é uma propriedade do dataset nem do método base isoladamente — ele depende
do seletor de memória**.

O caso mais claro é `split_cifar100` contra DER++, onde os quatro seletores não
concordam entre si:

| seletor | média FAA | sinais | p (sinal, bruto) |
|---|---:|---:|---:|
| `loss` | **+0,0035** | 44+/6− | 3,2e-08 |
| `hybrid` | +0,0011 | 32+/18− | 0,065 |
| `first` | −0,0035 | 13+/37− | 9,4e-04 |
| `representative` | **−0,0050** | 7+/43− | 2,1e-07 |

Duas direções opostas, ambas com `p` minúsculo e dezenas de seeds concordando, no mesmo
dataset e no mesmo método base. Trocar o seletor de `loss` para `representative` move a
diferença em 0,85 p.p. e **inverte a conclusão**. O mesmo padrão aparece em
`split_cifar10` contra DER++ (`hybrid` +0,0006, `first` −0,0078).

Isso é mais forte do que "o efeito depende do método base": **o efeito depende de uma
escolha de configuração que a maioria dos artigos não reporta**. É evidência direta para
a tese de protocolo — só que sobre um confundidor diferente do learning rate.

### 2. A magnitude não vira ruído com `n` maior; ela fica nítida

Nenhum dos contrastes grandes encolheu para dentro do IC95 ao passar de 10 para 50
seeds. Ao contrário: os IC95 são estreitos (0,0003 a 0,0102) e vários contrastes são
unânimes ou quase (`split_mnist`/`first` 50+/0−, `split_cifar10_cnn`/`first` 0+/50−,
`split_cifar100`/`representative` contra replay 50+/0−).

O caso mais forte do sweep inteiro é **`split_mnist` contra DER++**, positivo nos quatro
seletores, de +2,28 a +6,09 p.p., com 48 a 50 seeds concordando. Isso é consistente com
o `+3,18 p.p.` exploratório do host MLP (E3) que M1 pretende confirmar — mas **não é uma
confirmação dele**: seletor diferente, protocolo diferente, dados que precedem a
pergunta.

E o caso mais forte na direção oposta é **`split_cifar10_cnn`**, negativo em todos os
seletores contra os dois métodos base, de −1,26 a −3,25 p.p., com 41 a 50 seeds
concordando. No único backbone convolucional do sweep, o acoplamento **piora
consistentemente**.

### 3. Contrastes que 10 seeds não teriam detectado

Vários contrastes têm média pequena mas sinais fortemente assimétricos — o tipo de
achado que só aparece com `n` grande:

- `permuted_mnist`/`loss` contra DER++: média +0,0028 (IC95 0,0004), **49+/1−**;
- `permuted_mnist`/`first` contra DER++: +0,0012, 41+/9−;
- `split_cifar100`/`hybrid` contra replay: +0,0073, **50+/0−**.

Com 10 seeds, diferenças de 0,1 a 0,3 p.p. ficariam dentro do ruído. Com 50, a
consistência do sinal as torna detectáveis. Vale notar que **relevância prática e
significância estatística divergem aqui**: +0,12 p.p. com 41/50 seeds é estatisticamente
sólido e praticamente irrelevante. O artigo não deve citar esses contrastes como ganho.

### 4. Onde acurácia e esquecimento discordam

Pela regra editorial do projeto, os dois vão sempre juntos. Três casos em que eles
apontam para lados opostos:

| contraste | FAA | forgetting | leitura |
|---|---:|---:|---|
| `split_cifar100`/`representative` vs replay | **+0,0102** | **+0,0030** | ganha acurácia **e** esquece mais |
| `split_cifar100`/`hybrid` vs replay | +0,0073 | +0,0084 | idem, mais pronunciado |
| `split_cifar10_cnn`/`hybrid` vs DER++ | **−0,0135** | **−0,0146** | esquece bem menos **e ainda perde** acurácia |

A última linha é o exemplo canônico da regra: reduzir esquecimento em 1,46 p.p. parece
bom até se ver que vem junto com 1,35 p.p. a menos de acurácia final. Citar só
forgetting inverteria a conclusão.

## O que isto muda no artigo

1. **Há um segundo confundidor documentado, com `n` = 50.** A escolha do seletor de
   memória inverte o sinal de contrastes dentro do mesmo dataset e método base. Isso
   reforça a tese de protocolo por um caminho independente do learning rate.
2. **O material é exploratório e não vira confirmatório por reanálise.** Pode entrar
   como evidência motivacional ou como figura de limitação, nunca como resultado
   confirmado.
3. **Não recalcular Holm sobre outra família.** O artefato declara a família usada; a
   tabela acima reporta `p` brutos e diz isso explicitamente.
4. **Não citar contrastes de milésimos como ganho prático**, mesmo com 49/50 seeds
   concordando.
