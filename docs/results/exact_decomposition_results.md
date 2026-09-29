# Decomposição do braço `exact` — resultado

**Pré-registro:** `goals/protocol_exact_decomposition.md` (congelado em
`f5d77ce`, antes de qualquer seed).
**Execução:** 28-29/09/2026, 10 seeds, 220,7 min, GTX 1080 Ti.
**Artefatos:** `results/exact_decomposition/`.

---

## Pergunta

O braço `exact` faz **duas** coisas ao mesmo tempo:

1. congela a down-projection `A` — isto é **LoRA-FA** (arXiv 2308.03303),
   técnica publicada, reusada por **LoRA-Null** (arXiv 2503.02659) com a mesma
   motivação de preservação de conhecimento que a nossa;
2. mascara linhas de `B` com um tracker SlowHeat — isto é nosso.

O contraste confirmado `exact − lr_control` (forgetting −0,0921, 9−/1+,
p = 0,02148) media a **soma**. Nenhum braço separava as parcelas.

Quanto do ganho é LoRA-FA e quanto é SlowHeat?

## Desenho

Três braços, 10 seeds pareadas, hiperparâmetros copiados verbatim do manifest
da confirmação original (`results/qwen_lora_confirmation/seed_700001`) para
medir sob as mesmas condições do resultado que se decompõe.

| braço | `A` | máscara em `B` | treináveis |
|---|---|---|---:|
| `vanilla` | treinável | não | 6.769.920 |
| `frozen_a_control` | **congelada** | **não** | 4.214.016 |
| `exact` | congelada | sim | 4.214.016 |

`exact` e `frozen_a_control` têm **a mesma contagem de parâmetros**: o
contraste de interesse é pareado em capacidade por construção.

## Resultado

Endpoint primário **forgetting**, sinal exato bilateral, Holm sobre família
de 2:

| contraste | média | mediana | sinais | p | Holm | |
|---|---:|---:|---:|---:|---:|---|
| `frozen_a_control − vanilla` | **−0,1118** | −0,1304 | **1+/9−** | 0,02148 | **0,0430** | ✓ |
| `exact − frozen_a_control` | −0,0092 | +0,0020 | 5+/5 | 1,000 | 1,000 | ✗ |

FAA (secundário, exploratório): `frozen_a_control − vanilla` +0,0720 (8+/2−,
p = 0,109); `exact − frozen_a_control` +0,0069 (5+/5, p = 1,000).

**Desfecho (a) do pré-registro: todo o ganho do `exact` vem de congelar `A`.
A camada SlowHeat não adiciona nada — 5+/5 é empate perfeito.**

## A coincidência numérica

`frozen_a_control − vanilla` dá p = 0,02148 com 9 sinais negativos. A
confirmação original de `exact − lr_control` deu **p = 0,02148 com 9 sinais
negativos**. O efeito que atribuíamos ao SlowHeat é, número a número, o efeito
de LoRA-FA.

## Custo

`exact` roda a ~611 s contra ~336 s de `frozen_a_control`: **1,8× mais lento**
para não entregar diferença mensurável.

## Consequência

O `exact` não sustenta reivindicação de contribuição. O que ele faz de efetivo
é LoRA-FA, publicado em 2023. A leitura honesta para o artigo é:

> Replicamos LoRA-FA e verificamos que a proteção seletiva adicional não
> melhora retenção neste host, sob capacidade pareada e dez seeds.

Isto é resultado negativo pré-registrado e **será reportado**.

## Consistência com o resto da evidência

Três resultados independentes do mesmo período apontam na mesma direção:

1. `lr_control − vanilla` não replicou: reduzir plasticidade uniformemente
   empata com mascarar seletivamente;
2. a ablação de critério (10 seeds) mostrou que `|z·dL/dz|` empata com `|z|`
   (5+/5) — *o que* se mede não importa;
3. esta decomposição: a máscara em si não importa.

O padrão é consistente: **em LoRA, restringir o adaptador não ajuda.** O que
ajuda é reduzir a superfície treinável, que é o que LoRA-FA faz.

Contraste com o BERT, onde proteção seletiva bate máscara aleatória em
+12,23 pp (10/10 seeds). A diferença provável é geométrica: em LoRA a matriz
`A` é compartilhada por todas as saídas através do gargalo `z`, então proteção
por unidade de saída vaza.

## Limites

- Um host (Qwen2.5-0.5B), um benchmark (Split-CLINC150), `r = 16`.
- Não implementamos a correção de gradiente em forma fechada do LoRA-FA
  (`g_B = (r/α)²(AᵀA)⁻¹ g_B`); nosso `frozen_a_control` é LoRA-FA **sem** ela.
- `frozen_a_control − vanilla` compara capacidades diferentes (4,2M vs 6,8M
  treináveis). É inerente ao LoRA-FA e foi registrado em §D5 como algo a
  reportar, não a corrigir aumentando o rank.
- Não decide o SlowHeat fora de LoRA.

## Nota de método

Este braço custava um `requires_grad_(False)`. O resultado era descobrível
desde o início; ele não apareceu porque ninguém havia separado as duas
operações que o `exact` acumula. A seed de calibração isolada sugeriu
`frozen_a_control > exact` de forma dramática (−0,14 em forgetting), leitura
que **não sobreviveu** à segunda seed — registro deixado aqui como evidência
concreta de por que uma seed não conclui.
