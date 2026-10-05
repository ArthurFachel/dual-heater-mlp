# Compêndio de resultados — acurácia e esquecimento, por run

**Gerado em:** 2026-10-05, a partir dos artefatos em `results/`.
**Escopo:** todas as runs versionadas em que um braço SlowHeat aparece, com
acurácia e esquecimento lidos dos agregados, não transcritos de outros
documentos.

---

## ⚠️ Como ler isto

**Linhas de tabelas diferentes não são comparáveis entre si.** Cada bloco é um
host, um benchmark, um protocolo e uma banda de seeds. O mesmo nome de método
em dois blocos é o mesmo código rodando em condições diferentes.

**Confirmatório vs exploratório.** Cada bloco carrega o `status` que o próprio
artefato declara, **não o rótulo do protocolo**. Isso importa: duas runs que eu
inicialmente classifiquei como confirmatórias (Hard vs Soft, seletor de replay)
têm protocolo congelado mas se declaram `exploratory_reanalysis` e
`exploratory_not_independent_confirmation` nos JSON. Onde protocolo e artefato
divergem, **o artefato manda**. Exploratório aqui não é aviso de rodapé:
significa que o número não foi protegido contra escolha pós-hoc de endpoint,
braço ou corte.

Pelo critério do artefato, as runs com rótulo confirmatório são: a confirmação
Split-MNIST (lock `frozen_before_execution`, `sha256 = 015b3162...`, idêntico
nas duas execuções), a ablação de critério em BERT, o pareamento por superfície,
a passada 2 de penalidade e `qwen_lora_confirmation`
(`status = confirmatory_multi_seed`).

**`average_forgetting` exclui a última tarefa** (não há tarefa posterior sobre a
qual esquecer). `BWT = -average_forgetting` em todas as runs onde não há
transferência positiva; onde difere, está anotado.

**Procedência.** As duas runs de confirmação do Split-MNIST registram **árvore
Git suja** e portanto não estabelecem reprodução bit-a-bit.

---

## Índice

| # | run | host | status | seeds |
|---|---|---|---|---|
| 1 | Confirmação Split-MNIST | MLP | **confirmatório** | 20 (×2 execuções) |
| 2 | Baselines Split-MNIST | MLP | exploratório | 20 e 100 |
| 3 | SlowHeat+DER++ | MLP | exploratório | 20 |
| 4 | Split-CIFAR-10/100 MLP | MLP | exploratório | 10 |
| 5 | Split-CIFAR-10 CNN/VGG11/ResNet18 | CNN | exploratório | 5–10 |
| 6 | Hard vs Soft (5 alvos) | MLP+CNN | ⚠️ ver bloco | 10 |
| 7 | Seletor de replay (R-D) | MLP | ⚠️ ver bloco | 20 |
| 8 | BERT diagnóstico (6 condições) | BERT | exploratório | 10 |
| 9 | BERT ablação de critério (R-A/R-B) | BERT | **confirmatório** | 10 |
| 10 | BERT sequência longa `T=5` | BERT | exploratório | 1 |
| 11 | Qwen+LoRA confirmação | Qwen2.5-0.5B | **confirmatório** | 10 |
| 12 | Qwen+LoRA iso-plasticidade | Qwen2.5-0.5B | exploratório | 10 |
| 13 | Decomposição do `exact` (R-C) | Qwen2.5-0.5B | exploratório | 10 |
| 14 | Pareamento por superfície (R-E) | Qwen2.5-0.5B | **confirmatório** | 10 |
| 15 | Penalidade: passadas 1, L2, L3, 2 | MLP | mecanismo + confirmatório | 10–12 |

---

## 1. Confirmação Split-MNIST — **confirmatório**

**Pré-registro:** `status = frozen_before_execution`, `preregistered_at =
2026-08-15`, mesmo `sha256` nas duas execuções.
**Endpoint primário:** acurácia média final após a quinta tarefa.
**Artefatos:** `results/protocol_post_eval_fix/confirmation/`,
`results/protocol_post_eval_fix_d5b22ad/confirmation/`.

| método | FAA | esquecimento | BWT | acurácia task-aware |
|---|---:|---:|---:|---:|
| `slowheat_replay_hidden_beta_30_budget_0.25` | **0,7791** | **0,2578** | −0,2578 | 0,9847 |
| `replay` | 0,7704 | 0,2724 | −0,2724 | 0,9833 |

**Contraste primário** (`slowheat_replay − replay`, 20 pares):

| | valor |
|---|---|
| diferença média | **+0,00869** (+0,87 pp) |
| IC 95% (t, df=19) | [+0,00293; +0,01445] |
| t | 3,157 |
| p bicaudal (t) | **0,0052** |
| sinal exato | 17+ / 3−, p = 0,00258 |
| bootstrap IC 95% | [+0,00316; +0,01381] |

**As duas execuções independentes concordam em todos os campos científicos.**
Das 100 diferenças escalares entre os arquivos por seed, 21 divergem e **todas
são campos de custo** (tempo, memória de pico, tempo de seleção).

**Limite declarado:** as duas runs registram árvore Git suja. O diff executado
não foi fingerprintado, então reprodução bit-a-bit não é estabelecível pelos
manifestos.

---

## 2. Baselines Split-MNIST — exploratório

**Sem pré-registro.** 12 métodos, épocas iguais.
**Artefatos:** `results/protocol_post_eval_fix_d5b22ad/all_baselines_equal_epochs/`,
`results/100seeds/all_baselines_equal_epochs/`.

### 20 seeds

| método | FAA | esquecimento | BWT | task-aware |
|---|---:|---:|---:|---:|
| `derpp` | **0,8269** | 0,2023 | −0,2023 | 0,9846 |
| `replay_early_stopping` | 0,7807 | 0,2474 | −0,2474 | 0,9807 |
| **`slowheat_replay_...beta_30_budget_0.25`** | **0,7767** | **0,2618** | −0,2618 | 0,9836 |
| `replay_balanced` | 0,7750 | 0,2673 | −0,2673 | 0,9835 |
| `replay` | 0,7654 | 0,2796 | −0,2796 | 0,9828 |
| `replay_more_epochs` | 0,7602 | 0,2873 | −0,2873 | 0,9826 |
| `er_ace` | 0,7106 | **0,0415** | **+0,4848** | 0,9821 |
| `agem` | 0,2094 | 0,9780 | −0,9780 | 0,9116 |
| `vanilla` | 0,1963 | 0,9931 | −0,9931 | 0,6848 |
| `si` | 0,1961 | 0,9930 | −0,9930 | 0,6924 |
| `ewc` | 0,1961 | 0,9929 | −0,9929 | 0,6842 |
| `lwf_calibrated` | 0,1958 | 0,9932 | −0,9932 | 0,8657 |

### 100 seeds (maior *n* do repositório; nenhuma tabela publicada o usa)

| método | FAA | esquecimento | BWT |
|---|---:|---:|---:|
| `derpp` | 0,8260 | 0,2024 | −0,2024 |
| `replay_early_stopping` | 0,7802 | 0,2482 | −0,2482 |
| **`slowheat_replay_...`** | **0,7760** | **0,2623** | −0,2623 |
| `replay_balanced` | 0,7752 | 0,2667 | −0,2667 |
| `replay` | 0,7663 | 0,2783 | −0,2783 |
| `replay_more_epochs` | 0,7628 | 0,2838 | −0,2838 |
| `er_ace` | 0,7126 | 0,0398 | +0,4897 |
| `agem` | 0,2075 | 0,9800 | −0,9800 |
| `si` | 0,1961 | 0,9929 | −0,9929 |
| `vanilla` | 0,1960 | 0,9929 | −0,9929 |
| `ewc` | 0,1959 | 0,9928 | −0,9928 |
| `lwf_calibrated` | 0,1958 | 0,9930 | −0,9930 |

> **Leitura obrigatória:** `er_ace` tem o **menor esquecimento** da tabela
> (0,04) e BWT **positivo**, com FAA 6 pp abaixo do SlowHeat+Replay. Esquecimento
> baixo sem acurácia alta é um método que aprendeu pouco e esqueceu pouco. Este
> é o motivo pelo qual nenhuma das duas métricas pode ser lida sozinha.
>
> `replay_more_epochs` e `replay_early_stopping` **não têm a mesma contagem de
> passos** que as demais. Comparação equal-epochs e equal-examples são análises
> separadas.

---

## 3. SlowHeat + DER++ — exploratório

**Artefato:** `results/protocol_post_eval_fix_d5b22ad/slowheat_derpp_exploratory/`, 20 seeds.

| método | FAA | esquecimento | BWT | task-aware |
|---|---:|---:|---:|---:|
| `slowheat_derpp_hidden_beta_30_budget_0.25` | **0,8583** | **0,1485** | −0,1485 | 0,9844 |
| `derpp` | 0,8269 | 0,2023 | −0,2023 | 0,9846 |
| `slowheat_replay_hidden_beta_30_budget_0.25` | 0,7767 | 0,2618 | −0,2618 | 0,9836 |
| `replay` | 0,7654 | 0,2796 | −0,2796 | 0,9828 |

> **Este é o maior número do projeto para um braço SlowHeat em Split-MNIST
> (+3,14 pp sobre DER++), e ele NÃO é confirmatório.** Não há pré-registro, não
> há correção de multiplicidade declarada, e o `README.md` do projeto lista
> explicitamente superioridade sobre DER++ como claim **não suportado**.

---

## 4. Split-CIFAR MLP — exploratório

**Artefatos:** `results/split_mnist_protocol/split_cifar10/`, `.../split_cifar100/`, 10 seeds.
Host MLP (o loader entrega vetores achatados `[N, 3072]`).

### Split-CIFAR-10 (MLP)

| método | FAA | esquecimento | BWT |
|---|---:|---:|---:|
| `slowheat_er_ace_hidden_beta_30_budget_0.25` | **0,3033** | 0,2263 | **+0,0596** |
| `er_ace` | 0,2730 | 0,2542 | +0,0355 |
| `slowheat_replay_..._calibrated` | 0,2250 | 0,7566 | −0,7566 |
| `derpp` | 0,2193 | 0,7709 | −0,7709 |
| `slowheat_derpp_...` | 0,2111 | 0,7791 | −0,7791 |
| `slowheat_replay` | 0,2076 | 0,7833 | −0,7833 |
| `replay_balanced` | 0,2032 | 0,7848 | −0,7848 |
| `slowheat_replay_hidden_beta_30_budget_0.25` | 0,2025 | 0,7882 | −0,7882 |
| `vanilla` | 0,1718 | 0,8285 | −0,8285 |
| `ewc` | 0,1718 | 0,8284 | −0,8284 |
| `slowheat_hidden_beta_30_budget_0.25` | 0,1715 | 0,8309 | −0,8309 |
| `si` | 0,1710 | 0,8251 | −0,8251 |
| `slowheat` | 0,1684 | 0,8276 | −0,8276 |

### Split-CIFAR-100 (MLP)

| método | FAA | esquecimento | BWT |
|---|---:|---:|---:|
| `slowheat_er_ace_hidden_beta_30_budget_0.25` | **0,1491** | 0,1467 | **+0,0679** |
| `er_ace` | 0,1354 | 0,1543 | +0,0278 |
| `slowheat_replay_..._calibrated` | 0,1217 | 0,4142 | −0,4142 |
| `derpp` | 0,1152 | 0,4452 | −0,4452 |
| `slowheat_derpp_...` | 0,1107 | 0,4457 | −0,4457 |
| `slowheat_replay` | 0,1009 | 0,4803 | −0,4803 |
| `slowheat_replay_hidden_beta_30_budget_0.25` | 0,0992 | 0,4782 | −0,4782 |
| `replay_balanced` | 0,0958 | 0,4727 | −0,4727 |
| `si` | 0,0564 | 0,5293 | −0,5293 |
| `ewc` | 0,0559 | 0,5273 | −0,5273 |
| `slowheat_hidden_beta_30_budget_0.25` | 0,0550 | 0,5320 | −0,5320 |
| `slowheat` | 0,0547 | 0,5346 | −0,5346 |
| `vanilla` | 0,0546 | 0,5259 | −0,5259 |

> **O sinal do efeito SlowHeat depende do método base.** Ligado ao ER-ACE,
> melhora; ligado ao DER++, **piora** (−0,82 pp no CIFAR-10, −0,45 no
> CIFAR-100). Sozinho (`slowheat`, `slowheat_hidden_*`), é indistinguível do
> `vanilla` nos dois.
>
> Chance é 0,10 (CIFAR-10, 10 classes) e 0,01 (CIFAR-100). No CIFAR-100 a
> maioria dos braços está em 5–15× a chance, mas `vanilla`, `ewc`, `si` e
> `slowheat` ficam todos em ~5,5×, num aglomerado onde a separação é pequena.

---

## 5. Split-CIFAR-10 convolucional — exploratório

**Artefatos:** `results/split_mnist_protocol/split_cifar10_{cnn,vgg11,resnet18}*/`.

### CNN (10 seeds)

| método | FAA | esquecimento | BWT |
|---|---:|---:|---:|
| `slowheat_lpr` | **0,3745** | 0,5690 | −0,5690 |
| `slowheat_scroll` | 0,3711 | 0,2163 | −0,1876 |
| `lpr` | 0,3639 | 0,6176 | −0,6176 |
| `classifier_expander` | 0,3378 | 0,3030 | −0,1968 |
| `slowheat_classifier_expander` | 0,3258 | 0,2542 | −0,0989 |
| `scroll` | 0,3152 | 0,2340 | −0,1509 |
| `slowheat_none` | 0,1736 | 0,8501 | −0,8501 |
| `vanilla` | 0,1728 | 0,8514 | −0,8514 |
| `slowheat_unidirectional` | 0,1685 | 0,8255 | −0,8255 |
| `slowheat` | **0,1541** | 0,7920 | −0,7920 |
| `hard_freeze` | 0,1512 | 0,7831 | −0,7831 |

### VGG11 (10 seeds, `*_all_methods`)

| método | FAA | esquecimento | BWT |
|---|---:|---:|---:|
| `slowheat_classifier_expander` | **0,4351** | 0,2842 | −0,0836 |
| `classifier_expander` | 0,4234 | 0,3340 | −0,2719 |
| `slowheat_lpr` | 0,2703 | 0,7881 | −0,7881 |
| `lpr` | 0,2379 | 0,8309 | −0,8309 |
| `slowheat_scroll` | 0,2305 | 0,1798 | −0,1288 |
| `vanilla` | 0,1873 | 0,8978 | −0,8978 |
| `slowheat_none` | 0,1871 | 0,8950 | −0,8950 |
| `slowheat` | 0,1800 | 0,8789 | −0,8789 |
| `hard_freeze` | 0,1522 | 0,7923 | −0,7923 |

### ResNet18 (10 seeds, `*_all_methods`)

| método | FAA | esquecimento | BWT |
|---|---:|---:|---:|
| `classifier_expander` | **0,3730** | 0,3310 | −0,2712 |
| `slowheat_classifier_expander` | 0,3642 | 0,3365 | −0,2814 |
| `lpr` | 0,2529 | 0,7868 | −0,7868 |
| `slowheat_scroll` | 0,2488 | 0,1927 | −0,1192 |
| `slowheat_lpr` | 0,2407 | 0,7819 | −0,7819 |
| `slowheat_none` | 0,1754 | 0,8571 | −0,8571 |
| `vanilla` | 0,1736 | 0,8600 | −0,8600 |
| `slowheat` | **0,1352** | 0,8124 | −0,8124 |
| `scroll` | 0,1101 | 0,3926 | −0,3537 |

### Functional DualHeat (10 seeds)

| backbone | braço | FAA | esquecimento | BWT |
|---|---|---:|---:|---:|
| VGG11 | `slowheat_classifier_expander` | **0,4401** | 0,2578 | −0,0750 |
| VGG11 | `dualheat_classifier_expander` | 0,4271 | 0,2172 | **+0,0854** |
| VGG11 | `classifier_expander` | 0,4210 | 0,3426 | −0,2880 |
| ResNet18 | `classifier_expander` | **0,3760** | 0,3271 | −0,2650 |
| ResNet18 | `slowheat_classifier_expander` | 0,3623 | 0,3308 | −0,2797 |
| ResNet18 | `dualheat_classifier_expander` | 0,3573 | 0,2034 | −0,0398 |

> **`slowheat` puro fica ABAIXO do `vanilla` nos três backbones
> convolucionais** (0,1541 vs 0,1728; 0,1800 vs 0,1873; 0,1352 vs 0,1736). O
> ganho aparece só quando acoplado a `classifier_expander`, `lpr` ou `scroll` —
> e nesse caso a comparação é entre algoritmos, não isolamento do SlowHeat.
>
> **VGG11 aparece com 5 e com 10 seeds** em diretórios diferentes
> (`split_cifar10_vgg11` tem 5). Os números acima são os de 10 seeds.

---

## 6. Hard vs Soft — ⚠️ o artefato se declara `exploratory_reanalysis`

> **Correção de enquadramento.** O protocolo
> `docs/protocols/hard_vs_soft_protection.md` foi congelado antes da execução e
> a run tem árvore Git limpa, mas o `pair_report.json` carrega
> `status = "exploratory_reanalysis"` e `analysis_scope = "selected_pairs"`.
> **O artefato manda, não a minha memória do protocolo.** Trato como
> pré-registrado-mas-reanalisado: os quatro contrastes por dataset com Holm
> estão declarados, e os pares foram *selecionados* para a reanálise.

5 alvos, 10 seeds, Holm sobre os quatro contrastes de acurácia dentro de cada dataset.
**Artefatos:** `results/hard_vs_soft/*/pair_report.json`.

| alvo | contraste | FAA cand. | FAA ref. | ΔFAA | sinais | p |
|---|---|---:|---:|---:|---:|---:|
| **split_mnist_mlp** | Hard vs Soft | 0,1964 | 0,1915 | +0,50 pp | 5+/5− | 1,000 |
| | Hard vs Soft (replay) | 0,7671 | 0,7717 | −0,46 pp | 5+/5− | 1,000 |
| | Hard vs Replay | 0,7671 | 0,7601 | +0,69 pp | 7+/3− | 0,344 |
| | Hard vs Vanilla | 0,1964 | 0,1952 | +0,12 pp | 3+/7− | 0,344 |
| **permuted_mnist_mlp** | Hard vs Soft | 0,8965 | 0,8877 | +0,88 pp | 7+/3− | 0,344 |
| | Hard vs Soft (replay) | 0,9310 | 0,9433 | **−1,23 pp** | 0+/10− | **0,00195** |
| | Hard vs Replay | 0,9310 | 0,9320 | −0,10 pp | 5+/5− | 1,000 |
| | Hard vs Vanilla | 0,8965 | 0,7266 | **+16,99 pp** | 10+/0− | **0,00195** |
| **split_cifar10_mlp** | Hard vs Soft | 0,1796 | 0,1685 | +1,11 pp | 7+/3− | 0,344 |
| | Hard vs Soft (replay) | 0,2301 | 0,2074 | **+2,27 pp** | 10+/0− | **0,00195** |
| | Hard vs Replay | 0,2301 | 0,2046 | +2,55 pp | 9+/1− | 0,0215 |
| | Hard vs Vanilla | 0,1796 | 0,1713 | +0,84 pp | 6+/4− | 0,754 |
| **split_cifar100_mlp** | Hard vs Soft | 0,0373 | 0,0514 | **−1,41 pp** | 0+/10− | **0,00195** |
| | Hard vs Soft (replay) | 0,0597 | 0,0999 | **−4,01 pp** | 0+/10− | **0,00195** |
| | Hard vs Replay | 0,0597 | 0,0917 | **−3,19 pp** | 0+/10− | **0,00195** |
| | Hard vs Vanilla | 0,0373 | 0,0556 | **−1,82 pp** | 0+/10− | **0,00195** |
| **split_cifar10_cnn** | Hard vs Soft | 0,1441 | 0,1557 | −1,16 pp | 4+/6− | 0,754 |
| | Hard vs Soft (replay) | 0,3026 | 0,3379 | **−3,53 pp** | 0+/10− | **0,00195** |
| | Hard vs Replay | 0,3026 | 0,3648 | **−6,22 pp** | 0+/10− | **0,00195** |
| | Hard vs Vanilla | 0,1441 | 0,1738 | **−2,97 pp** | 0+/10− | **0,00195** |

### Esquecimento nos mesmos contrastes

| alvo | contraste | forget cand. | forget ref. | Δ | sinais | p |
|---|---|---:|---:|---:|---:|---:|
| split_mnist_mlp | Hard vs Soft | 0,9697 | 0,9891 | −1,94 pp | 0+/9− | 0,00391 |
| permuted_mnist_mlp | Hard vs Vanilla | 0,0627 | 0,2946 | **−23,19 pp** | 0+/10− | 0,00195 |
| permuted_mnist_mlp | Hard vs Soft (replay) | 0,0201 | 0,0173 | +0,27 pp | 9+/1− | 0,0215 |
| split_cifar100_mlp | Hard vs Soft | 0,4132 | 0,5300 | **−11,68 pp** | 0+/10− | 0,00195 |
| split_cifar10_cnn | Hard vs Soft (replay) | 0,5898 | 0,5611 | +2,87 pp | 10+/0− | 0,00195 |
| split_cifar10_mlp | Hard vs Soft (replay) | 0,7291 | 0,7819 | −5,27 pp | 0+/10− | 0,00195 |

> **Hard protection não bate soft em lugar nenhum**, e perde onde a capacidade
> é apertada (CIFAR-100: −1,41 pp, 10/10 seeds). A vantagem do hard medida em
> BERT **não transfere**.
>
> **O caso mais instrutivo é o split_cifar100_mlp:** o hard reduz o esquecimento
> em 11,68 pp, 10/10 seeds, **e perde 1,41 pp de acurácia**, também 10/10. Reduzir
> esquecimento e melhorar o resultado são coisas diferentes.

---

## 7. Seletor de replay (R-D) — ⚠️ `exploratory_not_independent_confirmation`

> **Correção de enquadramento.** O `sweep_report.json` declara
> `status = "exploratory_not_independent_confirmation"`. O desenho tem
> pré-registro (`goals/protocol_replay_selector_confirmation.md`), 20 seeds e
> Holm, mas o próprio artefato recusa o rótulo de confirmação independente.
> Trato como o artefato manda.

20 seeds, Split-CIFAR-100, Holm dentro de cada família
dataset/backbone/learner. Diferenças contra o seletor `first`.
**Artefato:** `results/replay_selector_confirmation/sweep_report.json`.

| learner | seletor | ΔFAA | sinais | p | Δesquecimento |
|---|---|---:|---:|---:|---:|
| `derpp` | `loss` | **−4,05 pp** | 0+/20− | 1,9e−06 | +5,30 pp |
| `derpp` | `representative` | +1,01 pp | 20+/0− | 1,9e−06 | −1,28 pp |
| `replay` | `loss` | −2,72 pp | 0+/20− | 1,9e−06 | +2,24 pp |
| `replay` | `representative` | +0,67 pp | 20+/0− | 1,9e−06 | −0,89 pp |
| `slowheat_derpp_...` | `loss` | −3,49 pp | 0+/20− | 1,9e−06 | +5,14 pp |
| `slowheat_derpp_...` | `representative` | +0,75 pp | 20+/0− | 1,9e−06 | −1,16 pp |
| `slowheat_replay_...` | `loss` | −2,86 pp | 0+/20− | 1,9e−06 | +2,68 pp |
| `slowheat_replay_...` | `representative` | +0,97 pp | 19+/1− | 4,0e−05 | −1,10 pp |

> **Unânime em 4 learners:** selecionar por perda alta **piora** (até −4,05 pp),
> selecionar por representatividade melhora. O resultado é sobre o seletor e
> vale igualmente com e sem SlowHeat.
>
> **Aviso do próprio artefato:** cada learner seleciona a própria memória, então
> os contrastes Replay/SlowHeat+Replay são comparações de algoritmo, **não
> isolamento do efeito SlowHeat**.

---

## 8. BERT diagnóstico mecanístico — exploratório

**10 seeds** `[2, 9, 10, 28, 30, 32, 57, 67, 2005, 2012]`, Split-CLINC150,
**2 tarefas**, validation-only.
**Artefato:** `results/bert_slowheat_review/diagnostic_summary.json`.

| condição | T1 aquis. | T1 retenção | T1 esquec. | T2 aquis. | FAA | BWT |
|---|---:|---:|---:|---:|---:|---:|
| `slowheat_hard` | 0,9630 | **0,5527** | 0,4103 | 0,8750 | **0,7138** | −0,4103 |
| `slowheat_random_hard` | 0,9630 | 0,4590 | 0,5040 | 0,8957 | 0,6773 | −0,5040 |
| `slowheat_beta_30` | 0,9630 | 0,4283 | 0,5347 | 0,8927 | 0,6605 | −0,5347 |
| `slowheat_beta_10` | 0,9630 | 0,2910 | 0,6720 | 0,9020 | 0,5965 | −0,6720 |
| `slowheat_beta_3` | 0,9630 | 0,1267 | 0,8363 | 0,9083 | 0,5175 | −0,8363 |
| `vanilla` | 0,9630 | 0,0540 | 0,9090 | **0,9153** | 0,4847 | −0,9090 |

> **A aquisição da T1 é idêntica (0,9630) em todas as condições** — a proteção
> só age depois da primeira consolidação. A troca aparece na T2: quanto mais
> retenção, menos aquisição nova (0,8750 no hard contra 0,9153 no vanilla).

---

## 9. BERT ablação de critério (R-A, R-B) — **confirmatório**

**Pré-registro:** `goals/protocol_importance_criterion_ablation.md`, congelado
em `a3f5117`. 10 seeds (banda 4.000.003+), família de 2, Holm, sinal exato.
**Artefato:** `results/criterion_ablation/`.

| braço | critério | T1 aquis. | T1 retenção | T1 esquec. | T2 aquis. | FAA |
|---|---|---:|---:|---:|---:|---:|
| `slowheat_magnitude_hard` | `\|z\|` | 0,8487 | **0,4423** ±0,0817 | 0,4063 | 0,7023 | 0,5723 |
| `slowheat_hard` | `\|z·dL/dz\|` | 0,8487 | 0,4407 ±0,0809 | 0,4080 | 0,7117 | **0,5762** |
| `slowheat_random_hard` | aleatório | 0,8487 | 0,3200 ±0,0630 | 0,5287 | 0,7543 | 0,5372 |
| `vanilla` | — | 0,8487 | 0,0887 ±0,0447 | 0,7600 | **0,8463** | 0,4675 |

**Família confirmatória** (endpoint: retenção da T1):

| contraste | média | sinais | p | p Holm |
|---|---:|---:|---:|---:|
| `magnitude − aleatório` | **+12,23 pp** | 10+/0− | 0,00195 | **0,00391** ✓ |
| `funcional − magnitude` | −0,17 pp | 5+/5− | 1,000 | 1,000 ✗ |

*(referência: `funcional − aleatório` = +12,07 pp, 10+/0−, fora da família)*

**Falsificador da §F** (`degeneracy_overlap.json`):

| diagnóstica | valor | limiar | veredito |
|---|---:|---:|---|
| variância do ranking, `funcional` | 2,9124e−03 | — | — |
| variância do ranking, `magnitude` | 2,9070e−03 | — | diferença de 0,2% |
| sobreposição top-k `magnitude`×`aleatório` | **0,5748** | > 0,80 degenera | **passa** |
| sobreposição top-k `magnitude`×`funcional` | 0,8814 | — | os critérios concordam |

> **A proteção seletiva funciona (+12,23 pp, 10/10). O componente do gradiente
> não compra nada (−0,17 pp, 5+/5−).** O empate não é artefato: as duas
> diagnósticas obrigatórias da §F confirmam que o braço `magnitude` foi
> genuinamente exercido.
>
> **Limite:** um host, um benchmark, **duas tarefas**. Não é superioridade sobre
> EWC/DER++/ER-ACE — esses métodos não existem neste host.

---

## 10. BERT sequência longa `T = 5` — **exploratório, 1 seed**

**Calibração** do §I de `goals/protocol_long_sequence_criterion.md`. Seed
`11_900_003`, **fora da banda confirmatória**. 167 s, GTX 1080 Ti.
**Artefato:** `results/long_sequence_criterion_calibration/`.

| braço | T1 aquis. | T1 retenção | T1 esquec. | últ. tarefa | FAA | FAA/chance | esquec. médio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `vanilla` | 0,8133 | **0,0000** | 0,8133 | 0,9133 | 0,1867 | 14,00× | 0,7925 |
| `slowheat_hard` | 0,8133 | **0,0000** | 0,8133 | 0,8233 | 0,1853 | 13,90× | **0,6750** |
| `slowheat_random_hard` | 0,8133 | **0,0000** | 0,8133 | 0,8667 | 0,1767 | 13,25× | 0,7200 |
| `slowheat_magnitude_hard` | 0,8133 | **0,0000** | 0,8133 | 0,8000 | 0,1733 | 13,00× | 0,6792 |

**Esquecimento por tarefa** (onde a informação ainda está):

| braço | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|
| `slowheat_hard` | 0,813 | 0,773 | 0,690 | **0,423** |
| `slowheat_magnitude_hard` | 0,813 | 0,767 | 0,707 | **0,430** |
| `slowheat_random_hard` | 0,813 | 0,803 | 0,797 | **0,467** |
| `vanilla` | 0,813 | 0,880 | 0,807 | **0,670** |

> **O endpoint primário pré-registrado tem variância zero.** Não é ruído (em
> `T = 2` a amplitude entre braços é 0,354) nem regime degenerado (14× a
> chance). É efeito de piso: a tarefa 1 satura no teto do esquecimento.
>
> **As 10 seeds confirmatórias NÃO foram gastas.** `average_forgetting` ordena
> os braços, mas promovê-lo depois de ver isso seria pesca de endpoint. Análise:
> [`retention_floor_effect.md`](retention_floor_effect.md).

---

## 11. Qwen+LoRA confirmação — **confirmatório**

**10 seeds** (banda 700.001+), Qwen2.5-0.5B, CLINC150, **10 tarefas**, braços
pareados em plasticidade efetiva medida (`E = 0,85`).
**Artefato:** `results/qwen_lora_confirmation/sweep.json`.

| braço | FAA | esquecimento | BWT | E |
|---|---:|---:|---:|---:|
| `exact` | **0,6384** | **0,3278** | −0,3278 | 0,850 |
| `lr_control` | 0,5790 | 0,4200 | −0,4200 | 1,000 |
| `vanilla` | 0,5706 | 0,4296 | −0,4296 | 1,000 |

> ⚠️ **O `E = 1,000` do `lr_control` NÃO é um bug.** `effective_plasticity()`
> mede `mean(1 / (1 + β·heat))`, ou seja, apenas a fração de learning rate que
> sobrevive **à máscara de heat** (`experiments/capacity_calibration.py:98`). O
> `lr_control` não tem máscara — ele reduz o learning rate globalmente — então
> seu heat é zero e a métrica dá 1,0 por construção. O braço remove
> plasticidade por outro mecanismo, que esta métrica não enxerga.
>
> **Isso não é um detalhe de implementação: é o próprio R-E.** Uma métrica de
> plasticidade que só mede um dos caminhos pelos quais a plasticidade é
> removida não pareia braços que usam caminhos diferentes. A correção do R-E
> (bloco 14) trocou para **plasticidade de superfície**, que atribui 0,62246 aos
> dois braços do contraste e torna o pareamento verificável.
>
> **Consequência para esta tabela:** `exact` (E medido = 0,85) e `lr_control`
> (E = 1,0 pela métrica antiga) **não estão pareados na mesma escala**, e a
> diferença de +5,94 pp entre eles mistura *como* com *quanto*. O resultado
> pareado defensável é o do bloco 14.

---

## 12. Qwen+LoRA iso-plasticidade — exploratório

**10 seeds** `[11..110]`.
**Artefatos:** `results/qwen_lora_iso_10seed/`, `results/qwen_lora_sweep_10seed/`.

### Com pareamento de plasticidade (`E = 0,85` alvo)

| braço | FAA | esquecimento | E |
|---|---:|---:|---:|
| `exact` | **0,6494** | **0,3160** | 0,850 |
| `vanilla` | 0,6016 | 0,3967 | 1,000 |
| `slice` | 0,5875 | 0,4126 | 0,850 |
| `leak` | 0,5865 | 0,4136 | 0,850 |
| `rank` | 0,5808 | 0,4191 | 0,850 |
| `lr_control` | 0,5684 | 0,4350 | 1,000 |

### Sem pareamento (`E` livre)

| braço | FAA | esquecimento | E medido |
|---|---:|---:|---:|
| `exact` | 0,6365 | 0,3338 | 0,9234 |
| `slice` | 0,6203 | 0,3725 | **0,0625** |
| `vanilla` | 0,6016 | 0,3967 | 1,000 |
| `rank` | 0,5983 | 0,4012 | 0,6454 |
| `leak` | 0,5958 | 0,4001 | 0,6586 |

> **O par de tabelas é a demonstração do argumento central do ICML.** O `slice`
> sem pareamento aparece em 2º lugar com `E = 0,0625` — removeu 94% da
> plasticidade. Com `E` pareado em 0,85, cai para 4º. **A ordenação dos métodos
> muda conforme quanta plasticidade cada um removeu**, e sem o instrumento isso
> é invisível.

---

## 13. Decomposição do `exact` (R-C) — exploratório

**10 seeds** (banda 5.175.094+), Qwen2.5-0.5B + LoRA, 10 tarefas.
**Artefato:** `results/exact_decomposition/seed_*/manifest.json`.

| braço | FAA | esquecimento |
|---|---:|---:|
| `exact` | **0,6400** | **0,3232** |
| `frozen_a_control` | 0,6331 | 0,3324 |
| `vanilla` | 0,5611 | 0,4442 |

| contraste | ΔFAA | sinais |
|---|---:|---:|
| `exact − frozen_a_control` | +0,69 pp | **5+ / 5−** |

> **Todo o ganho do `exact` vem de congelar a matriz `A` (= LoRA-FA). A máscara
> não adiciona nada:** 5+/5−, o empate mais limpo possível com n=10.
>
> O próprio manifesto declara `status = exploratory_single_seed` e
> `claim_scope = "feasibility and cost measurement; one seed supports no
> inferential comparison between arms"` por seed; a agregação das 10 é feita
> aqui a partir dos manifestos.

---

## 14. Pareamento por superfície (R-E) — **confirmatório**

**10 seeds** (banda 6.000.003+), Holm sobre a família.
**Artefato:** `results/plasticity_matched/analysis.json`.

| braço | FAA | esquecimento | plasticidade de superfície | params treináveis | aquis. última | retenção primeira |
|---|---:|---:|---:|---:|---:|---:|
| `frozen_a_control` | **0,6267** | **0,3480** | 0,62246 | 4.214.016 | 0,895 | **0,093** |
| `vanilla` | 0,5879 | 0,4125 | 1,00000 | 6.769.920 | **0,923** | 0,049 |
| `lr_control` | 0,5275 | 0,4741 | 0,62246 | 6.769.920 | 0,918 | 0,043 |

**Contraste primário** `frozen_a_control − lr_control`:

| endpoint | média | mediana | sinais | p bruto | p Holm |
|---|---:|---:|---:|---:|---:|
| esquecimento | **−0,1260** | −0,1287 | 0+/10− | 0,00195 | **0,00391** ✓ |
| FAA | **+0,0992** | +0,1105 | 10+/0− | 0,00195 | ✓ |

> **Os dois braços têm plasticidade de superfície idêntica (0,62246) e
> resultados muito diferentes.** Isso isola *como* a plasticidade foi removida,
> não *quanta* — que é a tese do artigo do ICML.
>
> **`lr_control` fica ABAIXO do `vanilla` em acurácia** (0,5275 vs 0,5879).
> Atenção: a hipótese `lr_control < vanilla` como regra geral **não replicou**
> (R-F) e teve 7 ocorrências corrigidas em `45de59e`. Aqui é o valor medido
> nesta run, não um achado replicado.

---

## 15. Re-avaliação de métodos de penalidade — mecanismo + confirmatório

### 15.1 Passada 1 Split-CIFAR-100 (12 seeds, mecanismo-only)

`E` = razão de normas do update (plasticidade efetiva). **Sem leitura de acurácia.**

| arm | `E` médio | min | max | acima de 1 | p | pareável? | cosseno médio | cosseno min |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `ewc` | 1,00128 | 1,00112 | 1,00145 | 12/12 | 0,00049 | **não** | 0,9968 | 0,8979 |
| `si` | 1,01238 | 1,01205 | 1,01279 | 12/12 | 0,00049 | **não** | 0,9357 | 0,0884 |
| `mas` | 1,01939 | 1,01503 | 1,02682 | 12/12 | 0,00049 | **não** | 0,4489 | **−0,8758** |

> **Sob AdamW os três métodos AMPLIFICAM o update** (`E > 1`), em 12/12 seeds.
> Um método de consolidação que aumenta a norma do passo não está controlando
> plasticidade. Nenhum é pareável, e a família confirmatória caiu de 6 para 0.
>
> **O cosseno do MAS chega a −0,88:** o update inverte de direção. Nenhum escalar
> de plasticidade descreve isso.

### 15.2 L2 — SGD puro (12 seeds, 3 learning rates)

| arm | `E` (lr=0,01) | divergente? |
|---|---:|---|
| `ewc` | 0,8876 | não |
| `mas` | 0,8991 | não |
| `si` | 9,97e+16 | **sim** (291 amostras não finitas) |

> **`E > 1` era artefato do otimizador adaptativo, não propriedade do método.**
> Sob SGD puro os `E` ficam abaixo de 1 e a passada 2 volta a ter objeto.

### 15.3 L3 — varredura de λ (10 seeds × 5 pontos)

| arm | λ=1 | λ=3 | λ=10 | λ=30 | λ=100 |
|---|---:|---:|---:|---:|---:|
| `ewc` | 0,9517 | 0,9124 | **10/10 div.** | div. | div. |
| `si` | 0,9730 | 0,9561 | 9/10 div. | div. | div. |
| `mas` | 0,9906 | — | — | **0,854** ✓ | **0,728** ✓ |

> De 15 células, **duas** têm poder, e as duas são do `mas`. `ewc` e `si` passam
> de "fraco demais" a "divergente" **sem janela utilizável entre os dois
> regimes**.

### 15.4 Passada 2 — `mas` a 30× contra controle pareado (12 seeds) — **confirmatório**

**Pré-registro:** `goals/protocol_penalty_pass2.md`. `lr_scale = 0,8541056558955461`
congelado antes. Família de 1, sem Holm.

| arm | FAA | esquecimento |
|---|---:|---:|
| `mas` (λ=30) | **0,1976** ±0,0269 | **0,7536** ±0,0645 |
| `lr_control` | 0,1601 ±0,0294 | 0,7937 ±0,0584 |
| `vanilla` | 0,1487 ±0,0243 | 0,8381 ±0,0556 |

| contraste | média | sinais | p |
|---|---:|---:|---:|
| **`mas − lr_control`** (primário, esquecimento) | **−0,0400** | 0+/12− | **0,00049** |
| `lr_control − vanilla` | −0,0445 | 0+/12− | 0,00049 |

| gate | valor |
|---|---|
| `control_is_distinguishable` | **true** |
| `acquisition_collapse` | false |
| **`near_chance_regime`** | **true** |
| melhor arm | 0,1976 contra chance 0,10 |

> **Q1 foi FALSIFICADA: o MAS bate o controle pareado.** E o resultado sai com
> ressalva, porque o próprio agregado marca `near_chance_regime = true` — os
> três arms terminaram entre 1,5× e 2,0× a chance. **O contraste não separa
> "consolidação" de "mudou menos".**
>
> Este é o precedente que gerou o gate de quase-chance do bloco 10.

---

## Resumo: o que o SlowHeat sustenta, por host

| host | evidência mais forte | status declarado no artefato |
|---|---|---|
| Split-MNIST | +0,87 pp sobre Replay, 17/20, p=0,0052 | **confirmatório** (lock), árvore suja |
| Split-CIFAR MLP | depende do método base; sozinho ≈ vanilla | exploratório |
| CNN/VGG/ResNet | `slowheat` puro **abaixo** do vanilla nos três | exploratório |
| Permuted-MNIST | hard +16,99 pp sobre vanilla, 10/10 | `exploratory_reanalysis` |
| BERT/CLINC150 | proteção seletiva +12,23 pp, 10/10 | **confirmatório**, 2 tarefas |
| Qwen+LoRA | ganho vem de congelar `A`, não da máscara | `exploratory_single_seed` (R-C) |

### Claims NÃO suportados por nenhuma tabela acima

- Superioridade sobre EWC, DER++ ou ER-ACE.
- Redução geral de esquecimento.
- Que o critério funcional `|z·dL/dz|` seja melhor que `|z|`.
- Que proteção hard seja melhor que soft fora de Transformers.
- Que o SlowHeat isolado ajude em backbones convolucionais.
- Qualquer coisa sobre sequências com mais de 2 tarefas em BERT.

---

## Procedência

| run | commit | árvore limpa? | status do artefato |
|---|---|---|---|
| Confirmação Split-MNIST (1ª) | `d5b22ad` | **não** | lock `frozen_before_execution` |
| Confirmação Split-MNIST (2ª) | `f2f7616` | **não** | lock idêntico, `sha256 015b3162…` |
| Hard vs Soft | pré-registrado antes | sim | `exploratory_reanalysis` |
| Seletor de replay | pré-registrado antes | sim | `exploratory_not_independent_confirmation` |
| Ablação de critério | `a3f5117` | sim | confirmatório |
| Qwen+LoRA confirmação | — | — | `confirmatory_multi_seed` |
| Passada 2 | `22461f5` | sim | confirmatório |

Runs descartadas ficam em `results/_abandoned_*` com `WHY_ABANDONED.md`.

**Precedente de invalidação:** a auditoria de 02/09/2026 encontrou avaliação de
fim de época sem restaurar o modo de treino, o que quebrava silenciosamente a
acumulação de importância do SlowHeat. **Toda confirmação executada antes da
correção foi descartada**, não reinterpretada.
