# CNN — evidência, protocolo e limites

Documento-guia das arquiteturas convolucionais. É a fonte para a seção de
generalidade do artigo. O contrato de mecanismo para convoluções está em
[functional_slowheat_cnn.md](functional_slowheat_cnn.md).

Escopo: `SlowHeatCNN`, `SlowHeatVGG11`, `SlowHeatResNet18` em Split-CIFAR-10 e
Split-CIFAR-100 class-incremental.

---

## Claim

Em benchmarks convolucionais, o efeito do Functional SlowHeat **depende do
método base ao qual ele é acoplado**, e a direção do efeito inverte:
acoplado a ER-ACE melhora consistentemente (até +4,15 p.p. em Split-CIFAR-10),
acoplado a DER++ **piora** (−1,13 p.p. em CIFAR-10, −0,53 p.p. em CIFAR-100),
ambos sobrevivendo a Holm. Acrescentar FastHeat (Functional DualHeat) ao
SlowHeat não produz ganho robusto em VGG11 nem em ResNet18.

Esta dependência de interação é o achado mais informativo do projeto para além
do MLP, e é honestamente reportável como limite do método.

---

## Evidência citável

### C1 — Interação com o método base (10 seeds, Holm por dataset)

Suíte `dualheat_pairs`, endpoint primário (acurácia média final
class-incremental), diferença sempre candidato menos referência, Holm aplicado
dentro de cada dataset sobre os 4 contrastes.

**Split-CIFAR-10**

| Contraste | Referência | Candidato | Diferença | p Holm | Sinais |
|---|---:|---:|---:|---:|---|
| vs ER-ACE | 26,00% | 30,15% | **+4,148 pp** | 0,0001 | 10+/0− |
| vs DER++ | 22,11% | 20,98% | **−1,133 pp** | <0,0001 | 0+/10− |
| vs Replay | 20,37% | 20,71% | +0,348 pp | 0,0509 | 7+/3− |
| vs Convencional | 17,18% | 17,18% | −0,004 pp | 0,9375 | 5+/4− |

**Split-CIFAR-100**

| Contraste | Referência | Candidato | Diferença | p Holm | Sinais |
|---|---:|---:|---:|---:|---|
| vs ER-ACE | 13,43% | 14,70% | **+1,274 pp** | 0,0004 | 10+/0− |
| vs Replay | 9,27% | 9,89% | **+0,621 pp** | 0,0026 | 9+/1− |
| vs DER++ | 11,36% | 10,83% | **−0,530 pp** | 0,0050 | 1+/9− |
| vs Convencional | 5,44% | 5,49% | +0,051 pp | 0,4100 | 7+/3− |

**Leitura.** O sinal de ER-ACE é forte e unânime nos dois datasets (10+/0− e
10+/0−). O sinal de DER++ é negativo e quase unânime (0+/10− e 1+/9−). Ambos
sobrevivem a Holm. A hipótese natural — SlowHeat e DER++ competem pelo mesmo
mecanismo de estabilidade, enquanto ER-ACE e SlowHeat são complementares —
**não foi testada** e não deve ser afirmada sem ablação dedicada.

**Artefatos:** `results/dualheat_pairs/{split_cifar10,split_cifar100}/pair_report.json`,
commit `be05068`, `status = exploratory_paired_suite`, 10 seeds pareadas.

### C2 — FastHeat sobre SlowHeat em VGG11 e ResNet18 (negativo)

Diferença de acurácia final, sempre `DualHeat − SlowHeat`, com Holm sobre os 4
contrastes primários por arquitetura:

| Arquitetura | Par | Diferença | p Holm |
|---|---|---:|---:|
| VGG11 | `dualheat_lpr − slowheat_lpr` | +1,430 pp | 0,0803 |
| VGG11 | `dualheat − slowheat` | −0,483 pp | 0,0803 |
| VGG11 | `dualheat_classifier_expander − ...` | −1,295 pp | 0,1903 |
| VGG11 | `dualheat_scroll − slowheat_scroll` | −0,146 pp | 0,8772 |
| ResNet18 | todos os 4 pares | −0,007 a −0,496 pp | 1,0000 |

**Nenhum contraste sobrevive a Holm a 5%.** O maior ganho médio (VGG11 + LPR,
+1,43 p.p.) tem `p` ajustado de 0,0803. Conclusão negativa/indeterminada:
FastHeat não trouxe melhoria robusta nas duas arquiteturas sob este protocolo.

**Artefatos:**
`results/split_mnist_protocol/split_cifar10_{vgg11,resnet18}_functional_dualheat/functional_dualheat_analysis.json`.

### C3 — Agregados disponíveis e não analisados

| Diretório | Conteúdo | Estado |
|---|---|---|
| `split_mnist_protocol/split_cifar10/` | 10 seeds × 19 métodos | não analisado |
| `split_mnist_protocol/split_cifar100/` | 10 seeds × 19 métodos | não analisado |
| `split_mnist_protocol/split_cifar10_{cnn,vgg11,resnet18}*` | sweeps por backbone | parcialmente analisado |
| `cache_derpp_10seeds/replay_selection_sweep/split_cifar*` | 50 seeds × 5 caches | não analisado |

O sweep de seleção de replay com **50 seeds** é o maior *n* disponível para
CNN e nenhuma tabela publicada o utiliza.

---

## Proveniência

| Item | C1 | C2 |
|---|---|---|
| Pré-registro congelado | não | manifesto de piloto, não pré-registro |
| Árvore Git limpa | **não** | **não** |
| Correção de multiplicidade | Holm por dataset | Holm por arquitetura |
| Replicado | não | não |

**Ressalva específica de C2:** o manifesto congelado do piloto de FastHeat
valida schema, status e pertinência à grade, mas **não recomputa o vencedor a
partir de um hash imutável do piloto**. Isso limita a força de uma alegação de
pré-registro. Ver [results_provenance_status.md](results_provenance_status.md),
item P3.

---

## O que NÃO se pode afirmar

- **Que SlowHeat melhora continual learning em visão.** O efeito é condicional
  ao método base e inverte de sinal.
- **Que a explicação da inversão seja competição de mecanismos de
  estabilidade.** É a hipótese plausível, mas não há ablação que a teste.
- **Que Functional DualHeat (FastHeat) ajuda.** Nenhum contraste sobrevive a
  Holm nas duas arquiteturas.
- **Que o método escala para CIFAR-100.** A acurácia absoluta é baixa em todos
  os braços (5–15%), então as diferenças ocorrem num regime de desempenho
  fraco.

---

## Ameaças à validade

1. **Acurácia absoluta baixa em CIFAR-100** (5,4% a 14,7%). Diferenças de
   ~1 p.p. num regime desses podem não refletir comportamento em modelos com
   desempenho útil.
2. **Sem tuning por método.** `lr=1e-3` fixo para todos; DER++ e ER-ACE podem
   estar sub-ajustados de formas diferentes, o que é uma explicação alternativa
   para a inversão de sinal observada em C1.
3. **Árvore Git suja em todos os agregados.**
4. **Os 4 contrastes por dataset foram corrigidos por Holm, mas os 4 datasets
   não foram corrigidos entre si.** Uma correção global sobre 16 contrastes
   enfraqueceria os resultados marginais.
5. **CIFAR-10 com Replay fica em `p` Holm = 0,0509**, ou seja, exatamente na
   fronteira. Não deve ser reportado como significativo.

---

## Protocolo

Dados, partições, backbones e comandos em [split_cifar.md](split_cifar.md).
Contrato do tracker convolucional, incluindo a limitação de topologia
residual, em [functional_slowheat_cnn.md](functional_slowheat_cnn.md).

**Restrição de despacho:** os nove identificadores LPR, Classifier Expander e
SCROLL exigem `--backbone cnn`; o runner levanta `ValueError` para outros
backbones.