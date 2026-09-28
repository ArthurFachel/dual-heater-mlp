# Próximo passo — CNN (Split-CIFAR-10/100, VGG11, ResNet18)

> Um de quatro documentos por arquitetura. Índice e decisões transversais em
> [proximo_passo_artigo.md](proximo_passo_artigo.md).
>
> Escopo: `SlowHeatCNN`, `SlowHeatVGG11`, `SlowHeatResNet18`. Fonte de
> evidência: [docs/architectures/arch_cnn.md](../docs/architectures/arch_cnn.md).

---

## 1. O que este host sustenta hoje

O achado do host não é um ganho, é uma **dependência de interação**:

| Contraste | CIFAR-10 | CIFAR-100 | Sinais |
|---|---:|---:|---|
| vs ER-ACE | **+4,15 p.p.** | **+1,27 p.p.** | 10+/0− nos dois |
| vs DER++ | **−1,13 p.p.** | **−0,53 p.p.** | 0+/10− e 1+/9− |
| vs Replay | +0,35 (p=0,0509) | **+0,62 p.p.** | 7+/3− e 9+/1− |

Os quatro entram em Holm por dataset; ER-ACE e DER++ sobrevivem nos dois, **com
sinais opostos**. Acoplado a ER-ACE o mecanismo ajuda; acoplado a DER++ ele
atrapalha.

Mais dois resultados, ambos negativos e ambos úteis:

- **C2 — FastHeat não ajuda.** Nenhum dos 8 contrastes DualHeat − SlowHeat
  sobrevive a Holm em VGG11 ou ResNet18. O maior (`VGG11 + LPR`, +1,43 p.p.)
  fica em `p = 0,0803`.
- **C3 — hard perde para soft.** Com replay, hard perde 3,53 p.p. para soft e
  6,22 p.p. para replay puro, 10/10 seeds. Protocolo congelado, árvore Git
  limpa (`6f4d12d`).

## 2. A assimetria que decide a leitura

Na CNN, hard reduz forgetting em **8,34 p.p.** contra convencional e mesmo
assim **perde 2,97 p.p.** de acurácia final. Esquecer menos não ajudou porque a
aquisição caiu junto.

Citar só forgetting inverteria a conclusão deste host. Isto vale como regra
editorial para o artigo inteiro: **nenhuma tabela reporta forgetting sem a
acurácia final ao lado.**

---

## 3. A âncora que este host sustenta

> *O efeito da proteção seletiva não tem sinal fixo: ele depende do método base
> ao qual o mecanismo é acoplado, e a dependência é forte o bastante para
> inverter o sinal sob a mesma correção de multiplicidade.*

É uma âncora de **limite**, não de eficácia. O host serve para delimitar
honestamente onde o mecanismo funciona, e o revisor vai valorizar isso
justamente porque o autor reportou a metade negativa.

Este é também o host que mais depende de resolver uma ameaça específica: com
`lr = 1e-3` fixo para todos os métodos, DER++ e ER-ACE podem estar
desigualmente ajustados, e uma interação com learning rate produziria o **mesmo
padrão de inversão**. Ou seja, a explicação alternativa não está excluída.

---

## 4. Caminho crítico

### C-1 — Ablação que testa a hipótese de competição de estabilidade (GPU leve)

A leitura natural — SlowHeat e DER++ disputam o mesmo orçamento de estabilidade,
enquanto ER-ACE é complementar — **não foi testada** e não pode ser afirmada.

Experimento mínimo: varrer a força do termo de consistência do DER++ (`alpha`,
`beta` do DER++) contra a força de proteção do SlowHeat, e verificar se a
degradação é monotônica na soma dos dois. Se for, a hipótese ganha suporte; se
não, o achado permanece descritivo e o artigo diz isso.

Sem esta ablação, a §6 do manuscrito continua reportando um padrão sem
mecanismo, o que é aceitável mas fraco.

### C-2 — Tuning por método, que é a explicação concorrente (GPU, caro)

Item 1 e 4 da §10 do manuscrito. Esta é a objeção mais provável do revisor e a
única capaz de derrubar a inversão inteira.

**Custo realista:** 4 métodos × grade de LR × 10 seeds × 2 datasets em 3 GPUs
Pascal. É a tarefa mais cara do projeto depois da confirmação do Qwen.

Recomendo **não** executar, e em vez disso declarar explicitamente na §6 que a
inversão é medida sob LR uniforme e que o tuning desigual não está excluído.
Isso já está escrito no manuscrito; só precisa subir de nota de rodapé para
limitação declarada.

### C-3 — Analisar os agregados que já existem em disco (CPU, barato, alto retorno)

Quatro conjuntos grandes estão medidos e **nunca foram analisados**:

| Diretório | Conteúdo |
|---|---|
| `split_mnist_protocol/split_cifar10/` | 10 seeds × 19 métodos |
| `split_mnist_protocol/split_cifar100/` | 10 seeds × 19 métodos |
| `cache_derpp_10seeds/replay_selection_sweep/split_cifar*` | **50 seeds** × 5 caches, 959 MB |

O sweep de 50 seeds é o maior `n` disponível para CNN em todo o repositório e
nenhuma tabela publicada o usa. Analisá-lo custa CPU e pode responder à
pergunta de interação sem GPU nenhuma.

**Este é o item de maior retorno por custo deste host.** Deve vir antes de C-1.

Ressalva: como os dados já existem, qualquer análise feita agora é
**exploratória por construção**. Declarar isso, não disfarçar de confirmação.

### C-4 — Fechar a ressalva de proveniência de C3 (CPU, trivial)

As 10 seeds de `hard_vs_soft/split_cifar10_cnn/` treinaram em 23/09, o passo de
análise abortou, e o relatório foi regenerado em 24/09 a partir dos artefatos
treinados. `pair_report.json` já separa `analysis_provenance` da identidade do
treino — só falta o texto do artigo dizer isso em vez de deixar implícito.

---

## 5. O que NÃO fazer aqui

- **Não afirmar a competição de estabilidade** sem C-1. É hipótese plausível,
  não resultado.
- **Não reportar o contraste CIFAR-10 vs Replay como significativo.** Holm dá
  `p = 0,0509`, exatamente na fronteira.
- **Não corrigir os 4 datasets entre si retroativamente.** Uma correção global
  sobre 16 contrastes enfraqueceria os marginais, e escolher a família depois de
  ver o resultado é exatamente o que a correção existe para impedir. Declarar a
  estrutura (Holm dentro do dataset, não entre datasets) e seguir.
- **Não tratar diferenças de ~1 p.p. em CIFAR-100 como relevantes** sem
  ressalva: a acurácia absoluta está entre 5,4% e 14,7% em todos os braços.
- **Não usar este host para alegar eficácia.** O sinal inverte; é um host de
  limite.

---

## 6. Resumo

| # | Ação | Custo | Bloqueia |
|---|---|---|---|
| C-3 | Analisar os agregados já em disco (50 seeds inclusive) | CPU, horas | nada; maior retorno por custo |
| C-1 | Ablação da hipótese de competição de estabilidade | GPU leve | a explicação da §6 |
| C-4 | Declarar a proveniência de análise de C3 no texto | CPU, minutos | honestidade da §8 |
| C-2 | Tuning por método | GPU, semanas | **recomendado não executar**; declarar como limitação |

## Referências

- [Evidência e limites do host](../docs/architectures/arch_cnn.md)
- [Protocolo Split-CIFAR](../docs/protocols/split_cifar.md)
- [Contrato do tracker convolucional](../docs/mechanisms/functional_slowheat_cnn.md)
- [Hard versus soft](../docs/results/hard_vs_soft_results.md)
- [Índice dos próximos passos](proximo_passo_artigo.md)
