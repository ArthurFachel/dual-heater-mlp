# Próximo passo — MLP (Split-MNIST, Permuted-MNIST, CIFAR/MLP)

> Um de quatro documentos por arquitetura. Índice e decisões transversais em
> [proximo_passo_artigo.md](proximo_passo_artigo.md).
>
> Escopo: `SlowHeatMLP` + `SlowHeatAdamW`. Fonte de evidência:
> [docs/architectures/arch_mlp.md](../docs/architectures/arch_mlp.md).

---

## 1. O que este host sustenta hoje

É o **único host com resultado confirmatório pré-registrado** do projeto.

| Evidência | O que é | Força |
|---|---|---|
| **E1** | SlowHeat+Replay − Replay em Split-MNIST, 20 seeds congeladas: **+0,87 p.p.** de acurácia final (`p = 0,0052`, 17/20) e **−1,46 p.p.** de forgetting (`p = 0,00059`, 18/20) | confirmatória, replicada em duas execuções independentes |
| **E2** | As duas execuções batem em 100% dos campos científicos; só custo difere | replicação |
| **E3** | Suíte `dualheat_pairs`, 10 seeds, 4 contrastes por dataset sob Holm | exploratória |
| **E4** | `hard_vs_soft`, 4 alvos MLP, protocolo congelado e **árvore Git limpa** (`6f4d12d`) | exploratória por declaração, mas a de melhor proveniência |

Este é o host onde a comparação contra baselines de continual learning
**é possível e já foi parcialmente executada**: Replay, DER++, ER-ACE, A-GEM,
EWC, SI e LwF estão implementados e rodados aqui.

## 2. O que este host ainda não sustenta

- **Superioridade geral.** O contraste confirmado é contra Replay puro, em
  Split-MNIST, com uma única configuração (`beta=30`, `budget=0,25`).
- **Vantagem sobre DER++ ou ER-ACE.** O `+3,18 p.p.` sobre DER++ em Split-MNIST
  é exploratório e não tem pré-registro próprio. É o resultado mais forte do
  host e o mais exposto.
- **Ganho task-aware.** `p = 0,043` no t pareado contra `p = 0,115` no sinal
  exato. Não reportar como positivo.
- **Reprodução bit a bit.** As duas execuções da confirmação registram
  `git dirty = true` e o diff não foi fingerprintado (P1 em
  `results_provenance_status.md`).

---

## 3. A âncora que este host sustenta

> *Proteção seletiva em nível de neurônio, acoplada a Replay, melhora
> retenção em Split-MNIST class-incremental sob protocolo congelado e
> replicado, a um custo de 1,8x–1,9x em tempo de parede.*

É uma âncora de **eficácia local**, estreita e verificada. Não tenta ser geral,
e essa é a sua força: é a única alegação de eficácia do projeto que resiste a
um revisor hostil, porque tem pré-registro, replicação independente e teste de
sinais concordante.

O papel do MLP no artigo é ser a **prova de que o mecanismo faz alguma coisa**.
Não é o lugar da contribuição principal.

---

## 4. Caminho crítico

### M1 — Confirmar o contraste contra DER++ (pré-registro obrigatório, CPU)

O `+3,18 p.p.` de SlowHeat+DER++ sobre DER++ em Split-MNIST é o resultado
exploratório mais forte do host e não tem confirmação. Ele é também o único
caminho para uma alegação que não seja "melhor que replay simples".

- Pré-registrar no formato de `protocol_lora_confirmation.md`: braços, seeds
  novas (banda própria, disjunta das 20 de `CONFIRMATORY_SEEDS`), endpoint
  primário único, teste de sinais, critério antes da run.
- Custo: CPU, horas. Não compete por GPU com nada.
- **Desfecho nulo é publicável** e encerra a alegação. O `+3,18` sobreviveu a
  Holm em 10/10 seeds na exploração, então a probabilidade de confirmação é
  razoável — mas o padrão de `resultados_confirmacao.md` (forte em 2 tarefas,
  nulo em 10) recomenda cautela.

### M2 — Fechar a lacuna de proveniência da confirmação existente (CPU, barato)

`git dirty = true` nas duas execuções da E1. Reexecutar a confirmação a partir
de árvore limpa, **sem tocar em pré-registro, seeds ou critério**, apenas para
produzir um artefato com proveniência íntegra. Os números já são conhecidos e
idênticos entre execuções, então isto não é p-hacking — é higiene de artefato.

Se os números mudarem, isso é por si só um achado e precisa ser reportado.

### M3 — Tuning declarado por método (opcional, CPU, caro em tempo humano)

Item 1 da §10 do manuscrito. Todos os agregados usam `lr = 1e-3` fixo, o que dá
ao revisor a objeção "seus baselines não foram ajustados". Em MLP isso é
factível em CPU, ao contrário dos hosts Transformer.

**Decidir se entra:** só vale a pena se a âncora do artigo for eficácia em
MLP/CNN. Se a âncora for de protocolo (ver o hub), isto vira trabalho futuro.

### M4 — Varredura de `beta` × `budget` com fronteira de Pareto (CPU)

Item 9 da §10. O par `(30; 0,25)` vem de exploração anterior à confirmação e
nunca foi justificado. Uma fronteira acurácia × forgetting responde à ameaça
4 da lista de validade e é barata neste host.

---

## 5. O que NÃO fazer aqui

- **Não reanalisar as 20 seeds da E1 com outra família de correção.** O
  pré-registro está congelado e replicado; mexer nele destrói o único ativo
  confirmatório do projeto.
- **Não promover o ganho task-aware.** Os dois testes discordam e isso está
  registrado.
- **Não citar o contraste contra Convencional em Split-MNIST** (−0,12 p.p.)
  como resultado. Compara 19,64% contra 19,52%, ambos no piso do acaso de 20%.
- **Não citar o `+7,67 p.p.` de Permuted-MNIST** como competitivo. A referência
  é fine-tuning sequencial sem regularização, a baseline mais fraca possível.
- **Não misturar este host com BERT/Qwen na mesma tabela de baselines.** DER++,
  ER-ACE e EWC não foram rodados nos hosts Transformer; uma tabela unificada
  criaria células vazias que o revisor vai ler como omissão.

---

## 6. Resumo

| # | Ação | Custo | Bloqueia |
|---|---|---|---|
| M1 | Pré-registrar e confirmar SlowHeat+DER++ vs DER++ | CPU, horas | a única alegação não trivial de eficácia |
| M2 | Reexecutar a confirmação com árvore limpa | CPU, horas | proveniência da E1 |
| M3 | Tuning declarado por método | CPU, dias | só se a âncora for eficácia |
| M4 | Fronteira de Pareto `beta` × `budget` | CPU, horas | ameaça de validade 4 |

M1 e M2 são independentes entre si e de tudo que acontece nos hosts
Transformer. Podem rodar em paralelo com o trabalho de Qwen sem competir por
GPU.

## Referências

- [Evidência e limites do host](../docs/architectures/arch_mlp.md)
- [Protocolo confirmatório](../docs/protocols/confirmatory_protocol.md)
- [Log cronológico Split-MNIST](../docs/results/split_mnist_experiment_log.md)
- [Hard versus soft](../docs/results/hard_vs_soft_results.md)
- [Índice dos próximos passos](proximo_passo_artigo.md)
