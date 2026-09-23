# Hard versus soft: o regime de proteção entre arquiteturas

Estado: **preparado, aguardando execução**. Nenhum resultado existe ainda.
O runner, o protocolo e os testes estão versionados; a fila espera o término
da confirmação Qwen.

## A pergunta

Nos hosts Transformer (BERT, Qwen) o projeto usa proteção **hard**: as unidades
consolidadas são congeladas, fator binário. Em MLP e CNN usa proteção **soft**:
`1 / (1 + 30·h)`. Essa diferença nunca foi uma decisão registrada — foi o
caminho que cada eixo tomou.

A consequência é um confundimento: **regime de proteção está perfeitamente
correlacionado com arquitetura**. Quando o manuscrito diz que BERT mostra
+22,92 p.p. sobre fine-tuning sequencial e que CNN mostra efeito dependente do
método base, não é possível saber se a diferença vem da arquitetura ou do
regime. Um revisor pode legitimamente perguntar por que a escolha mudou, e hoje
não há resposta empírica.

Esta suíte remove o confundimento rodando **os dois regimes nos hosts
não-Transformer**, sob um único protocolo pareado.

## O que já existia e o que foi acrescentado

O mecanismo hard **não precisou ser implementado**. O parâmetro `hard` vive em
`src/dual_heater/optim.py:121`, no otimizador genérico, não no de Transformers:

```python
def module_factor(source):
    if hard:
        return (source.slow_heat <= 0.0).to(dtype=source.slow_heat.dtype)
    return source.get_lr_scales()
```

O método `hard_freeze` já estava registrado em `experiments/split_mnist.py` e
já rodou em quatro agregados de CNN.

Acrescentado nesta preparação:

| Item | Onde |
|---|---|
| Método `hard_freeze_replay` | `experiments/split_mnist.py` (2 linhas) |
| Runner da suíte | `experiments/hard_vs_soft.py` |
| Fila que aguarda o Qwen | `run_hard_vs_soft.sh` |
| 13 testes | `tests/test_hard_vs_soft.py` |

`hard_freeze_replay` é o análogo direto de `slowheat_hard_replay` do BERT — o
braço que produziu o achado negativo mais importante do projeto. Sem ele, não
haveria como replicar aquele contraste fora dos Transformers.

## Por que os resultados de CNN existentes não servem

Os quatro agregados com `hard_freeze` são **anteriores à correção de
eval-mode**:

| Agregado | Commit | Data |
|---|---|---|
| `split_cifar10_cnn` | `3861862` | 28/08 |
| `split_cifar10_vgg11` | `37b9c25` | 29/08 |
| `split_cifar10_vgg11_all_methods` | `dc21eed` | 29/08 |
| `split_cifar10_resnet18_all_methods` | `dc21eed` | 29/08 |

A correção é `d5b22ad`, de 03/09 — verificado com
`git merge-base --is-ancestor`. E o bug só afetava sweeps **sem** nenhum método
FastHeat: confirmei que os quatro têm zero, ou seja, a condição se aplica
exatamente a eles. A importância foi acumulada só na primeira época de cada
estágio.

As suítes válidas pós-correção (`dualheat_pairs`, commit `be05068`) não
incluem `hard_freeze`. Portanto: **nenhum resultado válido de proteção hard
existe fora dos Transformers.**

## Desenho

### Braços

| Papel | Identificador |
|---|---|
| Hard | `hard_freeze` |
| Soft (iso-escopo) | `slowheat_beta_30_budget_0.25` |
| Hard + replay | `hard_freeze_replay` |
| Soft + replay | `slowheat_replay_beta_30_budget_0.25` |
| Referências | `replay`, `vanilla` |

### Duas decisões que determinam a validade

**1. O comparador soft não é o usado em `dualheat_pairs`.** Aquela suíte usa
`slowheat_hidden_beta_30_budget_0.25`, que **não protege a camada de saída**.
`hard_freeze` protege. Usar o comparador `_hidden_` misturaria regime de
proteção com escopo de proteção, e o contraste não responderia nada. Por isso
a suíte usa a variante sem `_hidden_`, que protege a saída nos dois braços.
Verificado em `tests/test_hard_vs_soft.py::test_hard_and_soft_arms_share_protection_scope`.

**2. Os dois regimes compartilham o mesmo budget de capacidade (0,25).**
`hard_freeze` binariza exatamente o conjunto que o budget seleciona, então os
braços protegem **as mesmas unidades** e diferem só em quão forte as seguram.

### Contrastes e endpoint

Endpoint primário: acurácia média final class-incremental. Holm sobre os quatro
contrastes dentro de cada dataset.

| # | Contraste | O que responde |
|---|---|---|
| 1 | Hard − Soft | **primário**: o regime importa? |
| 2 | Hard − Soft (replay) | o regime importa quando há replay? |
| 3 | Hard+replay − Replay | replica o achado negativo do BERT? |
| 4 | Hard − Vanilla | sanidade: a proteção faz algo? |

### Alvos

| Dataset | Backbone | Config |
|---|---|---|
| Split-MNIST | MLP | `baseline_config` |
| Permuted-MNIST | MLP | `permuted_mnist` |
| Split-CIFAR-10 | MLP | `split_cifar10` |
| Split-CIFAR-100 | MLP | `split_cifar100` |
| Split-CIFAR-10 | CNN | `split_cifar10_cnn` |

Combinações sem config versionado (Split-MNIST/CNN, Permuted/CNN,
CIFAR-100/CNN) são **recusadas com erro**, não rebaixadas silenciosamente para
MLP. Um diretório rotulado CNN rodando um MLP invalidaria a comparação inteira.

## Expectativa declarada antes da execução

Registrada em `hard_vs_soft_protocol.json`, campo `declared_expectations`:

> O BERT mostrou hard > soft em acurácia e hard < replay. Se o regime explica o
> resultado dos Transformers, hard deve vencer soft aqui também.
>
> **Um resultado nulo ou invertido é informativo e será reportado**: significaria
> que o achado do Transformer é específico da arquitetura.

Declarar isto antes importa porque os dois desfechos são publicáveis, e o
registro impede racionalização posterior.

Há um sinal preliminar, dos dados contaminados: hard perdeu para soft em VGG11
(−2,78 p.p.) e empatou na CNN pequena (−0,29 p.p., ns). Se isso se confirmar em
dados limpos, a leitura é que **hard funciona quando há capacidade sobrando** —
BERT com 11,2M de parâmetros e duas tarefas — e falha sob capacidade apertada.
Isso conecta com o critério de plasticidade efetiva do eixo Qwen.

## Garantias de proveniência

Esta suíte nasce corrigindo o problema que torna 89 de 93 agregados não
citáveis:

- **recusa árvore Git suja** (`--allow-dirty` para sobrepor conscientemente);
- **protocolo congelado antes do primeiro passo**, em
  `hard_vs_soft_protocol.json`; re-executar com protocolo diferente no mesmo
  diretório é recusado;
- **seeds fixas**, sorteadas por gerador determinístico e gravadas antes;
- **recusa seeds reservadas** à confirmação congelada;
- **retomável**: alvo concluído é pulado.

## Execução

A fila espera o Qwen terminar, então roda em CPU:

```bash
./run_hard_vs_soft.sh              # aguarda o Qwen, depois executa
./run_hard_vs_soft.sh --no-wait    # executa já
```

Recomendado em segundo plano, resistente a desconexão:

```bash
nohup ./run_hard_vs_soft.sh > results/run_logs/hard_vs_soft/queue.log 2>&1 &
```

O script verifica a cada 5 minutos se ainda há processo
`experiments/qwen_iso_plasticity.py`, espera 60 s após o fim e executa com
`CUDA_VISIBLE_DEVICES=''` — não toca na GPU nem em nenhum artefato do Qwen.

Logs por alvo em `results/run_logs/hard_vs_soft/<dataset>_<backbone>.log`;
resultados em `results/hard_vs_soft/<dataset>_<backbone>/`.

## Depois da execução

1. Ler o contraste primário (Hard − Soft) nos cinco alvos.
2. Se hard vencer soft de forma consistente, o achado do Transformer é sobre
   **regime**, e o manuscrito deve dizer isso.
3. Se perder ou for nulo, o achado é **específico da arquitetura** — e essa é
   uma afirmação mais forte e mais interessante do que a atual.
4. Em qualquer caso, escrever a seção e atualizar `docs/arch_mlp.md`,
   `docs/arch_cnn.md` e `docs/arch_bert.md`.

Nenhum desses caminhos exige nova execução: os dois desfechos são reportáveis.
