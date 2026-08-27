# Protocolo confirmatório e suíte de baselines

Status: implementado; nenhum artefato confirmatório está versionado.

## Separação entre confirmação e exploração

A confirmação independente contém apenas replay e o candidato congelado. Seu
endpoint primário é a acurácia média final depois da quinta tarefa. As seeds,
configuração e análise estão em
`experiments/confirmatory_split_mnist.py`; a cópia legível do pré-registro está
em `configs/split_mnist_confirmation_preregistration.json`.

Baselines adicionais, fairness, custo, ordens, arquiteturas e ablações são
secundários. Eles usam seeds distintas no notebook para não consumir nem
influenciar as seeds confirmatórias.

## Três leituras de fairness

1. **Mesmas épocas:** todos os métodos principais usam dez épocas por tarefa.
   Replay de 20 épocas e replay com early stopping aparecem com nomes próprios,
   sem serem misturados à comparação de dez épocas.
2. **Mesmos exemplos:** `max_train_examples_per_task=20000` limita exatamente a
   soma de exemplos atuais e de replay consumidos pelo learner em cada tarefa.
3. **Tempo e FLOPs:** tempo observado e estimativas de FLOPs são reportados
   juntos. Forward do professor, hooks, regularização, consolidação e máscaras
   não são tratados como custo zero.

Tempo de parede depende da máquina e deve ser comparado somente dentro da mesma
execução. A estimativa usa duas operações por peso linear no forward e duas
vezes esse custo no backward, com overheads algorítmicos separados.

## Semântica resumida dos baselines

- **DER++:** cross-entropy atual, regressão dos logits armazenados e
  cross-entropy dos exemplos da memória.
- **ER-ACE:** loss atual restrita às classes da tarefa corrente e loss de replay
  sobre todas as classes vistas.
- **SlowHeat + ER-ACE:** aplica a mesma loss do ER-ACE à melhor configuração
  pré-registrada do SlowHeat (`beta=30`, budget `0,25`, proteção apenas nas
  camadas ocultas). A importância funcional é acumulada a partir da loss
  combinada e consolidada ao fim de cada tarefa.
- **A-GEM:** projeta o gradiente atual quando ele conflita com o gradiente de
  referência da memória.
- **EWC:** Fisher diagonal online estimada separadamente a partir da loss da
  tarefa corrente, sem contaminar a estimativa com a própria penalidade EWC;
  a penalidade quadrática permanece centrada no último ponto consolidado.
- **SI:** acumula contribuição gradiente-deslocamento durante a tarefa e produz
  uma importância sináptica na fronteira.
- **LwF calibrada:** distillation nas classes antigas com pesos determinados
  pela fração de classes antigas e novas.
- **Replay balanceado:** dá peso igual à loss do lote atual e à loss do replay,
  independentemente dos tamanhos dos dois lotes.

## Ablação exploratória SlowHeat + DER++

O notebook contém uma seção isolada para
`slowheat_derpp_hidden_beta_30_budget_0.25`. Ela usa a mesma memória, rótulos e
logits armazenados do DER++, com `alpha=0.5` e `beta=0.5`, enquanto o otimizador
SlowHeat protege somente as camadas ocultas. A importância funcional é medida
a partir da loss completa do DER++.

O teste compara replay, DER++, SlowHeat+replay e SlowHeat+DER++ nas seeds
secundárias. O contraste relevante é SlowHeat+DER++ menos DER++, acompanhado
por custo e pelo controle SlowHeat+replay menos replay. Essa seção é
exploratória e não modifica o pré-registro confirmatório.

## Execução

Abra `notebooks/split_mnist_confirmatory_suite.ipynb`, revise os diretórios e o
device, depois altere `RUN_EXPERIMENTS` para `True`. A célula confirmatória deve
ser executada uma única vez em um diretório vazio. O notebook mantém as etapas
separadas para impedir que uma ablação modifique o objeto congelado.

O equivalente não interativo pode ser inspecionado e executado com:

```bash
python run_all_tests.py --num-seeds 10 --sections confirmation --dry-run
python run_all_tests.py --num-seeds 10 --sections confirmation --device cpu
```

O runner cria o lock do pré-registro antes do treino e retoma seeds concluídas
apenas quando a configuração salva coincide. O runner também grava
`run_identity.json`, com hash do código e da configuração; uma mudança nesses
insumos exige um novo diretório e não pode reutilizar resultados anteriores.
Cada seed registra ainda `data_identity.json`, com o hash dos tensores de tarefa
efetivamente consumidos.
Use um novo `--output-dir` para
uma execução confirmatória independente; não use `--fresh` para sobrescrever
um diretório que já contenha resultados observados.

A seção de generalização oferece MLPs maiores,
Permuted-MNIST domain-incremental, Split-CIFAR-10 em cinco tarefas de duas
classes e Split-CIFAR-100 em dez tarefas de dez classes. Os dois protocolos
CIFAR são Class-IL sem task ID e usam imagens normalizadas e achatadas no engine
MLP pareado. Cada seção CIFAR executa os 32 métodos visuais implementados ou
configurados pelo projeto. As análises secundárias usam dez seeds pareadas; a
confirmação permanece com as vinte seeds congeladas.

Para executar o produto completo de datasets e métodos, use
`python run_all_tests.py --num-seeds 10 --all-datasets-all-methods`. Esse modo seleciona o
sintético com seus 11 métodos próprios e Split-MNIST, Permuted-MNIST,
Split-CIFAR-10 e Split-CIFAR-100 com os 32 métodos do engine visual. O sintético
permanece CPU-only.

O protocolo exato de dados, arquitetura, métodos e saídas de CIFAR está em
`docs/split_cifar.md`. Essas seções são secundárias e não alteram o endpoint
confirmatório congelado.
