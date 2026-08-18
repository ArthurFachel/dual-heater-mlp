# Protocolo confirmatório e suíte de baselines

Status: implementado, não executado.

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
- **A-GEM:** projeta o gradiente atual quando ele conflita com o gradiente de
  referência da memória.
- **EWC:** Fisher diagonal online estimado dos gradientes de treino e penalidade
  quadrática em torno do último ponto consolidado.
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

A seção de generalização oferece cinco ordens de Split-MNIST, MLPs maiores,
Permuted-MNIST domain-incremental e CORe50 New Classes. As análises secundárias
usam dez seeds pareadas; a confirmação permanece com as vinte seeds congeladas.
CORe50 usa as dez ordens oficiais, nove experiências de tamanhos
`10,5,5,5,5,5,5,5,5`, Class-IL sem task ID como endpoint principal e Replay,
DER++, SlowHeat+Replay e SlowHeat+DER++. Ele requer as imagens RGB recortadas e
os filelists oficiais `NC_inc`, com `download=False`. O protocolo detalhado
está em `docs/core50_class_il.md`.
