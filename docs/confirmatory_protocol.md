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

## Execução

Abra `notebooks/split_mnist_confirmatory_suite.ipynb`, revise os diretórios e o
device, depois altere `RUN_EXPERIMENTS` para `True`. A célula confirmatória deve
ser executada uma única vez em um diretório vazio. O notebook mantém as etapas
separadas para impedir que uma ablação modifique o objeto congelado.

A seção de generalização oferece cinco ordens de Split-MNIST, MLPs maiores,
Permuted-MNIST domain-incremental, Split CIFAR-100 e TinyImageNet. O último
requer `train/` e `val/` locais em estrutura ImageFolder e deve ser chamado com
`download=False`.
