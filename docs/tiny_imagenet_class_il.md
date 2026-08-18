# Sequential Tiny ImageNet Class-IL

## Por que é a evidência visual mais forte desta suíte

Tiny ImageNet combina 200 classes, imagens visualmente mais diversas que os
benchmarks MNIST, dez tarefas disjuntas e interferência acumulada ao longo de
uma sequência maior. Em Class-IL, a inferência não recebe o identificador da
tarefa: depois do estágio final, o modelo precisa escolher globalmente entre
as 200 classes aprendidas. A matriz Task-IL, que restringe a decisão às vinte
classes da tarefa conhecida, é reportada somente para localizar erro de
classificador e não substitui o endpoint Class-IL.

O protocolo de tarefas segue Buzzega et al., *Dark Experience for General
Continual Learning: a Strong, Simple Baseline* (NeurIPS 2020): Sequential Tiny
ImageNet contém dez tarefas e cada tarefa introduz vinte classes. O artigo
avalia Class-IL e Task-IL e reporta resultados sobre dez execuções. Fonte
primária: [artigo nos Proceedings of NeurIPS](https://proceedings.neurips.cc/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf).

## Protocolo implementado

- 200 classes em ordem fixa `0..199`;
- 10 tarefas sequenciais de 20 classes, sem sobreposição;
- uma única cabeça de saída com 200 logits;
- treino e avaliação Class-IL mascaram somente classes futuras, nunca classes
  antigas já aprendidas;
- nenhum task ID é fornecido à inferência Class-IL;
- comparação pareada entre `replay` e `derpp`;
- mesmas inicializações, divisões, minibatches, exemplares e índices de replay
  por seed;
- 450 imagens de treino e 50 de validação por classe, reproduzindo a separação
  de 10% do conjunto de treino descrita no artigo;
- 10 seeds por padrão nas análises secundárias;
- acurácia média final Class-IL como leitura principal; Task-IL e o
  `classifier_gap` são diagnósticos secundários.

Esta implementação reutiliza deliberadamente o motor MLP do projeto para
isolar Replay × DER++ sob as mesmas condições usadas nas demais análises. Ela
reproduz a divisão e a semântica incremental do artigo, mas não pretende
reproduzir numericamente sua tabela: o trabalho original usa ResNet-18 não
pré-treinada e uma receita de otimização própria.

## Dados e execução

O diretório deve estar preparado para `torchvision.datasets.ImageFolder`:

```text
tiny-imagenet-200/
├── train/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01443537/
    └── ...
```

A distribuição original deixa as imagens de validação em uma pasta única;
elas precisam ser reorganizadas em subpastas por classe antes da execução.
`train/` e `val/` devem conter exatamente o mesmo mapeamento de classes.

Para validar o plano sem treinar:

```bash
python run_all_tests.py --sections tiny-imagenet --dry-run
```

Para executar ou retomar as dez seeds:

```bash
python run_all_tests.py \
  --sections tiny-imagenet \
  --device cuda \
  --tiny-imagenet-dir /path/to/tiny-imagenet-200
```

Os artefatos são gravados em
`results/split_mnist_protocol/tiny_imagenet/`. `aggregate.json` contém o
contraste pareado `derpp - replay`; cada diretório `seed_*` contém as matrizes
Class-IL e Task-IL completas.
