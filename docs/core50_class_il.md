# CORe50 New Classes / Class-IL

## Motivação

CORe50 foi criado especificamente para Continual Object Recognition. Ele contém
50 objetos domésticos, organizados em 10 categorias e capturados em 11 sessões
com mudanças de fundo, iluminação, pose, mão e oclusão. O conjunto completo tem
164.866 imagens RGB-D; o protocolo deste projeto usa somente as imagens RGB
recortadas de 128×128.

Fonte primária: [Lomonaco e Maltoni, CoRL 2017](https://proceedings.mlr.press/v78/lomonaco17a.html).
Dados, filelists e documentação: [site oficial do CORe50](https://vlomonaco.github.io/core50/).

## Protocolo implementado

O adapter segue o cenário oficial **NC incremental** em modo Class-IL:

- classificação no nível de objeto, com 50 classes;
- 9 experiências: 10 classes na primeira e 5 classes em cada uma das oito
  seguintes;
- 10 ordens oficiais, representadas por `Run0` até `Run9`;
- filelists oficiais `NC_inc`, incluindo o remapeamento de labels de cada run;
- teste baseado nas sessões fixas definidas pelo benchmark;
- uma única cabeça de 50 classes e nenhum task ID na inferência Class-IL;
- classes futuras mascaradas até serem apresentadas;
- Task-IL reportado somente como diagnóstico;
- Replay, DER++, SlowHeat+Replay e SlowHeat+DER++;
- contrastes pareados contra Replay e DER++.

Para manter o custo compatível com a suíte MLP, cada run usa uma subamostra
determinística por classe: 400 imagens para treino, 100 para validação e 100
para teste. As imagens são redimensionadas para 64×64 e normalizadas com as
estatísticas usuais do ImageNet. Portanto, a estrutura NC e as dez ordens são
oficiais, mas os números não pretendem reproduzir a tabela do artigo, que usa
outra arquitetura e receita de treino.

## Estrutura dos dados

Passe a raiz das imagens RGB recortadas em `--core50-dir`. O runner aceita os
filelists dentro da raiz ou no diretório pai:

```text
core50_128x128/
├── s1/
│   ├── o1/
│   └── ...
├── ...
├── s11/
└── batches_filelists/
    └── NC_inc/
        ├── Run0/
        │   ├── train_batch_00_filelist.txt
        │   ├── ...
        │   ├── train_batch_08_filelist.txt
        │   └── test_filelist.txt
        └── ... Run9/
```

Os arquivos necessários são `cropped_128x128_images.zip` e
`batches_filelists.zip`, disponibilizados no site oficial. O projeto não faz o
download automaticamente.

## Execução

Plano sem treinamento:

```bash
python run_all_tests.py --sections core50 --dry-run
```

Execução das dez ordens oficiais:

```bash
PYTHONPATH=src:. python run_all_tests.py \
  --sections core50 \
  --device cuda \
  --core50-dir /path/to/core50_128x128 \
  --output-dir results/core50_four_methods
```

Os artefatos são separados em `seed_0/` até `seed_9/`. O nome `seed` é
mantido pela infraestrutura compartilhada, mas nesses diretórios o valor
também identifica a ordem oficial `Run0`–`Run9`.
