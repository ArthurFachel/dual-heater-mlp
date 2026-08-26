# Método versus método + DualHeat

## Pergunta e escopo

**Quanto o componente acrescenta a cada método de treinamento existente?**

Nesta suíte, DualHeat é o nome da proposta modular; a implementação utilizada
é **Functional SlowHeat** (`SlowHeatMLP` + `SlowHeatAdamW`). A classe legada
`DualHeatMLP` não é avaliada e não foi renomeada. Não misturar seus resultados.

O protocolo é exploratório. Ele não substitui, amplia ou transforma em
confirmação independente o protocolo congelado de Replay em
`configs/split_mnist_confirmation_preregistration.json`.

## Quatro pares, oito execuções por seed

| Método original | Identificador com componente |
|---|---|
| `vanilla` (treinamento convencional com AdamW) | `slowheat_hidden_beta_30_budget_0.25` |
| `replay` | `slowheat_replay_hidden_beta_30_budget_0.25` |
| `derpp` | `slowheat_derpp_hidden_beta_30_budget_0.25` |
| `er_ace` | `slowheat_er_ace_hidden_beta_30_budget_0.25` |

Não é uma disputa de DualHeat isolado contra os quatro métodos. O contraste
de interesse é sempre **componente + método − o mesmo método**.

Todos os candidatos usam importância funcional `|z * dL/dz|`, `beta=30`,
budget de plasticidade `0.25`, proteção somente das unidades ocultas,
proteção fatorada das linhas de entrada e colunas de saída, consolidação MAX
e política de estado do otimizador `follow_update`. O classificador não tem
importância própria protegida, mas suas colunas podem receber a máscara das
unidades ocultas anteriores. Os nomes fixam beta/budget mesmo se os defaults
genéricos do runner forem diferentes.

São escolhas herdadas da exploração em Split-MNIST, não hiperparâmetros ótimos
demonstrados para todos os datasets. Não ajustar escolhas nos resultados de
teste nem selecionar somente os pares favoráveis para o artigo.

## Condições comparáveis

- Mesma arquitetura MLP/ReLU e parâmetros treináveis iniciais, byte a byte.
- Mesmas seeds, divisões de treino/validação/teste e ordem de tarefas.
- Mesmos índices de minibatches e replay dentro de cada par. Em DER++, os
  valores dos logits armazenados podem divergir após o treinamento, como
  consequência do componente; o procedimento de armazenamento é o mesmo.
- Mesmas épocas, learning rate, weight decay, tamanho de memória e losses do
  método base. Sem busca individual de hiperparâmetros nesta primeira suíte.
- Fronteiras de tarefa conhecidas para consolidar; avaliação principal sem
  task ID. Acurácia task-aware é somente um diagnóstico.

O relatório verifica por seed igualdade de parâmetros treináveis, épocas,
passos, exemplos atuais e de replay, bytes de replay e logits. Rejeita pares
incompletos, seeds repetidas ou configurações incompatíveis. Esses controles
igualam a exposição dentro de cada par, **não** FLOPs, tempo ou memória total.
Métodos diferentes podem consumir quantidades diferentes de exemplos.

## Executar

Split-MNIST pelo ponto de entrada principal:

```bash
python run_all_tests.py --num-seeds 10 --sections dualheat-pairs --dry-run
python run_all_tests.py --num-seeds 10 --sections dualheat-pairs --device cpu --no-download
```

Saída: `results/split_mnist_protocol/dualheat_pairs/`.

Para escolher os datasets, use a entrada dedicada:

```bash
PYTHONPATH=src:. python -m experiments.dualheat_pairs \
  --datasets split_mnist permuted_mnist split_cifar10 split_cifar100 \
  --num-seeds 10 --device cuda --dry-run
```

Remova `--dry-run` para treinar; acrescente `--no-download` se os dados já
estiverem disponíveis. Essa execução completa pode ser cara. Comece com um
dataset. CIFAR usa imagens achatadas em MLP; esta suíte não é um benchmark CNN.
As configurações completas aparecem no dry-run e no `pair_protocol.json`.

A entrada dedicada salva cada dataset em
`results/dualheat_pairs/<dataset>/`. Ela gera seeds reproduzíveis próprias;
o runner principal mantém seu gerador existente. Para obter exatamente as
mesmas seeds entre entradas, forneça `--seeds` na dedicada ou
`--baseline-seeds` no runner principal, junto de `--num-seeds`.

A primeira execução salva o protocolo antes do treino. Uma retomada exige o
mesmo protocolo e configurações por seed. Alterações de seeds, arquitetura ou
treinamento requerem outro `--output-dir`. As seeds da confirmação congelada
são recusadas em novas execuções desta suíte.

## Reanalisar resultados existentes sem treinar

```bash
PYTHONPATH=src:. python -m experiments.dualheat_pairs \
  --summarize-from results/split_mnist_protocol/split_mnist_all_methods \
  --output-dir results/dualheat_pairs_existing/split_mnist
```

O comando lê `multi_seed_config.json` e os arquivos `config.json` e
`results.json` de cada seed declarada. Não usa ICs marginais para inferir
diferenças, não descarta seeds ausentes e não altera arquivos brutos. Um
diretório histórico pode conter outros métodos além dos oito necessários.
A análise registra hashes SHA-256 das entradas; isso identifica os arquivos,
mas não recupera proveniência ausente nem comprova que nunca foram observados.

## Saídas e análise

- `pair_summary.csv`: uma linha por par, com acurácias, diferença em pontos
  percentuais, IC95% t pareado, p ajustado, forgetting e razão dos tempos médios.
- `pair_differences.csv`: valores sem/com e diferenças de cada métrica por seed.
- `pair_report.json`: estatística completa, bootstrap pareado, sinais
  positivos/negativos/empates, configurações e hashes de origem.
- `pair_report.md`: tabela legível, incluindo ganhos negativos e limitações.

O endpoint principal é a acurácia média final. O ajuste de Holm cobre os
quatro testes de acurácia dentro de cada dataset; os IC95% t e bootstrap são
pontuais, não simultâneos. Outras métricas e comparações entre datasets são
exploratórias. Uma seed gera somente descrição, sem IC ou p inferencial.
Mesmo com várias seeds, significância não equivale a relevância prática ou
generalização: as seeds não são amostras de todos os possíveis problemas.

O tempo é o observado pelo runner, com ordem fixa, sem benchmark isolado com
aquecimento e sem garantia de sincronização explícita para medições CUDA.
FLOPs são estimativas. Memória de replay/logits **não** mede o estado adicional
do componente, as cópias temporárias ou o pico de memória; esse pico não é
medido aqui. Não apresentar esses números como prova de eficiência geral.

## Critério para ampliar o artigo

Examinar se o ganho se repete entre métodos e datasets, preservando aquisição
e considerando custo. Mostrar também onde o componente prejudica. Acrescentar
um método exige auditar sua integração, manter sua loss e procedimento de
treinamento e testar seu par; adicionar um prefixo ao nome não basta.

Essa matriz avalia quatro integrações concretas. Ela não prova compatibilidade
com qualquer MLP, qualquer otimizador ou qualquer cenário de aprendizado.
