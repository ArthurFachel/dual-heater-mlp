# Protocolo confirmatório e suíte de baselines

Status: implementado e **executado**. Duas execuções independentes de 20 seeds
estão em disco sob `results/protocol_post_eval_fix/confirmation/` e
`results/protocol_post_eval_fix_d5b22ad/confirmation/`. O resultado está na
seção [Resultado confirmatório](#resultado-confirmatório-20-seeds).

Ressalva de proveniência: ambas as execuções registram `git dirty = true`. O
diff executado não foi fingerprintado, portanto não há reprodução bit a bit a
partir da proveniência isoladamente.

## Correção de implementação de 2026-09-02

Uma auditoria do runner encontrou que as avaliações realizadas ao fim de cada
época chamavam `model.eval()` sem restaurar o modo anterior. Como os hooks de
importância funcional só acumulam em modo de treino, execuções SlowHeat sem um
método FastHeat na mesma lista acumulavam importância apenas na primeira época
de cada estágio. A presença de FastHeat ativava uma restauração global no início
das épocas seguintes e, portanto, fazia o resultado de um método depender dos
outros métodos incluídos no sweep.

O protocolo confirmatório congelado usa dez épocas e somente `replay` e
`slowheat_replay_hidden_beta_30_budget_0.25`; logo, qualquer execução dele feita
antes dessa correção foi afetada e não deve ser usada como resultado
confirmatório. Seeds, hiperparâmetros, endpoint e plano de análise permanecem
inalterados. A correção apenas restaura a semântica pretendida do treinamento e
é protegida por testes de independência da composição do sweep.

Uma execução confirmatória posterior à correção deve usar um diretório novo. O
`source_fingerprint` impede que checkpoints ou seeds produzidos pelo código
anterior sejam retomados como se pertencessem à implementação corrigida.

As duas execuções versionadas satisfazem essa exigência. O módulo
`experiments/evaluation.py`, que implementa o contexto de avaliação sem efeito
colateral, foi introduzido no commit `d5b22ad` (2026-09-03) — exatamente o
commit registrado em
`results/protocol_post_eval_fix/confirmation/environment.json`. A segunda
execução usa `f2f7616` (2026-09-04). Ambas são posteriores à correção e ambas
gravaram em diretórios de saída novos.

## Resultado confirmatório, 20 seeds

Contraste congelado: `slowheat_replay_hidden_beta_30_budget_0.25` menos
`replay`. Endpoint primário: acurácia média final class-incremental após a
quinta tarefa. Pré-registro em
`results/protocol_post_eval_fix*/confirmation/preregistration.lock.json`, com
`status = frozen_before_execution`, `preregistered_at = 2026-08-15` e
`sha256 = 015b3162…a9b2` idêntico nas duas execuções, o que prova que o
pré-registro não foi editado entre elas.

Médias marginais das 20 seeds:

| Métrica | Replay | SlowHeat + Replay |
|---|---:|---:|
| Acurácia média final | 0,77040 | 0,77909 |
| Average forgetting | 0,27244 | 0,25781 |
| Classifier gap | 0,21293 | 0,20558 |
| Task-aware final | 0,98333 | 0,98467 |
| Tempo (s) | 3,036 | 5,828 |

Diferenças pareadas por seed, sempre candidato menos referência:

| Métrica | Diferença média | IC95% t (gl=19) | t | p bilateral | Sinais | Bootstrap IC95% |
|---|---:|---:|---:|---:|---|---:|
| **Acurácia média final** | **+0,00869** | [+0,00293; +0,01445] | 3,157 | **0,00520** | 17+ / 3− | [+0,00316; +0,01381] |
| Average forgetting | −0,01463 | [−0,02207; −0,00718] | −4,112 | 0,00059 | 2+ / 18− | [−0,02118; −0,00748] |
| Classifier gap | −0,00735 | [−0,01304; −0,00166] | −2,704 | 0,01408 | 4+ / 16− | [−0,01242; −0,00195] |
| Task-aware final | +0,00134 | [+0,00005; +0,00263] | 2,168 | 0,04309 | 14+ / 6− | [+0,00019; +0,00251] |
| Tempo (s) | +2,7923 | [+2,7491; +2,8356] | 135,06 | <1e−6 | 20+ / 0− | [+2,7543; +2,8339] |

O endpoint primário é positivo e significativo: **+0,869 pontos percentuais**,
`p = 0,0052`, com 17 das 20 seeds favoráveis. O forgetting cai 1,46 p.p. com
`p = 0,00059`. O custo é a contrapartida: o tempo quase dobra, com efeito
presente em todas as 20 seeds.

O ganho de acurácia task-aware (`p = 0,043`, teste de sinais `p = 0,115`) é
fraco e não sobrevive a nenhuma correção de multiplicidade. Somente o endpoint
primário foi pré-registrado; as demais linhas são secundárias e descritivas.

### Replicação entre as duas execuções

Comparei os 20 `results.json` par a par entre os dois diretórios. Dos 100
campos escalares, 21 diferem e **todos são de custo** (`elapsed_seconds`,
`optimizer_step_seconds`, `peak_memory_*`, `selection_seconds`). Nenhuma
métrica científica difere. Os arquivos têm hashes SHA-256 distintos, ou seja,
são duas execuções independentes, em commits diferentes, que reproduziram
resultados numericamente idênticos.

### O que este resultado não estabelece

- Não é superioridade geral do SlowHeat sobre métodos de continual learning.
  O contraste é contra `replay` simples, num único benchmark.
- Não cobre SlowHeat + DER++, que permanece exploratório e sem pré-registro
  próprio.
- Não vale como comparação de custo: o overhead de tempo é grande e
  sistemático.
- A árvore Git suja impede reprodução bit a bit pela proveniência.

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
`docs/protocols/split_cifar.md`. Essas seções são secundárias e não alteram o endpoint
confirmatório congelado.
