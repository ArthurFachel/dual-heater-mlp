# Auditoria: quais resultados têm controle de plasticidade igual

**Data:** 03/11/2026. **Natureza:** leitura de artefatos e documentos, sem rodar nada novo.
**Gatilho:** `docs/results/uniform_lr_control_exploratory.md`. Em BERT, reduzir a plasticidade de forma uniforme (mesma superfície, sem seleção) já bate a máscara aleatória do mesmo tamanho por +5,93 pp de retenção da T1, 10/10. Ou seja: "bate aleatório" mistura seleção com remoção de plasticidade, e a máscara aleatória sozinha não é controle suficiente.

**Critério:** um resultado do tipo "proteção ajuda" só isola o método se houver um braço que remove a MESMA plasticidade sem selecionar unidades (lr uniforme pareado pela superfície, ou `lr_control` pareado). Contra `vanilla` ou contra máscara aleatória, não isola.

| # | resultado | contraste publicado | tem controle de plasticidade igual? | o que muda |
|---|---|---|---|---|
| 1 | Split-MNIST confirmação | SlowHeat+Replay − Replay = +0,87 pp, 17/20 | **não**. Existe `replay_global_lr_reduction` (ablações, 20 seeds), mas com lr × 1/31 em todos os parâmetros, saída incluída. O SlowHeat `hidden_beta_30_budget_0.25` só age nas camadas ocultas e deixa pelo menos 25% das unidades com lr cheio, então remove bem menos plasticidade. Não é pareado (FAA 0,671, abaixo dos dois). Valor exato da plasticidade do SlowHeat aqui não medido nesta auditoria. | Falta Replay com lr uniforme pareado (~0,27). Barato: MLP, minutos. Sem isso, os +0,87 pp podem ser só lr menor. |
| 2 | BERT R-A (confirmatório) | magnitude − aleatório = +12,23 pp retenção T1 | **agora sim, exploratório**: SlowHeat − lr uniforme = +3,90 pp retenção (10/10), FAA +0,65 pp (6/4, n.s.) | A frase muda para ~+4 pp de retenção sobre controle pareado, sem ganho de FAA. Precisa de pré-registro. Curva por orçamento rodando (`results/budget_curve/`). |
| 3 | BERT R-B | funcional − magnitude = −0,17 pp | não se aplica: os dois braços selecionam, com a mesma plasticidade | Nada. |
| 4 | Permuted-MNIST hard (`exploratory_reanalysis`) | hard − vanilla = +16,99 pp, 10/10 | **não**. Só contra vanilla. | É o caso mais exposto. No spike em MLP (`~/Projects/dualheat-local-rules-spike`, Spike 6), lr uniforme sozinho explicou quase todo o ganho no Permuted-MNIST. Não citar sem controle. |
| 5 | Qwen+LoRA R-E (confirmatório) | `frozen_a_control − lr_control`, superfície 0,62246 nos dois | **sim** | Nada. É o resultado mais bem controlado do projeto. |
| 6 | Qwen+LoRA R-C | `exact − frozen_a_control` = +0,69 pp, 5/5 | sim, implicitamente (frozen_a é o controle) | Nada; já é negativo para a máscara. |
| 7 | Penalidade passada 2 (confirmatório) | `mas − lr_control`, lr_scale congelado | **sim** | Nada, além da ressalva de quase-chance já registrada. |
| 8 | SlowHeat+DER++ (exploratório) | +3,14 pp sobre DER++ | **não** | Mesmo problema do item 1. |
| 9 | Hard vs Soft | hard − soft | não se aplica: compara dois jeitos de proteger | Nada. |
| 10 | Split-CIFAR, CNN/VGG/ResNet | SlowHeat ≈ vanilla ou abaixo | não se aplica: não há ganho a explicar | Nada. |
| 11 | Seletor de replay (R-D) | representativo − first | não se aplica: não é sobre SlowHeat | Nada. |
| 12 | NCP, máscara dura de atenção | SlowHeat − aleatório e − uniforme (lr × 0,75 em todas as cabeças) | **sim**, segundo a nota do vault de 02-03/11 (fora deste repo, `/mnt/B-SSD/fachel/ncp`; não reconferido no artefato nesta auditoria) | Nada; efeito pequeno (~1,3% do esquecimento) e já declarado. |

## Achado lateral (já estava nos dados)

Nas ablações de Split-MNIST (`results/protocol_post_eval_fix_d5b22ad/ablations/methods/aggregate.json`, 20 seeds), `replay_calibrated` chega a FAA 0,8276, contra 0,7767 do SlowHeat+Replay e 0,7654 do Replay. Uma calibração de offset entre classes antigas e novas, sem proteção nenhuma, rende cerca de 6× o ganho do SlowHeat sobre Replay. Vale citar como baseline barato e forte no paper.

## Ações

1. **Item 1:** rodar Replay com lr uniforme pareado à superfície do SlowHeat (fator = média de `1/(1+β·heat)` medida, não 1/31), 20 seeds, mesmas seeds da confirmação. Minutos de CPU/GPU.
2. **Item 4:** não citar o +16,99 pp sem o mesmo controle.
3. **Item 2:** pré-registrar SlowHeat − lr uniforme depois de ler a curva por orçamento.
4. **Item 8:** idem item 1 se o número for usado em algum lugar.
