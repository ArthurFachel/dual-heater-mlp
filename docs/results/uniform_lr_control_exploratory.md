# R-A contra controle de lr uniforme — exploratório

**Natureza:** exploratório, NÃO pré-registrado. 10 seeds numa banda nova (13.000.003+), disjunta de todas as bandas em goals/ e docs/. Nada aqui é confirmatório.
**Execução:** 03/11/2026, 10,7 min, Titan Xp (GPU 2), worktree `/mnt/B-SSD/fachel/dh-uniform-ctrl` (HEAD aed2957 + patch abaixo). Repo principal do Lab não foi tocado.
**Artefatos:** `results/uniform_lr_control/` (diagnostic_summary.json, por seed), log em `results/uniform_lr_control.log`.

## Pergunta

R-A mostrou que a proteção dura ranqueada por `|z|` bate uma máscara aleatória do mesmo tamanho (+12,23 pp de retenção da T1, 10/10). Máscara aleatória fixa a CONTAGEM protegida, mas não separa "proteger as unidades certas" de "remover plasticidade". O braço novo responde isso.

## Braço novo: `slowheat_uniform_hard`

`experiments/split_clinc150.py`, classe `UniformPlasticity`, modo `plasticity_mask_mode="uniform_hard"`:
- toda entrada coberta pela máscara recebe o mesmo fator de update, igual à média da máscara dura sobre a superfície coberta;
- recalculado em cada consolidação, no mesmo ponto em que os braços duros trocam a máscara;
- mesma plasticidade de superfície dos braços duros, sem seleção e sem nada congelado.

Verificado no smoke (1 seed, 1 época): nível = 0,25 (2.364.672 de 3.152.896 entradas protegidas nos braços duros, igual ao R-A); nos braços duros o drift das protegidas é exatamente 0; no uniforme, todas as entradas se movem com RMS ~1/4 do drift plástico do braço duro. 65 testes existentes passam com o patch.

## Resultado (10 seeds, validação)

| braço | T1 aquis. | T1 retenção | T2 aquis. | FAA |
|---|---:|---:|---:|---:|
| vanilla | 0,8270 | 0,0983 | 0,8440 | 0,4712 |
| máscara aleatória | 0,8270 | 0,3433 | 0,7470 | 0,5452 |
| lr uniforme | 0,8270 | 0,4027 | 0,7343 | 0,5685 |
| SlowHeat `\|z\|` | 0,8270 | **0,4417** | 0,7083 | **0,5750** |

Contrastes pareados, teste de sinal exato bilateral (sem correção; exploratório):

| contraste | retenção T1 | FAA | aquisição T2 |
|---|---|---|---|
| SlowHeat − aleatório | +9,83 pp, 10+/0−, p=0,002 | +2,98 pp, 10+/0−, p=0,002 | −3,87 pp, 2+/8− |
| lr uniforme − aleatório | +5,93 pp, 10+/0−, p=0,002 | +2,33 pp, 10+/0−, p=0,002 | −1,27 pp, 2+/8− |
| **SlowHeat − lr uniforme** | **+3,90 pp, 10+/0−, p=0,002** | +0,65 pp, 6+/4−, p=0,75 | −2,60 pp, 2+/7− |

## Leitura

1. **R-A replica** em seeds novas: SlowHeat − aleatório = +9,83 pp de retenção, 10/10 (original +12,23 pp).
2. **A máscara aleatória é um controle fraco.** Remover a mesma plasticidade de forma uniforme já bate a máscara aleatória por +5,93 pp de retenção, 10/10. Congelar 75% de unidades sorteadas é pior do que só reduzir o passo de todas.
3. **A seleção sobrevive no endpoint de retenção:** SlowHeat − lr uniforme = +3,90 pp, 10/10. Cerca de 40% do ganho do R-A sobre o aleatório vem da seleção; os outros ~60% são explicados por remover plasticidade.
4. **Mas na acurácia final a seleção empata com o lr uniforme** (+0,65 pp, 6/4). O SlowHeat compra retenção da T1 aprendendo menos a T2 (−2,60 pp). Superfície igual não é plasticidade efetiva igual: o braço duro protege justamente as unidades que a T2 também queria usar.

## Consequência para o paper

- A frase do R-A precisa mudar de "seleção bate aleatório por +12 pp" para "seleção bate um controle de plasticidade igual por ~+4 pp de retenção, sem ganho em FAA".
- Para virar resultado, isto precisa de pré-registro (banda nova, endpoint primário retenção T1, contraste SlowHeat − lr uniforme) e as 10 seeds confirmatórias.
- Limites: um host (BERT-mini), um benchmark, duas tarefas, validação apenas.

## Patch

`git diff` no worktree mostra as mudanças em `experiments/split_clinc150.py` (classe `UniformPlasticity`, modo `uniform_hard`, refresh na consolidação, `uniform_plasticity_history` no resultado). Runner: `scripts/run_uniform_lr_control.py`.
