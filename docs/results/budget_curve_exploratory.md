# Curva retenção x aquisição por orçamento — exploratório

**Natureza:** exploratório, NÃO pré-registrado. 10 seeds, banda 13.500.003+ (disjunta da run anterior, 13.000.003+). Orçamentos de plasticidade 0,1 / 0,25 / 0,5 / 0,75 (fração de unidades plásticas; 0,1 = protege 90%). FFN e atenção com o mesmo orçamento.
**Execução:** 03/11/2026, 43,4 min, Titan Xp. Runner `scripts/run_budget_curve.py`, análise `scripts/analyze_budget_curve.py`. Artefatos em `results/budget_curve/`.
**Sanidade:** o braço vanilla, rodado de novo em cada orçamento, deu valores idênticos nos 4 (determinismo).

## Médias (10 seeds, validação)

| orçamento | braço | retenção T1 | aquisição T2 | FAA |
|---|---|---:|---:|---:|
| 0,1 | aleatória | 0,4207 | 0,6873 | 0,5540 |
| 0,1 | lr uniforme | 0,4907 | 0,6577 | **0,5742** |
| 0,1 | SlowHeat `\|z\|` | 0,4907 | 0,6523 | 0,5715 |
| 0,25 | aleatória | 0,3107 | 0,7437 | 0,5272 |
| 0,25 | lr uniforme | 0,3530 | 0,7233 | 0,5382 |
| 0,25 | SlowHeat `\|z\|` | 0,3987 | 0,7030 | 0,5508 |
| 0,5 | aleatória | 0,2123 | 0,8000 | 0,5062 |
| 0,5 | lr uniforme | 0,2247 | 0,7823 | 0,5035 |
| 0,5 | SlowHeat `\|z\|` | 0,2750 | 0,7673 | 0,5212 |
| 0,75 | aleatória | 0,1447 | 0,8253 | 0,4850 |
| 0,75 | lr uniforme | 0,1463 | 0,8207 | 0,4835 |
| 0,75 | SlowHeat `\|z\|` | 0,1817 | 0,8183 | 0,5000 |

## SlowHeat − lr uniforme no mesmo orçamento (pp, sinais, teste de sinal sem correção)

| orçamento | retenção T1 | aquisição T2 | FAA |
|---|---|---|---|
| 0,1 | −0,00 (5/5) | −0,53 (3/7) | −0,27 (5/5) |
| 0,25 | +4,57 (9/1, p=0,021) | −2,03 (1/9, p=0,021) | +1,27 (7/3) |
| 0,5 | +5,03 (10/0, p=0,002) | −1,50 (3/7) | +1,77 (9/1, p=0,021) |
| 0,75 | +3,53 (10/0, p=0,002) | −0,23 (3/7) | +1,65 (10/0, p=0,002) |

O ponto 0,25 replica a run anterior em seeds novas (+3,90 pp lá, +4,57 pp aqui).

## Na mesma aquisição da T2 (fronteira)

Retenção do SlowHeat contra a curva do lr uniforme interpolada linearmente na mesma aquisição da T2 (médias, sem estatística por seed):

| orçamento | aquisição T2 | SlowHeat | lr uniforme interpolado | diferença |
|---|---:|---:|---:|---:|
| 0,1 | 0,6523 | 0,4907 | fora da faixa | — |
| 0,25 | 0,7030 | 0,3987 | 0,3956 | +0,30 pp |
| 0,5 | 0,7673 | 0,2750 | 0,2573 | +1,77 pp |
| 0,75 | 0,8183 | 0,1817 | 0,1511 | +3,06 pp |

## Leitura

1. **A seleção desloca a fronteira, mas pouco.** Na mesma aquisição da T2, o SlowHeat retém de +0,3 a +3,1 pp a mais que o lr uniforme. Boa parte da vantagem no mesmo orçamento (+3,5 a +5,0 pp) vem de o SlowHeat ser menos plástico que o uniforme naquele orçamento, não de selecionar melhor.
2. **A seleção vale mais quando a proteção é leve.** Protegendo 25% (orçamento 0,75), ela ganha +3,1 pp na fronteira e +1,65 pp de FAA (10/10). Protegendo 90% (0,1), não vale nada: empata em tudo.
3. **O melhor FAA da curva inteira é do lr uniforme com proteção pesada** (0,5742 no orçamento 0,1, SlowHeat 0,5715). Quem quer só acurácia final não precisa de seleção nenhuma: basta reduzir o passo.
4. **A máscara aleatória é o pior braço em quase toda a curva**, o que confirma que ela é um controle fraco.

## Consequência

- O claim defensável fica: "com proteção leve, a seleção por `|z|` desloca a fronteira retenção x aquisição em ~2-3 pp sobre um controle de plasticidade igual". Não: "a seleção melhora o resultado final", porque o melhor FAA não depende dela.
- Um pré-registro honesto usa como endpoint a retenção na mesma aquisição (fronteira), ou FAA no orçamento 0,75, com o lr uniforme como controle. O endpoint precisa ser escolhido sabendo que estes números são exploratórios.
- Limites: BERT-mini, CLINC150, duas tarefas, validação apenas, interpolação feita nas médias.
