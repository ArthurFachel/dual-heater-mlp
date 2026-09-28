# Ablação do critério de importância — resultado

**Pré-registro:** `goals/protocol_importance_criterion_ablation.md` (congelado em
`a3f5117`, antes de qualquer seed).
**Execução:** 28/09/2026, 10 seeds, 10,9 min, GTX 1080 Ti.
**Artefatos:** `results/criterion_ablation/`.

---

## Pergunta

O SlowHeat ranqueia unidades por importância funcional de primeira ordem,
`|z · dL/dz|` (`src/dual_heater/slow_heat.py:192`). Isso exige um hook de
backward e custa mais que medir só a magnitude da ativação, `|z|`.

A hipótese ordenada, congelada antes de rodar:

```
aleatório  <  magnitude |z|  <  funcional |z · dL/dz|
```

A segunda desigualdade é a que justifica o componente do gradiente.

## Desenho

Quatro braços em BERT + Split-CLINC150, todos compartilhando método
(`slowheat_ffn_attention`), `slow_strength = 3.0` e orçamento de capacidade. A
única variável é o critério de ranking.

| braço | critério |
|---|---|
| `vanilla` | sem proteção |
| `slowheat_random_hard` | aleatório |
| `slowheat_magnitude_hard` | `\|z\|` |
| `slowheat_hard` | `\|z · dL/dz\|` |

Os três braços protegidos protegem **2.364.672 entradas cada** — o pareamento
de capacidade foi verificado, não assumido.

Família confirmatória de 2, Holm, teste de sinal exato bilateral, endpoint
primário retenção da tarefa 1, 10 seeds na banda 4.000.003+ (disjuntas das
já gastas no BERT).

## Resultado

| contraste | média | mediana | sinais | p | p Holm |
|---|---:|---:|---:|---:|---:|
| `magnitude − aleatório` | **+12,23 pp** | +13,67 pp | **10+/0** | 0,00195 | **0,00391** ✓ |
| `funcional − magnitude` | −0,17 pp | +0,33 pp | 5+/5 | 1,000 | 1,000 ✗ |

Endpoints absolutos (retenção T1 / FAA):

| braço | retenção T1 | FAA |
|---|---:|---:|
| `slowheat_magnitude_hard` | **44,23%** | 57,23% |
| `slowheat_hard` (funcional) | 44,07% | 57,62% |
| `slowheat_random_hard` | 32,00% | 53,72% |
| `vanilla` | 8,87% | 46,75% |

**A primeira aresta da hipótese existe e é forte. A segunda não existe.**

## O empate não é artefato

A §F do pré-registro declarou um falsificador: se o ranking de magnitude fosse
quase plano, `_apply_capacity_budget` escolheria o top-k por ruído e o braço
degeneraria numa máscara quase-aleatória — um empate seria então artefato, não
achado. O protótipo pré-implementação tinha medido variância 0,028 contra
1,159, o que tornava o risco concreto.

Variância do ranking normalizado, medida nas 10 seeds:

| braço | variância |
|---|---:|
| `slowheat_hard` (funcional) | 2,9124e-03 |
| `slowheat_magnitude_hard` | 2,9070e-03 |
| `slowheat_random_hard` | 3,1250e-03 |

Diferença de **0,2%** entre funcional e magnitude. Os dois rankings são
igualmente informativos; o braço `magnitude` foi genuinamente exercido. O
empate é real.

## Interpretação

O componente do gradiente — a parte cara do critério, e a que distingue o
SlowHeat de métodos baseados em ativação — **não compra nada neste host**.
Medir apenas `|z|` produz o mesmo resultado.

Isso não invalida a proteção seletiva: ela bate a máscara aleatória com
+12,23 pp em 10/10 seeds. O que o resultado diz é que *o que* se mede para
escolher as unidades pode ser mais simples do que o método afirma.

### Relação com o AWARe (arXiv 2608.11758, 2026)

O AWARe ranqueia neurônios por saliência de **ativação** e congela a fração de
maior pontuação com máscara binária — essencialmente o braço `magnitude` desta
ablação. O resultado, então, lê-se como: **o critério do AWARe iguala o nosso**,
sob capacidade pareada e mesmas seeds.

Diferença de desenho que permanece: o orçamento do SlowHeat é **por camada**
(`_apply_capacity_budget` opera sobre o `importance_memory` de cada tracker),
enquanto o AWARe ranqueia globalmente. Isso não foi testado aqui.

## Consequência para o artigo

O manuscrito não pode reivindicar a importância funcional como vantagem de
critério. A reivindicação defensável que resta é sobre **proteção seletiva
versus não-seletiva** (+12,23 pp, 10/10), que é mais modesta e não é exclusiva
nossa.

Este é um resultado negativo pré-registrado e **será reportado**. Num artigo de
protocolo ele é ativo, não passivo: é evidência de que o instrumento rejeita
hipóteses, inclusive as nossas.

## Limites

- Um host (BERT), um benchmark (Split-CLINC150), duas tarefas por sequência.
- Trackers de FFN e atenção, **não** adaptadores LoRA. O critério é o mesmo
  código, mas a transferência para a geometria do LoRA não está estabelecida.
- O empate diz que `|z|` iguala `|z·dL/dz|` aqui; não diz que ambos são ótimos.

## Bugs corrigidos durante a execução

1. **`run_diagnostic` não propagava `importance_criterion`.** O `replace()` que
   monta a config por braço omitia o campo, então `slowheat_magnitude_hard`
   teria executado como funcional — empate garantido, sem nada no artefato
   denunciando. Detectado por teste antes da primeira seed.

2. **A telemetria media `task_ema`, que `consolidate()` zera.** A métrica
   reportava 0,0 para todos os critérios, exatamente quando o falsificador era
   necessário. Corrigida para ler `importance_memory`, o vetor que o top-k de
   fato ranqueia, com teste de não-regressão.
