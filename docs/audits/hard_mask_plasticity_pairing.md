# Pareamento de plasticidade sob máscara binária (`--hard`)

**Data:** 08/10/2026. **Natureza:** correção de bug de harness, encontrada em
pré-check de CPU antes de um run autorizado. Nenhum resultado publicado é
afetado — ver §"Alcance retroativo".

---

## O defeito

`--hard` e `--target-plasticity` eram aceitos juntos pela CLI de
`experiments/qwen_lora_slowheat.py` e **o pareamento não acontecia**.

`_factor` (`src/dual_heater/lora_slowheat.py`) tem dois caminhos:

```python
def _factor(tracker, hard):
    if hard:
        return (tracker.slow_heat <= 0.0).to(...)   # binário; ignora slow_strength
    return tracker.get_lr_scales()                   # 1/(1 + beta * slow_heat)
```

`calibrate_to_target_plasticity` resolvia `slow_strength` por bisseção nos dois
casos. No caminho `hard`, `slow_strength` não entra no cálculo da máscara: o
solver rodava as 60 iterações, devolvia um knob, e a plasticidade ficava onde o
orçamento padrão a tinha deixado.

Medido antes da correção, num MLP sintético (`exact`, rank 8):

| `hard` | plasticidade antes | knob resolvido | plasticidade depois | erro vs alvo 0,85 |
|---|---:|---:|---:|---:|
| `False` | 0,6950 | 0,803735 | 0,8500 | 0,0000 |
| `True` | 0,5000 | 3 (inalterado) | **0,5000** | **0,3500** |

Varredura de alvos no braço `hard`: 0,60 / 0,70 / 0,85 / 0,95 **todos**
alcançavam 0,5000. O solver não tinha alça sobre a máscara.

Consequência se tivesse rodado: o braço de tratamento treinaria a plasticidade
0,50 contra um controle a 0,85, e qualquer diferença viria confundida com
35 pontos de dose — exatamente o que `target_plasticity` existe para impedir.

O número 0,5000 é do MLP de brinquedo; em Qwen seria outro. O que não depende do
brinquedo é a estrutura: nenhum alvo era alcançável.

## A correção

O knob do braço `hard` passa a ser **`plasticity_budget`**, a fração de unidades
protegidas, que é o que `_apply_capacity_budget` consome para decidir
`max_protected`. Pareamento por **fração protegida**, não por força de máscara.

Três razões para esta escolha e não por um piso na máscara:

1. preserva a propriedade que o `hard` entrega — drift exatamente zero nas
   unidades protegidas. Um piso ε > 0 a destruiria, e o próprio docstring do
   `slice` registra que relaxar o piso "TRADES AWAY the mechanism's exactness
   claim";
2. é o análogo direto do host BERT, cujos braços foram pareados em **entradas
   protegidas** (2.364.672 em cada braço de R-A);
3. mantém `slow_strength` intocado no braço `hard`, onde ele é inerte — mexer
   nele seria um no-op que ainda por cima corromperia os hiperparâmetros
   registrados no manifesto.

Depois da correção, mesmo probe:

| `hard` | alvo | alcançado | erro |
|---|---:|---:|---:|
| `False` | 0,60 / 0,70 / 0,85 / 0,95 | 0,6000 / 0,7000 / 0,8500 / 0,9500 | 0,0000 |
| `True` | 0,60 / 0,70 / 0,85 / 0,95 | 0,5938 / 0,6875 / 0,8438 / 0,9375 | ≤ 0,0125 |

### O erro residual do braço `hard` não é bug

Uma máscara binária é **função em degraus** do seu knob: uma unidade entra ou
sai do conjunto protegido por vez. Com rank 16, um degrau vale 1/16 = 0,0625, e
o erro observado (≤ 0,0125) está dentro de um degrau.

Isso é uma **limitação declarada**, não uma imprecisão a esconder: o protocolo
que usar `hard` precisa ler `achieved` do manifesto e reportá-lo, nunca assumir
que `target` foi atingido. O docstring do controlador diz isso explicitamente.

## Correção secundária: bracket unilateral

O caminho `slice` checava só uma ponta do intervalo, e o caminho soft checava as
duas em ramos separados. Um bracket unilateral pode devolver `solved = 1.0` com
`achieved` longe do alvo quando o knob não alcança — a mesma classe de falha
silenciosa. Agora as duas pontas são medidas e comparadas contra o alvo num
único caminho, com a direção de monotonicidade declarada por ramo
(`increasing`).

## Verificação

Cinco testes novos em `tests/test_lora_slowheat.py`:

| teste | o que fixa |
|---|---|
| `test_hard_controller_solves_the_budget_not_the_strength` | knob certo, `slow_strength` intocado (3 métodos) |
| `test_hard_controller_moves_the_protected_fraction` | mais plasticidade ⇒ menos unidades protegidas |
| `test_hard_mask_stays_binary_after_calibration` | a calibração não compra plasticidade afrouxando a máscara |
| `test_soft_controller_still_solves_the_strength` | o ramo `hard` não captura o caminho soft |
| `test_controller_fails_loudly_on_an_unreachable_hard_target` | alvo inalcançável reporta `solved = 0.0` |

**Quatro mutações, quatro mortas:**

| mutação | mortos |
|---|---|
| restaurar o bug (`elif self.config.hard` → `elif False`) | 4 testes |
| solver não reaplica o orçamento | 4 testes |
| ramo `hard` afrouxa a máscara para 0,1 (vira opção "piso") | 4 testes |
| bracket volta a ser unilateral | 3 testes |

Suíte completa: **1185 passed, 2 skipped** (os 2 exigem CUDA). Antes da mudança
eram 1178.

`ruff check` nos arquivos tocados: limpo. O RUF046 em `slow_heat.py:139` é
pré-existente, verificado com `git stash`.

## Alcance retroativo

**Nenhum resultado publicado é afetado.** Verificado nos manifestos:

- R-C (`results/exact_decomposition/seed_5*`, 10 seeds): `hard: False`;
- piloto do TRACE (`results/trace_full/`): `hard: False`.

O defeito só dispara para quem passar `--hard`, e nenhum run o fez. A flag
existia na CLI, era aceita sem aviso, e nenhum teste cobria a combinação — por
isso os 1178 passavam.

## Por que isto aparece aqui

Terceira ocorrência do mesmo padrão no projeto, e vale registrar a série:

| caso | defeito | como foi pego |
|---|---|---|
| R-A (28/09) | `run_diagnostic` não propagava `importance_criterion` | teste antes da primeira seed |
| R-E (28/09) | escopo do denominador da métrica | revisão de código |
| aqui (08/10) | knob inerte sob máscara binária | pré-check de CPU antes do disparo |

Os três teriam produzido artefatos plausíveis e um resultado errado. Nenhum
seria detectável lendo o manifesto depois: o campo `target_plasticity` registra
o alvo pedido, não o alcançado.

Daí a regra que o controlador agora documenta: **ler `achieved`, nunca assumir
`target`.**

## Arquivos

- `src/dual_heater/lora_slowheat.py` — ramo `hard` no controlador, bracket
  bilateral unificado, docstring do contrato.
- `src/dual_heater/slow_heat.py` — `reapply_capacity_budget()`, que re-deriva
  `slow_heat` do orçamento corrente sem consolidar de novo (uma segunda
  consolidação dobraria `task_ema` dentro de `importance_memory` e mudaria o
  próprio ranking sob calibração).
- `tests/test_lora_slowheat.py` — os cinco testes.
