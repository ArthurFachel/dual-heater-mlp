# Aplicação 1 — DualHeat-LoRA (legado)

**Fonte:** `src/dual_heater/lora.py` · **Testes:** `tests/test_lora.py` (6)
**Estado:** protótipo histórico. **Não** oferece proteção independente por
saída. Não deve ser usado como extensão validada do Functional SlowHeat.

## Claim

Nenhum. Este módulo é o ponto de partida que motivou as aplicações 2 e 3, e a
sua limitação está declarada na própria docstring desde a origem:

> o hook não garante proteção independente por saída e deve ser tratado como
> mecanismo experimental.

## Mecanismo

No DualHeat original a proteção lenta atua sobre o gradiente de `self.weight`,
via `register_hook` no parâmetro. Em LoRA a base é congelada e o único caminho
treinável é o delta de baixo posto, então não existe um `.weight` treinável
equivalente. O protótipo aplica o hook ao tensor `delta`, no espaço de saída,
antes de somá-lo à base:

```text
base  = F.linear(x, base_weight, base_bias)     # congelada (buffer)
delta = F.linear(F.linear(x, lora_A), lora_B) * scaling
delta.register_hook(escala_por_neuronio_de_saida)
z = base + delta
```

A inibição lateral rápida (fast heat) não precisou de adaptação: ela já opera
sobre a saída, então funciona igual vindo de um Linear cheio ou de base+LoRA.

## Por que a proteção não é exata

O hook reduz as contribuições de gradiente associadas às saídas protegidas e
propaga a escala para `A` e `B`. Mas `lora_A` é compartilhada entre todas as
saídas: updates induzidos por saídas **pouco** protegidas ainda alteram o delta
de saídas protegidas através de `lora_B`.

Esse vazamento é medido e assertado, não suposto. Em
`tests/test_lora_slowheat.py::test_leak_without_the_a_bound_lets_the_protected_output_drift`
a linha de delta de uma saída com `slow_heat = 1` sob máscara hard **muda** após
um passo do otimizador, quando só as linhas de `B` estão mascaradas.

## Diferenças em relação ao DualHeatLinear original

1. `base_weight`/`base_bias` são buffers, não `Parameter` — ficam fora do
   optimizer automaticamente, sem precisar filtrar por `requires_grad`.
2. O hook é registrado no tensor `delta` a cada forward, não uma vez no
   `__init__`, porque `delta` é um tensor novo a cada chamada.
3. `post_mag` reduz sobre todas as dimensões exceto a última, para funcionar
   tanto com entradas 2D `(batch, features)` quanto 3D `(batch, seq, hidden)`.
4. A escala é aplicada por broadcasting sobre a última dimensão do gradiente
   (`out_features`), em vez do `.view(-1, 1)` do original — lá o hook era sobre
   a matriz de pesos, aqui é sobre a ativação.

## Hiperparâmetros

| Símbolo | Nome | Default | Papel |
|---|---|---|---|
| `r` | rank | 8 | posto do adaptador |
| α | `lora_alpha` | 16.0 | `scaling = alpha / r` |
| α_f | `fast_decay` | 0.93 | EMA do fast heat |
| γ | `fast_strength` | 2.0 | inibição lateral divisiva |
| δ | `fast_decay_rate` | 0.04 | decay ativo do fast heat |
| β | `slow_strength` | 2.0 | `grad *= 1/(1 + β·slow_heat)` |
| — | `slow_window` | None | janela da média amostral |

## O que NÃO se pode afirmar

- Que proteger uma saída aqui impede que ela mude. Não impede.
- Que o mecanismo foi avaliado em continual learning. Não há run agregada.
- Que ele é comparável às aplicações 2 e 3 — ele não entrou no benchmark de
  Qwen, cujos braços usam `lora_slowheat.py`.

## Substituto recomendado

Para proteção exata por saída com `A` treinável, use o mecanismo `leak` da
[aplicação 3](lora_slowheat_mechanisms.md), que fecha exatamente este vazamento
limitando cada linha de `A` pela saída mais protegida que ela alcança.

## Referências

- [Índice das aplicações de LoRA](lora_applications.md)
- [Catálogo histórico, seção 3.6](project_methods_and_results.md)
