# Protocolo congelado — ablação do critério de importância

> Pré-registro. Escrito **antes** de qualquer seed desta ablação. Não é editado
> depois de ver acurácia; alterações exigem commit anterior à run e linha nova
> na tabela K.

**Estado: CONGELADO em 28/09/2026**, antes da primeira seed.
**Host:** BERT + Split-CLINC150 (2 tarefas), o mesmo do diagnóstico mecanístico.

---

## A. O que motiva

A proteção seletiva do projeto rankeia unidades por **importância funcional de
primeira ordem**, `|z · dL/dz|` (`src/dual_heater/slow_heat.py:192`). Nunca foi
testado se a parte do gradiente é necessária.

O BERT já mostrou que *ranking importa*: o braço `slowheat_hard` (ranking
aprendido) supera `slowheat_random_hard` (máscara aleatória com a mesma
capacidade) em +3,65 p.p. de FAA e +9,37 p.p. de retenção, com retenção positiva
em 10/10 seeds.

Isso deixa uma lacuna: **entre "aleatório" e "funcional" existe um ponto
intermediário nunca medido** — rankear por magnitude de ativação `|z|`, sem
gradiente. A hipótese ordenada é:

```text
aleatorio  <  magnitude |z|  <  funcional |z * dL/dz|
```

## B. Por que esta ablação não é "mais um mecanismo"

A §3 de [plano_metodo_lora.md](plano_metodo_lora.md) desaconselha propor novos
mecanismos: quatro já falharam e o espaço está saturado. **Esta ablação não
propõe mecanismo.** Ela troca **uma linha** do critério de importância e mantém
todo o resto idêntico — normalização, EMA, orçamento de capacidade, máscara,
consolidação. É uma pergunta sobre *o que precisa ser medido*, não sobre uma
arquitetura nova.

Viabilidade verificada por protótipo descartável em 28/09 (monkeypatch, nada em
produção): o pipeline aceita a troca sem refatoração. Numa perda construída com
gradientes 1000× diferentes entre unidades, a variância do ranking normalizado
foi **1,159** (funcional) contra **0,028** (magnitude) — o funcional separa, a
magnitude quase não.

## C. Pergunta confirmatória

Em BERT/Split-CLINC150 com máscara hard e capacidade pareada, o critério
`magnitude` produz retenção da primeira tarefa **diferente** de `aleatório` e de
`funcional`?

## D. Os desfechos, declarados antes

| desfecho | leitura | consequência |
|---|---|---|
| `funcional` > `magnitude` > `aleatório` | hipótese ordenada confirmada; o gradiente agrega informação | quantifica o valor do gradiente; mecanismo segue para fase 2 |
| `funcional` ≈ `magnitude` > `aleatório` | o gradiente **não** agrega; magnitude basta | **simplifica o método** — achado prático, e mais barato |
| `funcional` ≈ `magnitude` ≈ `aleatório` | o ranking não é o gargalo | **fecha a linha de mecanismo**; o problema é a geometria da `A` compartilhada |
| `magnitude` < `aleatório` | magnitude é pior que acaso | achado negativo informativo sobre saturação |

Os quatro são publicáveis. **Nenhum autoriza iniciar um quinto mecanismo** sem
a hipótese exigida na §5 do plano.

## E. Decisões congeladas

| # | Decisão | Valor congelado | Justificativa |
|---|---|---|---|
| I1 | Host | BERT + Split-CLINC150, 2 tarefas | é onde o contraste ranking-vs-aleatório já foi medido, com efeito real |
| I2 | Braços | `slowheat_random_hard`, **`slowheat_magnitude_hard`** (novo), `slowheat_hard` | os três pontos da hipótese ordenada. Todos com `mask_mode` hard e `slow_strength = 3.0`, idênticos ao diagnóstico existente |
| I3 | Braço de referência | `vanilla` | reportado sempre, fora da família confirmatória |
| I4 | Seeds | 10, banda `4_000_003, 4_025_011, 4_050_017, 4_075_037, 4_100_043, 4_125_059, 4_150_061, 4_175_087, 4_200_089, 4_225_097` | banda própria. As seeds do diagnóstico BERT existente (`[2, 9, 10, 28, 30, 32, 57, 67, 2005, 2012]` em `bert_slowheat_review`, e `[11, 22, 33]` em `bert_slowheat_diagnostic`) **não podem ser reusadas** |
| I5 | Endpoint primário | retenção da primeira tarefa (`first_task_retention`) | é o endpoint onde o efeito ranking-vs-aleatório foi unânime (10/10) |
| I6 | Endpoints secundários | FAA, forgetting, acurácia da última tarefa | reportados sempre, nunca promovidos |
| I7 | Família confirmatória | **2 contrastes**: `magnitude − aleatório` e `funcional − magnitude` | são as duas arestas da hipótese ordenada. `funcional − aleatório` já é conhecido e fica fora |
| I8 | Teste | sinal exato bicaudal sobre as 10 diferenças pareadas por seed | mesmo teste do resto do projeto |
| I9 | Correção múltipla | Holm sobre a família de 2, α = 0,05 | declarada antes da run |
| I10 | Capacidade | `plasticity_budget` idêntico entre os três braços | sem isso o contraste mistura critério com quantidade de proteção — o erro que o `lr_control` existe para evitar |
| I11 | Hiperparâmetros | idênticos ao `diagnostic_conditions()` existente: `slow_strength = 3.0`, `epochs_per_task = 4`, `max_length = 128`, `batch_size = 2` | re-tunar por braço destruiria a comparação |

## F. Métrica diagnóstica obrigatória — o falsificador

O protótipo expôs um risco que **precisa ser declarado antes**: a proteção usa
`_apply_capacity_budget`, que protege as top-k unidades por importância. Se o
ranking de magnitude for quase plano, a seleção do top-k passa a ser decidida
por ruído, e **o braço `magnitude` degenera em máscara quase-aleatória**.

Se isso acontecer, um empate `magnitude ≈ aleatório` **não** significa "a medida
não importa" — significa que a medida não foi efetivamente testada. Sem declarar
isso agora, depois fica impossível distinguir achado de artefato.

Duas métricas obrigatórias, registradas por braço e por estágio:

1. **variância do ranking normalizado** (`task_ema.var()`);
2. **sobreposição do conjunto top-k protegido** entre `magnitude` e
   `aleatório`, e entre `magnitude` e `funcional` (índice de Jaccard).

**Regra de interpretação, congelada:** se a sobreposição top-k entre `magnitude`
e `aleatório` for **> 0,8**, o braço `magnitude` é declarado *degenerado* e
qualquer empate entre os dois é reportado como **inconclusivo**, não como
evidência de que o critério é irrelevante.

## G. Análise declarada

```text
para cada braco b em {aleatorio, magnitude, funcional}:
    para cada seed k: R_b,k = retencao da primeira tarefa

d1_k = R_magnitude,k - R_aleatorio,k
d2_k = R_funcional,k - R_magnitude,k

p1 = sinal exato bicaudal sobre {d1_k}
p2 = sinal exato bicaudal sobre {d2_k}
p1, p2 := Holm(p1, p2)

reportar sempre, junto: variancia do ranking e sobreposicao top-k
```

Sem análise intermediária, sem parada antecipada, sem inspeção seed a seed antes
das 10 terminarem.

## H. Implementação exigida antes da run

1. Parâmetro `importance_criterion` em `_initialize_importance_state`
   (`slow_heat.py`), com default `"functional"`. **Cinco classes herdam o mixin**
   (`SlowHeatLinear`, `SlowHeatConv2d`, `SlowHeatChannelTracker`,
   `SlowHeatFFNTracker`, `SlowHeatAttentionTracker`) e há **cinco pontos de
   chamada** — com default, nenhum muda de comportamento.
2. **Teste TDD de não-regressão:** com `importance_criterion="functional"`, o
   `task_ema` resultante deve ser bit-idêntico ao atual. Isto é obrigatório —
   sem ele, a ablação pode medir uma refatoração acidental em vez do critério.
3. Teste de que `"magnitude"` ignora o gradiente: dois gradientes diferentes com
   a mesma ativação devem produzir o mesmo `task_ema`.
4. Braço `slowheat_magnitude_hard` adicionado a `diagnostic_conditions()`.
5. Emissão das duas métricas da seção F na telemetria de consolidação.
6. Seeds de I4 registradas em código, como as demais bandas.
7. Suíte verde e este arquivo commitado **antes** da primeira seed.

## I. Custo

O diagnóstico BERT existente roda com `--device cpu` por padrão, mas o host tem
3 Pascal livres. **O custo real deve ser medido com 1 seed antes de lançar as
10** — a estimativa por analogia já falhou uma vez neste projeto (protocolo do
seletor, seção F), por comparar tempos de CUDA com execução em CPU.

Regra: rodar 1 seed, medir, e só então projetar. Se a projeção passar de 4 h,
reavaliar o escopo antes de continuar.

## J. O que esta ablação não resolve

- **Não é contribuição de âncora.** É diagnóstico que fortalece a seção de
  mecanismo; não muda o que o campo faz e não substitui a re-avaliação sob
  plasticidade pareada.
- **Um host, duas tarefas.** O resultado não se generaliza automaticamente para
  Qwen/LoRA nem para sequências longas.
- **Não testa a `A` compartilhada.** O BERT usa `SlowHeatFFNTracker` em FFN e
  atenção, não adaptadores LoRA. Um empate aqui é evidência sobre o critério,
  não sobre a geometria do LoRA.
- **`magnitude` depende da escala das ativações.** Normalização de camada e
  escala do backbone influenciam o ranking muito mais do que no caso funcional.

## K. Registro de alterações

| Data | Alteração | Antes da run? |
|---|---|---|
| 28/09/2026 | criação e congelamento. I1 a I11 fechados, falsificador da seção F declarado. | sim — nenhuma seed executada |
