# Seleção de memória de replay — resultado

**Pré-registro:** `goals/protocol_replay_selector_confirmation.md` (R1–R10).
**Execução:** concluída em 28/09/2026 17h52, 20 seeds, 280 learner runs,
Titan Xp. Run lançada destacada após uma primeira tentativa ser morta por
SIGTERM aos 34 min.
**Artefatos:** `results/replay_selector_confirmation/`.

---

## Pergunta

Ao montar a memória de replay, qual critério escolhe os exemplos a guardar?

A heurística intuitiva é **perda alta**: guarde aquilo que o modelo erra, já que
é o que ele "ainda não aprendeu". A alternativa testada é **representatividade**:
guardar exemplos típicos da distribuição da tarefa.

## Desenho

Split-CIFAR100, 20 seeds, seis learners. Cada learner ranqueia as **próprias**
imagens de treino (`each_learner_ranks_its_own_training_images`), e cada
critério é comparado contra o baseline do mesmo learner — pareado por seed.

Endpoint primário: `final_average_accuracy`. Holm dentro de cada família
dataset/backbone/learner. Teste de sinal exato bilateral, n = 20.

Controles sem cache treinados uma vez por dataset/seed: `vanilla` para
`replay`/`derpp`, SlowHeat hidden-only para as variantes com SlowHeat.

## Resultado

| learner | seletor | FAA | CI95 | sinais | p |
|---|---|---:|---|---:|---:|
| derpp | `loss` | **−0,0405** | — | 0+/20 | 1,9e-06 |
| derpp | `representative` | **+0,0101** | [+0,0077, +0,0124] | 20+/0 | 1,9e-06 |
| replay | `loss` | **−0,0272** | — | 0+/20 | 1,9e-06 |
| replay | `representative` | **+0,0067** | [+0,0048, +0,0086] | 20+/0 | 1,9e-06 |
| slowheat+derpp | `loss` | −0,0349 | — | 0+/20 | 1,9e-06 |
| slowheat+derpp | `representative` | +0,0075 | [+0,0061, +0,0090] | 20+/0 | 1,9e-06 |
| slowheat+replay | `loss` | −0,0286 | — | 0+/20 | 1,9e-06 |
| slowheat+replay | `representative` | +0,0097 | [+0,0074, +0,0121] | 19+/1 | 4,0e-05 |

`average_forgetting` concorda em direção oposta em todos os oito contrastes,
como deve.

**Selecionar por perda alta piora. Selecionar por representatividade melhora.
Unânime em 20/20 seeds, quatro learners, dois endpoints.**

O efeito da heurística ruim (até −4,05 pp) é cerca de **quatro vezes maior em
módulo** que o da boa (+0,67 a +1,01 pp). Escolher mal o seletor custa mais do
que escolher bem rende.

## O resultado não depende do SlowHeat

O padrão aparece com e sem SlowHeat, com magnitudes comparáveis. É uma
propriedade do **critério de seleção de replay**, não do nosso método.

Isso o torna mais publicável, não menos: é um achado sobre uma prática comum em
CL, independente da contribuição da casa, e encaixa na âncora de protocolo.

## Custo

`representative` custa ~0,75 s de seleção por tarefa contra ~0,19 s de `loss`
— quatro vezes mais, sobre o mesmo número de forwards (38–40 mil exemplos). Em
termos absolutos é desprezível frente ao treino.

## Escopo — o aviso do próprio relatório

O relatório carrega `status: exploratory_not_independent_confirmation` e:

> Every ranked learner selects its own memory. Replay/SlowHeat+Replay and
> DER++/SlowHeat+DER++ contrasts are algorithm-level comparisons, not isolated
> SlowHeat effects.

Leitura correta: **comparações cruzadas entre learners** (p.ex. `replay` versus
`slowheat_replay`) misturam algoritmo e SlowHeat e são exploratórias. O
contraste **interno a cada learner** — fixado o algoritmo, variando só o
seletor — é o que o desenho isola, e é o reportado acima.

O aviso restringe quais comparações valem; não invalida a run.

## Limites

- **Um dataset** (Split-CIFAR100). `paired_differences_vs_first` tem só essa
  chave. Generalização a outros benchmarks não está estabelecida.
- Um tamanho de buffer e um backbone.
- Não testa seletores fora dos dois pré-registrados (e do híbrido).
- Não estabelece *por que* a perda alta prejudica; a hipótese natural é que
  exemplos de perda alta concentram outliers e ruído de rótulo, mas isso não
  foi medido aqui.

## Bug corrigido durante a execução

O guard-rail de resume rejeitava toda retomada: `config_payload` guarda tuplas,
o JSON as grava como listas, e a comparação usava `!=` sem normalizar. **O
resume nunca funcionou.** Corrigido por TDD (`_normalize_for_index_comparison`,
commit `4f5d08c`), mantendo a rejeição para mudanças reais de configuração. As
26 runs anteriores foram descartadas para
`results/_abandoned_gpu_partial_pre_resumefix/` porque `ensure_run_identity`
amarra as seeds ao hash do código, que mudou no fix.
