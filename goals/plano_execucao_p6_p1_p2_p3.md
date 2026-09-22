# Plano de execução: P6, congelamento do protocolo, P2, P3, P4

> **NÃO APLICADO.** Nada deste documento foi implementado, executado ou
> commitado. É o detalhamento dos próximos passos para revisão antes de
> execução. Nenhuma linha de código foi escrita, nenhuma run foi disparada.

Última atualização: 22 de setembro de 2026.
Base: `goals/qwen_heat_roadmap.md` (Gate 0 fechado), `goals/protocol_iso_plasticity.md`
(rascunho), `goals/opcoes_novidade_e_proximos_passos.md`.
HEAD na hora de escrever: `d4eacf5`.

---

## 0. Ordem proposta, e por quê

```text
P6  Meta 2 reduzida (medição pura, sem acurácia)   ~30 min CPU
 |
 v
P1  Congelar o protocolo (texto)                   dá para fechar A4 já
 |
 v
P2  Runner experiments/qwen_iso_plasticity.py      código
 |
 v
P3  Testes + mutações                              código
 |
 v
P4  Run de calibração em GPU                       exige autorização
```

P6 vem antes do congelamento porque decide se as camadas 19 a 23 e 3/21 precisam
de tratamento separado nos braços. É medição sem acurácia, então a regra "nada
com acurácia antes do protocolo congelado" não é violada. Se P6 mostrar que a
anomalia é artefato de seed ou de ordem, o protocolo congela mais simples.

A ordem é reversível: se preferir o resultado principal antes, P1 -> P2 -> P3 ->
P4 e P6 depois. O que não pode é rodar P4 sem P1 congelado.

---

## 1. P6 — Meta 2 reduzida

**Pergunta:** por que L3 e L21 concentram importância em 1 a 2% das unidades
(`ratio` 0,018 e 0,009) enquanto as outras 22 usam 63 a 84%? E por que PR cai
nas últimas camadas (L19 2493, L20 2269, L22 1212, L23 628)?

### 1.1 Mudança necessária antes de medir

`experiments/qwen_capacity_diagnostic.py` não tem como inverter a ordem das
tarefas: `--tasks N` sempre pega `CLINC150_DOMAINS` na ordem do dicionário, e a
lista é fixada em `main()` por `[: args.tasks]`.

Opções:

| Opção | Mudança | Veredito |
|---|---|---|
| `--domains banking credit_cards` | flag nova, `nargs="+"`, valida contra `CLINC150_DOMAINS`, substitui o `[: args.tasks]` | **preferida** — explícita, sem inferir intenção |
| `--task-order reverse` | inverte a lista construída | funciona, mas o nome promete mais do que faz |

Custo: ~5 linhas. Ordem invertida é `credit_cards -> banking`.

### 1.2 Runs

```sh
cd /mnt/B-SSD/fachel/dual-heater-mlp
for SEED in 0 1 2; do
  for ORDER in "banking credit_cards" "credit_cards banking"; do
    CUDA_VISIBLE_DEVICES= HF_HOME=.hf-cache PYTHONPATH=. \
      .venv/bin/python experiments/qwen_capacity_diagnostic.py \
        --domains $ORDER --seed $SEED --capacity-scope local \
        --output results/qwen_layer_anomaly/seed${SEED}_$(echo $ORDER | tr ' ' '_')/manifest.json
  done
done
```

6 runs, CPU, ~5 min cada pela medição de P0 (~30 min no total). Sem download, o
cache já tem modelo e dataset. `--iso-plasticity` não é necessário aqui: a
pergunta é sobre importância crua, não sobre plasticidade pareada. Manter fora
reduz o custo pela metade.

### 1.3 O que ler em cada saída

| Quantidade | Onde | O que decide |
|---|---|---|
| `participation_ratio` por camada | `stages[*].layers[*]` | a anomalia sobrevive a seed e a ordem? |
| `concentration_ratio` por camada | idem (a chave nova) | L3/L21 continuam nominais? |
| `mean_heat` por camada | idem | o gradiente de magnitude entre camadas é estável? |
| histograma e quantis da importância crua de L3/L21 | a escrever | a concentração é de 1 unidade ou de um grupo? |
| interseção dos top-100 por camada entre tarefas | a escrever | as unidades dominantes são as mesmas entre tarefas? |
| `concentration_ratio` por camada sob escopo `hierarchical` | rodar 1 seed com `--capacity-scope hierarchical` | a redistribuição de cota corrige a subproteção? |

A interseção entre tarefas e o histograma cru não existem no runner hoje:
`heat_concentration` devolve contagem, quantis e razão de participação, não os
índices. Se forem necessários, é um helper novo em `capacity_calibration.py`
(por exemplo `dominant_unit_overlap(heat_a, heat_b, k)`), com teste e mutação
próprios. Registrar antes de escrever que a leitura depende disso.

### 1.4 Entregável e gate

`docs/qwen_layer_anomaly.md`, no formato de
`docs/bert_slowheat_diagnostic_results.md`: pergunta, protocolo, tabela por
seed e por ordem, leitura, e **o que continua sem explicação**.

Gate 2 (do roadmap): explicação sustentada por medição, ou registro explícito de
que segue sem explicação. Não inventar mecanismo para fechar a lacuna.

Hipótese que eu levaria para a run, a ser confirmada ou descartada: a queda de PR
no fim da rede é efeito de profundidade (magnitude de `|z dL/dz|` cai mais perto
da cabeça porque a loss é dominada pelos logits), e L3/L21 são outra coisa, sem
relação com profundidade. Se for isso, o tratamento separado vale só para L3/L21.

---

## 2. P1 — Congelar o protocolo

Não exige código nem run. Exige quatro decisões.

### 2.1 A4 pode fechar agora, com dado

O critério declarado está em regime informativo: `heavy_tailed_warning=False`,
`PR/N` agregado 11,5% (o guard-rail é 5%), `PR` por camada 63 a 84%, exceto as
seis camadas em zona ruim. Piso de 0,60 avaliado em `b=0,25` seleciona
`beta=10` (`E=0,760`) na grade, ou o maior beta entre 10 e 30 se o solver for
contínuo. `selection` deixa de ser `None`. **Recomendação: declarar 0,60 em
`b=0,25`.**

### 2.2 A1 a A3

| Item | Recomendação | Custo de errar |
|---|---|---|
| A1 `E*` | 0,75 primário, 0,50 secundário | alto: `E*=0,50` só tem 3 budgets alcançáveis e contraste de 1,27x |
| A2 política de `beta` | re-resolvido por fronteira | médio: fixo é auditável, re-resolvido habilita M3 |
| A3 seeds | calibração {0,1,2}, confirmação 10 declaradas | alto se as de confirmação forem escolhidas depois |

### 2.3 Como congelar

1. Preencher o bloco A e trocar o status no topo de `RASCUNHO` para `CONGELADO`.
2. Registrar na tabela K a data e o commit do congelamento.
3. Commit isolado, mensagem `docs: freeze the iso-plasticity ablation protocol`.
4. A partir daí o arquivo não é editado. Mudança só por commit anterior à run
   correspondente, registrada na tabela K.

---

## 3. P2 — Runner `experiments/qwen_iso_plasticity.py`

### 3.1 Reaproveitar de `experiments/split_clinc150.py`

| Símbolo | Linha | Uso |
|---|---|---|
| `build_clinc150_tasks` | 477 | construir as tarefas |
| `load_clinc150_tasks` | 532 | carregar do cache |
| `text_task_fingerprint` | 587 | identidade do protocolo no manifesto |
| `_evaluate` | 659 | matriz de acurácia; retorna `(class_acc, task_acc, macro_f1)` |
| `_trimmed_batch` | 620 | batch de avaliação |
| `_mask_unseen_logits` | 641 | Class-IL: mascara classes não vistas |
| `capture_parameter_drift_reference` | 848 | referência de drift antes de cada tarefa |
| `summarize_parameter_drift` | 869 | drift protegido e plástico |
| `select_replay_examples` | 604 | só quando replay entrar, na Meta 3 |
| `CLINC150Task`, `TokenizedTextSplit`, `CLINC150_DOMAINS` | 369, 329, 60 | tipos e ordem |

`_evaluate` é privada e o projeto já a usa entre módulos do mesmo pacote. Antes
de escrever o runner: confirmar assinatura por keyword (`task_classes`,
`seen_classes`, `batch_size`, `device`) e que ela restaura `model.training`.

**Não** reaproveitar `_find_slowheat_model` (linha 808): é tipada para
`SlowHeatBertForSequenceClassification`. O Qwen precisa do seu próprio finder,
como o de `experiments/qwen_slowheat_smoke.py`.

### 3.2 Escrever novo

| Função | Contrato |
|---|---|
| `build_arms(layers, target_plasticity, budgets, scope)` | retorna `IsoPlasticityPoint` por budget alcançável via `iso_plasticity_family_scoped`; descarta inalcançável, nunca satura |
| `permute_slowheat_heat(model, *, seed)` | controle iso-E correto |
| `register_reduced_lr_arm(model, optimizer)` | confirma que nenhuma máscara foi registrada |
| `resolve_beta(model, arm, target, policy)` | política `first` ou `per_boundary` |
| `run_arm(...)` | treino por tarefa, `consolidate()` na fronteira, `_evaluate` nos pontos declarados |
| `write_manifest(...)` | protocolo, hash do protocolo, escopo, braços, `beta` e `E` atingido por fronteira |

O controle correto, contra o que existe hoje:

```python
# ERRADO para iso-E: hard em identidades sorteadas (split_clinc150.py:820)
#   state.slow_heat.zero_(); state.slow_heat[indices] = 1.0

# CORRETO: permutar o vetor inteiro preserva a distribuição e E
def permute_slowheat_heat(model, *, seed: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for tracker in model.get_slow_states():
            heat = tracker.slow_heat
            if int(torch.count_nonzero(heat).item()) == 0:
                continue
            permutation = torch.randperm(heat.numel(), generator=generator)
            heat.copy_(heat[permutation].to(heat.device))
```

Permutar o vetor completo, zeros incluídos, preserva a contagem de protegidas e o
multiconjunto de valores, então `E` fica idêntico por construção. `slow_heat` vem
de `_SlowHeatImportanceMixin` e existe tanto em `SlowHeatFFNTracker` quanto no
estado do BERT.

### 3.3 Braços por `E*`

| Família | budgets | alcançáveis | observação |
|---|---|---|---|
| `E*=0,75` | 0,05 / 0,10 / 0,25 / 0,50 | 4 | carrega o contraste (1,90x contagem, 2,58x `beta`) |
| `E*=0,50` | 0,05 / 0,10 / 0,25 / 0,50 | 3 | `b=0,50` cai no piso, `solve_*` retorna `None` |

Mais os três controles por família: permutado, LR reduzido (`lr * E*`, sem
máscara), vanilla (`E=1`). Total 7 braços em `E*=0,75` e 6 em `E*=0,50`.

### 3.4 Pitfalls que o runner tem de respeitar

- Carregar em fp32 e passar `fp16=True`, **ou** carregar fp16 com `fp16=False`.
  Nunca os dois: o scaler espera pesos mestres fp32.
- `gradient_checkpointing` conflita com forward hooks. Desligado.
- Nenhum `device_map="auto"`: uma GPU só, DDP desabilitado (GPUs heterogêneas).
- `freeze_unbound_parameters=True` congela o envelope sem binding. A cabeça
  `score` é aleatória e **precisa** ficar plástica; ela é exposta por
  `exempt_parameter_names()` e contada separada de mascarados.
- Um binding por parâmetro. Quando produtor e consumidor protegem o mesmo peso,
  combinar por mínimo, não registrar duas vezes.
- Alocação de estado ativo: só para máscaras habilitadas.
- `evaluate_all`/`_evaluate`: `seen_tasks` é o terceiro posicional, `output_dir`
  o quarto. Não usar keyword `tasks=`.
- `softmax` de Class-IL: `_mask_unseen_logits` com `seen_classes` por tarefa.

### 3.5 Ordem de treino por braço

```text
para cada tarefa t:
  model.train()
  para cada step: forward, backward, optimizer.step()
  model.eval(); _evaluate em todas as tarefas vistas
  se t < T-1:
     model.consolidate(strategy="max")
     se policy == "per_boundary": re-resolver beta do braço em E(beta) = E*
     model.register_plasticity_masks(optimizer, hard=False)
     capturar drift de referência para a próxima tarefa
```

Detalhe que decide a leitura: a resolução de `beta` depende da importância da
primeira tarefa, então `beta` só existe a partir da fronteira 1->2. Com
`policy="first"` resolve uma vez e congela. Reportar os dois ou declarar um; não
misturar.

---

## 4. P3 — Verificação em CPU

Modelo Qwen2 minúsculo, sem download, no formato de `tests/test_slow_heat_qwen.py`.
Arquivo novo: `tests/test_qwen_iso_plasticity.py`.

### 4.1 Asserções

1. todo braço atinge `E*` dentro de 1e-6 (`effective_plasticity` contra o alvo);
2. braço permutado tem distribuição de heat idêntica (`torch.sort` dos valores
   bate) e `E` idêntico, com `permutation != identity` verificado;
3. braço de LR reduzido não registra máscara nenhuma (`mask_bindings()` vazio);
4. drift protegido exatamente zero sob máscara hard;
5. manifesto contém todos os braços, `beta` por fronteira, `E` atingido, escopo,
   hash do protocolo e a lista de budgets descartados por inalcançáveis.

### 4.2 Mutações

Injetar no código de produção e confirmar que cada uma quebra algum teste:

| Mutação | Teste que deve quebrar |
|---|---|
| inverter o eixo da máscara | 4 |
| trocar permutação por proteção hard | 2 |
| ignorar `capacity_scope` | 1 |
| aplicar `E*` errado num braço | 1 |
| saturar `beta` em vez de descartar budget inalcançável | 5 |
| esquecer `seen_classes` na avaliação | avaliação |
| registrar máscara no braço de LR reduzido | 3 |

Mutação sobrevivente é lacuna de teste ou equivalência provada. Investigar e
documentar qual, como já foi feito para as duas equivalências de
`test_capacity_calibration.py`.

Cuidado conhecido: teste que passa de primeira não é evidência. E atenção
redundante de garantia dupla: attention causal mais pooling do último token não-pad
já zeram gradiente de padding, então um teste de invariância a padding passa mesmo
com a máscara de validade removida. Testar a máscara no tracker diretamente e
documentar o teste como guarda de regressão.

### 4.3 Critério de saída

Suíte completa verde (498 + os novos), com escopo e mutação testadas, não só "os
testes passaram".

---

## 5. P4 — Run de calibração

**Exige autorização do Fachel.** Não é CPU.

### 5.1 Pré-requisitos

- protocolo congelado e commitado (P1);
- runner e testes verdes (P2, P3);
- smoke de memória em GPU: a estimativa de ~8,3 GB nunca foi medida. Rodar
  `experiments/qwen_slowheat_smoke.py` em GPU antes, com `--budget 0.50` (o braço
  mais protegido) e medir pico alocado e reservado.

### 5.2 Configuração

| Item | Valor |
|---|---|
| GPU | **uma** placa, `CUDA_VISIBLE_DEVICES` fixado |
| Precisão | fp16 (Pascal, CC 6.x, sem bf16) |
| `batch_size` | 2 |
| Tarefas | 2 (`banking -> credit_cards`) |
| Passos por tarefa | 30 |
| Seeds | {0, 1, 2} (calibração) |
| Braços | 7 em `E*=0,75`, 6 em `E*=0,50` |
| Runs totais | 3 seeds x 13 braços = 39 |

Custo estimado: cada run é 2 tarefas x 30 passos com batch 2, mais avaliação em
2 pontos. Ordem de minutos por run, então poucas horas no total. Medir e reportar
o tempo real da primeira run antes de multiplicar.

### 5.3 Saída

`results/qwen_iso_plasticity/seed{SEED}/manifest.json` por run, mais um agregado
com diferenças pareadas por seed e contagem de sinais. Copiar para `artifacts/`
para versionar, seguindo a convenção do repo (`results/` é ignorado).

### 5.4 Gate 1

Conforme seção I do protocolo. Reprovado é resultado registrado, não obstáculo.

---

## 6. Riscos de execução

| Risco | Onde aparece | Mitigação |
|---|---|---|
| `_evaluate` mudar de assinatura e quebrar o runner | P2 | confirmar antes; o teste de P3 cobre |
| Anomalia de L3/L21 ser artefato de seed | P6 | é exatamente o que P6 mede; se for, o protocolo simplifica |
| Ordem das tarefas mudar a leitura da Meta 1 | P6, P4 | declarar a ordem no protocolo; P6 mede o efeito |
| Estimativa de memória otimista | P4 | smoke em GPU com o braço mais protegido |
| Braços indistinguíveis do LR reduzido | P4, Gate 1 | resultado negativo publicável; o controle é obrigatório |
| Nome `SlowHeat` herdar o endpoint negativo do BERT | P9 | decidir o nome antes de escrever a Meta 3 |
| Manipulação de `slow_heat` pós-consolidação divergir do caminho de produção | P2, P3 | comparar `mask_bindings()` com e sem o hook de produção |

---

## 7. Estado: o que este documento não faz

- não escreve código (P2, P3 e a flag `--domains` de P6.1 não existem);
- não roda nada (nem as 6 medições de P6, que são CPU e baratas);
- não altera `goals/protocol_iso_plasticity.md`, que segue em rascunho;
- não foi commitado.

Aprovação necessária em dois pontos: quais decisões de A1 a A4 você quer, e se a
ordem proposta (P6 antes de P4) fica.
