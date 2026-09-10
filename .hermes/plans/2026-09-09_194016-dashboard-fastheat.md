# Dashboard BERT: visualizar FastHeat além do SlowHeat

## Goal
Adicionar o FastHeat como mais um sinal visualizável no dashboard BERT ao vivo,
além do SlowHeat já exibido.

## Current context / assumptions
- Repositório: `/home/fachel/dual-heater-mlp-research` (branch de trabalho atual).
- O dashboard é servido por `experiments/live_dashboard.py` (HTTP read-only) e
  renderizado por `experiments/live_dashboard.html`.
- A fonte dos dados é `experiments/live_telemetry.py`:
  - `build_heat_snapshot(...)` monta `{"schema_version":1, ..., "attention":[...], "ffn":[...]}`.
  - Cada entrada `ffn[index]` hoje tem: `layer`, `units`, `task_importance`,
    `consolidated_importance`, `heat`, `plasticity`, frações de proteção.
  - Cada entrada `attention[index]` tem a mesma estrutura por cabeça.
- O FastHeat está em `dual_heater/fast_heat.py` → `FastHeatGate.fast_heat`
  (um buffer por gate). No BERT há um gate por camada FFN, obtido por
  `model.get_fast_states()` (retorna lista de `FastHeatGate`, uma por camada,
  na ordem das camadas). `SlowHeatBertForSequenceClassification.get_fast_states()`
  já existe.
- O dashboard HTML já tem um seletor `#heatMetric` (Importância atual /
  consolidada / Heat aplicado / Plasticidade) e `drawFFN(layer, metric)` lê
  `layer[metric] || []` por neurônio. Atenção não possui FastHeat (o gate é
  somente pós-GELU FFN), então FastHeat só faz sentido no painel FFN.
- Hipótese de design: enriquecer cada entrada `ffn[index]` com `fast_heat`
  (lista) e adicionar a opção `fast_heat` no seletor. O `drawFFN` existente
  renderiza a nova opção sem mudanças adicionais. Nome `fast_heat` é
  consistente com o buffer real `FastHeatGate.fast_heat`.
- Não vou subir o `TELEMETRY_SCHEMA_VERSION`: adicionar campo opcional é
  retrocompatível; snapshots antigos simplesmente não terão `fast_heat`.
- Nenhuma mudança no runner (`split_clinc150.py`) é necessária: ele já chama
  `publish_heat` com o modelo completo, então o FastHeat chega ao snapshot
  automaticamente depois da mudança em `build_heat_snapshot`.

## Architecture / proposed approach
`build_heat_snapshot` lê os gates via `model.get_fast_states()` e anexa
`fast_heat` à entrada `ffn` correspondente (pareada por índice, com guarda de
tamanho para modelos sem FastHeat). No HTML, adicionamos a opção `fast_heat`
ao seletor `#heatMetric`; o `drawFFN` já renderiza o campo. Atualizamos apenas
telemetria + HTML + documentação + testes focados.

## Step-by-step tasks (TDD por tarefa)

### Tarefa 1 — Telemetria: incluir fast_heat no snapshot
Arquivo a editar: `experiments/live_telemetry.py`.

1. RED — adicionar teste em `tests/test_live_telemetry.py` (novo modelo com
   gate FastHeat + nova função):
   ```python
   from dual_heater.fast_heat import FastHeatGate

   class _FastHeatModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.ffn = nn.ModuleList([SlowHeatFFNTracker(5)])
           self.fast = nn.ModuleList([FastHeatGate(5, unit_dim=-1, config=None)])

       def get_ffn_trackers(self):
           return list(self.ffn)

       def get_fast_states(self):
           return list(self.fast)


   def test_heat_snapshot_includes_fast_heat_per_ffn_layer(tmp_path):
       model = _FastHeatModel()
       model.fast[0].fast_heat.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
       writer = TelemetryWriter(tmp_path, identity={"run": 9})
       snapshot = writer.publish_heat(
           model, context={"method": "dualheat", "stage": 0}
       )
       writer.close()
       assert snapshot["available"] is True
       assert snapshot["ffn"][0]["fast_heat"] == [0.1, 0.2, 0.3, 0.4, 0.5]
   ```
   Rode e veja falhar:
   ```
   python -m pytest -q tests/test_live_telemetry.py::test_heat_snapshot_includes_fast_heat_per_ffn_layer
   # esperado: KeyError 'fast_heat'
   ```

2. GREEN — em `build_heat_snapshot`, logo após o bloco que monta `snapshot["ffn"]`
   (linhas ~139-143 do arquivo atual), adicionar:
   ```python
       fast_getter = getattr(model, "get_fast_states", None)
       if callable(fast_getter):
           gates = list(fast_getter())
           for index, entry in enumerate(snapshot["ffn"]):
               if index < len(gates):
                   entry["fast_heat"] = _tensor_values(gates[index].fast_heat)
   ```
   Reposicione antes do `snapshot["available"] = bool(...)` final.

3. GREEN — roda o teste novo e a suíte de telemetria:
   ```
   python -m pytest -q tests/test_live_telemetry.py
   # esperado: all passing (6 testes; novo incluso)
   ```

### Tarefa 2 — Dashboard: opção FastHeat no seletor de sinal
Arquivo a editar: `experiments/live_dashboard.html` (bloco JS do `#heatMetric`,
~linha 144).

1. RED — em `tests/test_live_dashboard.py`, na função
   `test_dashboard_serves_local_read_only_api`, adicionar após o assert que lê
   `/` (o HTML serve o `#heatMetric`):
   ```python
       assert b'value="fast_heat"' in html
   ```
   Rode e veja falhar:
   ```
   python -m pytest -q tests/test_live_dashboard.py::test_dashboard_serves_local_read_only_api
   # esperado: AssertionError (option ausente)
   ```

2. GREEN — no HTML, no `<select id="heatMetric">`, adicionar uma opção após a
   de `heat`:
   ```html
   <option value="fast_heat">FastHeat</option>
   ```
   `drawFFN` já faz `const values = layer[metric] || []`, então selecionar
   FastHeat desenha o heat por neurônio da camada FFN selecionada sem mais
   mudanças.

3. Rode e veja passar:
   ```
   python -m pytest -q tests/test_live_dashboard.py
   # esperado: 3 testes passando
   ```

### Tarefa 3 — Documentação
Arquivo a editar: `docs/live_dashboard.md`.

1. Na seção "Dados apresentados", após o item de "importância atual ... por
   cabeça" / "todos os neurônios FFN", adicionar:
   ```text
   - FastHeat por neurônio FFN da camada selecionada (sinal `FastHeat`);
   ```
2. Na seção "Dados apresentados", onde descreve `heat-latest.json`, ajustar para
   mencionar que cada camada FFN pode carregar `fast_heat` opcionalmente.

## Tests / validation (resumo)
Comandos e saídas esperadas (rodar ao final de tudo):
```
cd /home/fachel/dual-heater-mlp-research

python -m pytest -q tests/test_live_telemetry.py tests/test_live_dashboard.py
# esperado: 10 passed (7 telemetria + 3 dashboard)

python -m pytest -q
# esperado: 266+ = suíte completa verde (os testes novos somam; conta total sobe)

python -m ruff check experiments/live_telemetry.py experiments/live_dashboard.py tests/test_live_telemetry.py tests/test_live_dashboard.py
# esperado: All checks passed!

git diff --check
# esperado: sem saída
```

Validação manual (opcional, exige um run com FastHeat — método `dualheat`):
```
python -m experiments.split_clinc150 --device cuda --seeds 11 \
  --batch-size 4 --replay-batch-size 4 \
  --methods dualheat --telemetry --telemetry-every 10 \
  --output-dir /tmp/bert_dualheat_live

python -m experiments.live_dashboard --run-dir /tmp/bert_dualheat_live --port 8765
```
Abrir `http://127.0.0.1:8765`, no painel "Heat funcional" escolher sinal
"FastHeat": o canvas FFN da camada selecionada deve pintar os neurônios com os
valores de `gate.fast_heat`. Atenção não muda (sem FastHeat por cabeça).

## Risks, tradeoffs, and open questions
- **Pareamento por índice**: `get_fast_states()` assume um gate por camada FFN,
  na ordem das camadas. Se no futuro existir FastHeat por cabeça de atenção ou
  múltiplos gates por camada, esse pareamento quebra silenciosamente. A guarda
  `if index < len(gates)` evita crash, mas pode parear na camada errada. Abrir
  um teste de integração BERT real (Tarefa 4 sugerida abaixo) mitiga.
- **Atenção sem FastHeat**: selecionar `FastHeat` na grade de atenção mostra
  células vazias (não há `fast_heat`). Aceitável; documentar que FastHeat é FFN.
- **Schema version**: mantido em 1. Snapshots antigos não têm `fast_heat`;
  o dashboard trata como vazio. Consistente com leitura tolerante já existente.
- **Tradeoff de escopo**: a opção `fast_heat` no `heatMetric` é compartilhada
  entre atenção e FFN. Alternativa mais isolada seria um painel próprio do
  FastHeat; maior escopo, não necessário agora (YAGNI).
- **Teste JS real**: pytest só valida que a opção está no HTML servido, não o
  comportamento de canvas. Decide-se aceitar; o teste E2E é manual.

## Suggested follow-up (não obrigatório nesta entrega)
Tarefa 4 — Integração BERT: em `tests/test_slow_heat_bert.py`, adicionar teste
que chama `build_heat_snapshot`/`publish_heat` num `SlowHeatBertForSequenceClassification`
com `fast_heat` e asserta `ffn[0]["fast_heat"]` com tamanho `intermediate_size`
e `max(abs)>0` após um forward. Isto fecha o laço "gate existe -> snapshot tem
fast_heat" no modelo real, reduzindo o risco do pareamento por índice.
