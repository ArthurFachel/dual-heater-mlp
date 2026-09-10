# Dashboard ao vivo do BERT com SlowHeat

O dashboard acompanha o treinamento sem controlar ou alterar o experimento.
O runner grava eventos numéricos e estados destacados do grafo; um processo
HTTP separado lê esses artefatos em `localhost`. Nenhum texto, token ou matriz
de atenção entre palavras é registrado.

## Execução

No primeiro terminal:

```bash
python -m experiments.split_clinc150 \
  --device cuda \
  --seeds 11 \
  --batch-size 4 \
  --replay-batch-size 4 \
  --methods replay slowheat_replay \
  --telemetry \
  --telemetry-every 10 \
  --output-dir results/bert_mini_live
```

No segundo terminal:

```bash
python -m experiments.live_dashboard \
  --run-dir results/bert_mini_live \
  --port 8765
```

Acesse `http://127.0.0.1:8765`. O servidor aceita somente requisições de
leitura e permanece vinculado ao endereço local. Fechar o painel não interrompe
o treinamento; reabri-lo recupera todo o histórico disponível.

## Dados apresentados

- progresso por seed, método, tarefa, época e lote;
- perda atual e média móvel, learning rate, tokens por segundo e ETA;
- memória CUDA alocada, reservada e pico;
- importância atual, memória consolidada, Heat e plasticidade por cabeça;
- todos os neurônios FFN da camada selecionada, incluindo o FastHeat;
- FastHeat por neurônio FFN da camada selecionada (sinal `FastHeat`);
- fração protegida por família e camada;
- eventos de consolidação, avaliação e checkpoint;
- matriz tarefa por estágio e curvas comparativas entre métodos.

Os escalares são acrescentados a `telemetry/events.jsonl`. O estado completo
mais recente fica em `telemetry/heat-latest.json`, e cada fronteira de tarefa
gera uma fotografia imutável em `telemetry/heat/`. Cada camada FFN pode carregar
um campo opcional `fast_heat` (valores do `FastHeatGate.fast_heat` por neurônio)
quando o modelo possui FastHeat; camadas sem FastHeat omitem esse campo.

O evento `method_end` registra `telemetry_overhead_seconds` e
`telemetry_overhead_ratio`. Isso permite conferir, em cada GPU e configuração
real, a meta operacional de overhead máximo de 5% para `--telemetry-every 10`.
O painel e o emissor usam somente a biblioteca padrão do Python e o JavaScript
do navegador; não há serviço externo nem dependência obrigatória adicional.

## Retomada e falhas

Use `--resume` normalmente. A nova sessão continua a sequência de eventos e
mantém o mesmo identificador de protocolo. Um diretório de telemetria com outra
configuração, outro tokenizer ou outros dados é rejeitado. A última linha JSONL
incompleta, possível após interrupção abrupta, é ignorada pelo leitor.

Erros não são ocultados: o runner registra o tipo e a mensagem resumida antes
de propagar a exceção original. O painel permanece disponível para inspecionar
os dados gravados antes da falha.
