# Resultados: seleção iso-plasticidade em sequência de 10 tarefas

**Estado:** run de confirmação concluída em 23/09/2026.
**Protocolo:** `goals/protocol_iso_plasticity.md` (congelado em 22/09, emendado
para 120 passos em 22/09 — ver tabela K).
**Dados:** `results/qwen_iso_plasticity/confirm120_seed{10..19}/manifest.json`
**Reprodução:** `PYTHONPATH=. python experiments/analyze_confirmation.py`

Integridade verificada antes de agregar: `protocol_hash` idêntico nas 10 seeds
(`2cd785d8...`), `task_fingerprint` idêntico, fp32, 120 passos, 10 tarefas,
escopo `local`. O agregador se recusa a rodar se qualquer um divergir.

---

## A pergunta

Com a **mesma quantidade de plasticidade efetiva removida** (E fixo), importa
*quais* unidades são protegidas?

O braço `permuted` existe para isolar exatamente isso: mesmo E, mesmo β, mesmo
número de unidades protegidas, mas a identidade das unidades é embaralhada. Se
a seleção aprendida não bater a permutada, o mecanismo está medindo apenas
"quanta" plasticidade foi removida, não "onde".

---

## Resultado principal

### E\* = 0,75 (10 seeds, 10 tarefas, 120 passos/tarefa)

| braço | FAA | retenção t0 | forgetting | aquisição t_últ |
|---|---|---|---|---|
| vanilla | 0,407 ± 0,026 | 0,020 | 0,523 | 0,820 |
| permutado b=0,25 | 0,406 ± 0,036 | 0,034 | 0,521 | 0,823 |
| reduced_lr | 0,424 ± 0,044 | 0,032 | 0,494 | 0,815 |
| iso b=0,25 | 0,433 ± 0,047 | 0,046 | 0,492 | 0,827 |
| **hard b=0,75** | **0,458 ± 0,040** | **0,095** | **0,461** | 0,821 |

### E\* = 0,50

| braço | FAA | retenção t0 | forgetting | aquisição t_últ |
|---|---|---|---|---|
| vanilla | 0,407 | 0,020 | 0,523 | 0,820 |
| iso b=0,25 | 0,389 | 0,049 | 0,531 | 0,818 |
| reduced_lr | 0,394 | 0,028 | 0,510 | 0,808 |
| **hard b=0,50** | **0,447** | **0,114** | **0,465** | 0,814 |

`iso_b0.5` foi descartado em todas as seeds nessa família: o budget 0,50 não
alcança E\*=0,50 na fronteira. Motivo registrado em cada manifesto.

---

## O que sobrevive à correção para múltiplos testes

São 24 comparações pareadas. Sem correção, 11 dão p < 0,05 — mas esperar ~1
falso positivo por acaso é o padrão nesse volume. Sob Holm (FWER), **2 de 24**
sobrevivem:

| E\* | comparação | endpoint | diferença | seeds | p | p_Holm |
|---|---|---|---|---|---|---|
| 0,75 | hard − vanilla | retenção t0 | **+0,075** | 10/10 | 0,0020 | **0,047** |
| 0,50 | hard − vanilla | retenção t0 | **+0,094** | 10/10 | 0,0020 | **0,047** |

As duas são o braço hard contra o controle sem proteção, na retenção da
primeira tarefa, com concordância unânime das 10 seeds.

### Sinais consistentes que não atingem significância corrigida

| E\* | comparação | endpoint | diferença | seeds | p | p_Holm |
|---|---|---|---|---|---|---|
| 0,50 | hard − iso | retenção t0 | +0,065 | 9/10 | 0,0039 | 0,086 |
| 0,75 | hard − iso | retenção t0 | +0,050 | 9/10 | 0,0039 | 0,086 |
| 0,50 | hard − iso | FAA | +0,059 | 9/10 | 0,0059 | 0,111 |
| 0,75 | hard − vanilla | FAA | +0,051 | 9/10 | 0,0059 | 0,111 |
| 0,75 | iso − permutado | FAA | +0,027 | 8/10 | 0,037 | 0,316 |
| 0,50 | iso − permutado | retenção t0 | +0,018 | 8/10 | 0,0078 | 0,125 |

---

## Leitura honesta

**1. O braço hard funciona.** É o único resultado que sobrevive à correção, e
sobrevive nos dois alvos de plasticidade com 10/10 seeds. Retenção da primeira
tarefa sobe de 0,020 para 0,095-0,114 — de essencialmente acaso para 5x o
acaso. O custo em aquisição da última tarefa é nulo (0,821 vs 0,820).

**2. A hipótese central ficou sem suporte estatístico.** `iso − permutado` é a
comparação que responde "importa *quais* unidades?". Ela dá o sinal certo
(positivo em FAA e retenção, negativo em forgetting, 8/10 seeds em três dos
seis testes), mas nenhuma sobrevive a Holm. Com 2 tarefas a diferença era
+0,080 de retenção em 3/3 seeds; com 10 tarefas caiu para +0,012 a +0,018.

**3. "Quanto" passou a importar mais que "quais".** O hard bate o iso soft com
o mesmo E (+0,050 e +0,065 de retenção, 9/10 seeds), e a diferença hard-vs-iso
é maior que a diferença iso-vs-permutado. Como ambos protegem as *mesmas*
unidades e diferem só na dureza da máscara, o ganho vem da forma da proteção,
não da escolha do alvo.

**4. O P6 explica o item 2.** A sobreposição de unidades dominantes entre
tarefas é de 0,679 (top-10), contra 0,002 de acaso. Se as tarefas disputam
largamente as mesmas unidades, embaralhar a seleção com E fixo acaba protegendo
um conjunto parecido — e a vantagem da seleção aprendida encolhe conforme a
sequência cresce.

---

## Limites

- **Poder estatístico no talo.** Com 10 seeds o menor p possível no teste exato
  é 2/1024 = 0,00195. Sobre 24 comparações, Holm multiplica por 24 e dá 0,047.
  Ou seja: mesmo um efeito perfeito em 10/10 seeds mal passa de 0,05. Uma
  família de 26 comparações não conseguiria produzir *nenhum* resultado
  significativo. Para afirmar algo sobre `iso − permutado` seria preciso mais
  seeds, não mais braços.
- **Regime de esquecimento severo.** Forgetting de ~0,46 no melhor braço.
- **Duas tarefas ≠ dez.** O resultado forte de 2 tarefas (+0,080, 3/3) não se
  sustentou em 10. A conclusão de 2 tarefas era otimista.
- **120 passos não é convergência.** É o joelho da curva (0,78-0,81 contra teto
  de ~0,94). Um orçamento maior pode mudar o quadro.

---

## Próximos passos possíveis

1. **Mais seeds para `iso − permutado`.** É a hipótese central e está no limite
   do poder. Restringir a família de comparações a essa pergunta (2-3 testes em
   vez de 24) já mudaria o limiar de Holm de 0,047 para ~0,006.
2. **Sequências intermediárias** (4, 6 tarefas) para mapear onde a vantagem da
   seleção aprendida se dissolve.
3. **Investigar o hard.** É o efeito mais forte e o menos explorado: só foi
   testado em `budget = E*`, porque sob máscara hard E = budget exatamente.
