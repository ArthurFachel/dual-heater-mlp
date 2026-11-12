# Protocolo confirmatório: Qwen2.5-0.5B, máscara dura selecionada vs aleatória FIXA (F16)

Status: CONGELADO antes de qualquer run (commit que adiciona este arquivo). Autor: Venus, autorizado
pelo Fachel ("já emende o próximo passo", 09/11/2026, depois do F15; autorização padrão para
confirmatórios de 07/11). Gatilho: F15 (docs/results/confirmatory_qwen_hard_permuted_results.md)
confirmou hard - hard_permuted = +4,10 pp de retenção t0 [+0,61; +7,59], p 0,026. Ressalva declarada lá:
o hard_permuted ressorteia o conjunto a cada fronteira, então nenhuma unidade fica congelada a sequência
toda, enquanto o conjunto do hard tende a ser estável. Um aleatório FIXO é o controle mais duro. Este
protocolo testa se a vantagem da seleção sobrevive a ele.

## Pergunta
Com E = 0,75, máscara binária e a mesma contagem de unidades protegidas por camada, congelar as unidades
mais importantes retém mais a 1a tarefa do que congelar um conjunto aleatório sorteado UMA vez por run e
mantido em todas as fronteiras?

## Braços
- hard_b0.75: igual ao F14/F15.
- hard_random_fixed_b0.75 (novo): depois de cada consolidate, a contagem k de unidades protegidas de cada
  camada é a mesma do hard (vem do mesmo orçamento); as k protegidas são as k primeiras de uma ordem
  aleatória por camada que depende só de (seed + 500000) e do índice da camada (fix_random_heat). Mesma
  ordem em todas as fronteiras: o conjunto é fixo (aninhado se k crescer).
- hard_permuted_b0.75: igual ao F15 (ressorteio a cada fronteira). Entra para o secundário fixo vs
  ressorteado, que testa a ressalva do F15 diretamente.

## Configuração (idêntica ao F15, só muda a lista de braços)
experiments/qwen_iso_plasticity.py, branch exp/f16-hard-random-fixed (base 45f0611 + braço opt-in
--hard-random-fixed-control e digest do conjunto protegido por fronteira; 5 testes novos, 8 mutações
mortas): --domains banking credit_cards kitchen_and_dining home auto_and_commute travel utility work
small_talk meta --steps-per-task 120 --learning-rate 1e-5 --batch-size 2 --max-length 64
--target-plasticity 0.75 --hard-permuted-control --hard-random-fixed-control
--arms hard_b0.75 hard_random_fixed_b0.75 hard_permuted_b0.75 --device cuda (fp32, capacity local).
Um processo por seed, os 3 braços no mesmo processo e GPU. Runner scripts/f16_confirm_qwen_hard_random_fixed.sh;
saída results/f16_confirm_qwen_hard_random_fixed/seed<k>/manifest.json.

## Seeds (novas)
400, 401, ..., 409 (10 seeds). Verificado em 09/11 por grep em scripts/goals/protocolos e diretórios de
resultados dos 4 worktrees: nenhum uso.

## Endpoint primário
D = retention_first_task(hard_b0.75) - retention_first_task(hard_random_fixed_b0.75), pareado por seed.

## Análise (scripts/analyze_f16.py)
- t pareado bicaudal de D, alfa 0,05 (contraste único).
- CONFIRMADO: média de D > 0 e p < 0,05 (a seleção vence também o aleatório fixo).
- REFUTADO: IC 95% inclui 0 ou média <= 0. Senão inconclusivo.
- Secundários (descritivos): sinal; FAA e aquisição (hard - fixo); hard_random_fixed - hard_permuted na
  retenção (fixo vs ressorteado: se > 0, o controle do F15 era mais fraco, como suspeitado); hard -
  hard_permuted (réplica do F15).
- Integridade (não é endpoint; se falhar, run inválida, sem reinterpretação): máscara hard e mesma contagem
  protegida nos 3 braços em todas as fronteiras; random_fixed = True e protected_digest constante entre
  fronteiras no braço fixo sempre que a contagem não mudou; protected_digest do fixo diferente do hard;
  permutation_reordered = True no hard_permuted.
- Nenhuma seed descartada; run que falhar é rerodada com a mesma seed e o fato é reportado.

## Validação da análise antes da run
Sem dado exploratório do braço fixo. Validada (1) nos manifests do F15, onde reproduz hard -
hard_permuted = +4,10 pp, e (2) num smoke curto com os 3 braços para as chaves e a integridade.

## Limitações declaradas
As do F14/F15. O sorteio é dentro de cada camada (perfil por camada igual ao do hard). Um sorteio por
seed: a variância do conjunto aleatório entra pelo pareamento por seed. O F15 mostrou
efeito concentrado em 3 de 10 seeds e 3 seeds no piso; o mesmo pode acontecer aqui e reduz o poder.
