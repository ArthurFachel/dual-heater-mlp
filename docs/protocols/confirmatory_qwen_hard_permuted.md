# Protocolo confirmatório: Qwen2.5-0.5B, máscara dura em unidades selecionadas vs aleatórias (F15)

Status: CONGELADO antes de qualquer run (commit que adiciona este arquivo). Autor: Venus, autorizado
pelo Fachel ("1, 2", 09/11/2026, sobre os próximos passos; autorização padrão para confirmatórios de
07/11). Gatilho: F14 (docs/results/confirmatory_qwen_hard_results.md) confirmou hard_b0.75 - reduced_lr
= +5,67 pp de retenção da 1a tarefa, mas o hard difere do reduced_lr em duas coisas ao mesmo tempo: QUAIS
unidades perdem plasticidade (as mais importantes) e COMO (binário, 0 ou 1). Este protocolo isola a primeira.

## Pergunta
Com a mesma plasticidade efetiva (E = 0,75) e a mesma máscara binária, congelar as 25% unidades FFN mais
importantes de cada camada retém mais a primeira tarefa do que congelar 25% unidades escolhidas ao acaso?

## Braços
- hard_b0.75: igual ao F14 (top 25% por importância, por camada, máscara binária).
- hard_permuted_b0.75 (novo): mesmo consolidate e mesmo orçamento; logo depois de cada consolidação o vetor
  slow_heat de cada camada é permutado (permute_slowheat_heat, semente seed + 1000*fronteira), então o
  número de unidades congeladas por camada é idêntico e só a identidade muda. A permutação é refeita em
  cada fronteira, porque o consolidate reconstrói slow_heat a partir da importância acumulada. Mesma
  construção do braço permuted soft da confirm120.
- reduced_lr: igual ao F14 (lr x 0,75, sem máscara). Replica o F14 em seeds novas (secundário).

Sem vanilla: o F14 já ancorou o esquecimento e o braço não entra em nenhum contraste deste protocolo.

## Configuração (idêntica ao F14, só muda a lista de braços)
experiments/qwen_iso_plasticity.py, branch exp/f15-hard-permuted (base d67a536 + braço opt-in
--hard-permuted-control; testes em tests/test_qwen_iso_plasticity.py, 4 mutações mortas):
--domains banking credit_cards kitchen_and_dining home auto_and_commute travel utility work small_talk meta
--steps-per-task 120 --learning-rate 1e-5 --batch-size 2 --max-length 64 --target-plasticity 0.75
--hard-permuted-control --arms hard_b0.75 hard_permuted_b0.75 reduced_lr --device cuda (fp32, capacity
local, beta per_boundary). Um processo por seed, os 3 braços no mesmo processo e GPU.
Runner scripts/f15_confirm_qwen_hard_permuted.sh; saída results/f15_confirm_qwen_hard_permuted/seed<k>/manifest.json.

## Seeds (novas)
300, 301, ..., 309 (10 seeds). Verificado em 09/11: nenhum script, protocolo, experimento ou diretório de
resultados do Qwen usa 300-309 (F14 usou 200-209, confirm120 10-19).

## Endpoint primário
D = retention_first_task(hard_b0.75) - retention_first_task(hard_permuted_b0.75), pareado por seed.

## Análise (scripts/analyze_f15.py)
- Teste t pareado bicaudal de D, alfa 0,05 (contraste único, sem correção).
- CONFIRMADO: média de D > 0 e p < 0,05 (a seleção importa sob máscara dura).
- REFUTADO: IC 95% inclui 0 ou média <= 0 (o ganho do F14 vem da forma binária/esparsa, não da escolha).
- Senão inconclusivo.
- Secundários (descritivos): sinal; FAA e aquisição da última tarefa (hard - hard_permuted);
  hard - reduced_lr (réplica do F14) e hard_permuted - reduced_lr na retenção.
- Checagem de integridade (não é endpoint): permutation_reordered = True e protected_units iguais entre
  hard e hard_permuted em todas as fronteiras de todas as seeds; se falhar, a run é inválida e é reportada
  assim, sem reinterpretação.
- Nenhuma seed descartada; run que falhar é rerodada com a mesma seed e o fato é reportado.

## Validação da análise antes da run
Não existe dado exploratório do braço hard_permuted. A análise foi validada (1) nos manifests do F14, onde
reproduz hard_b0.75 - reduced_lr = +5,67 pp, e (2) num manifest de smoke em CPU com os três braços, para
as chaves e a checagem de integridade. Por isso o tamanho do efeito não tem estimativa prévia; 10 seeds é
o mesmo n do F14.

## Limitações declaradas
As do F14 (classificação class-IL, um modelo, um dataset, 120 passos por tarefa, só E* = 0,75). A
permutação sorteia unidades dentro de cada camada (o perfil por camada fica igual); não testa seleção entre
camadas. Uma permutação por seed e fronteira: o controle aleatório tem variância própria, absorvida no
pareamento por seed. GPUs 1080 Ti e Titan Xp misturadas entre seeds (não dentro de uma seed).
