# Piloto exploratório: tempo da máscara (limite de atualizações e warmup da cabeça)

Status: EXPLORATÓRIO. Não é protocolo confirmatório; nenhum resultado daqui vira claim sem um confirmatório
com seeds novas. Pedido do Fachel em 09/11/2026 ("ideia para testarmos... pode rodar no Lab", "rode em
todas as gpus"). Escrito antes da run, com a leitura declarada abaixo.

## Motivação medida
- results/explore_hard_churn (seed 900, exploratório): o conjunto protegido do hard troca só 5-15% por
  fronteira, mas acumula: na última fronteira só 66% das unidades protegidas após a tarefa 1 continuam
  protegidas (acaso = 25%).
- F16 (refutado): a seleção dentro da máscara dura tem efeito pequeno e instável. A máscara dura bate o
  lr reduzido (F14).
- Literatura: o esquecimento se concentra nos primeiros passos de uma tarefa nova (stability gap, De
  Lange et al. 2023); treinar a cabeça antes do backbone reduz distorção de features (LP-FT, Kumar et
  al. 2022).

## Braços (todos E* = 0,75, configuração do F14-F16)
- hard_b0.75 (referência) e reduced_lr (controle pareado).
- hard_freeze1_b0.75: a máscara é calculada só na 1a fronteira e depois fica fixa.
- hard_freeze3_b0.75: recalculada nas 3 primeiras fronteiras, depois fixa.
- hard_warmup20_b0.75: nas tarefas 2 a 10, os 20 primeiros dos 120 passos treinam só a cabeça (grad None
  no resto); depois, hard normal.
- reduced_lr_warmup20: o mesmo warmup no controle pareado.
Código: experiments/qwen_iso_plasticity.py, branch exp/pilot-mask-timing, flags --pilot-freeze-after e
--pilot-head-warmup (opt-in); 8 testes novos, 8 mutações mortas.

## Seeds
500-509 (banda exploratória nova, verificada sem uso). Runner scripts/pilot_mask_timing.sh, saída
results/pilot_mask_timing/seed<k>/manifest.json.

## Leitura declarada antes da run
Endpoint principal do piloto: FAA (final_average_accuracy). Retenção da tarefa 1 só como secundário,
porque congelar a máscara depois da T1 favorece a T1 por construção. Guarda: aquisição da última tarefa
não pode cair mais de 2 pp.
- freeze: candidato a confirmatório se hard_freezeX - hard >= +2 pp de FAA, >= 8/10 seeds, guarda ok.
- warmup: candidato se (hard_warmup20 - reduced_lr_warmup20) - (hard - reduced_lr) >= +2 pp de FAA
  (o warmup ajuda a máscara mais do que ajuda o controle), >= 8/10, guarda ok; e também reportar
  hard_warmup20 - hard.
- Resto: descritivo (média, IC 95% pareado, seeds a favor). Sem correção múltipla: é exploratório.

## Limitações
Classificação, um modelo, um dataset, 120 passos, só E* = 0,75; X (1, 3) e N (20) escolhidos sem busca.
A retenção absoluta no regime está perto do piso (F16: hard 4,8%).
