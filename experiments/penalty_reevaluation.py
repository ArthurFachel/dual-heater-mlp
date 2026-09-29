"""Constantes do piloto de re-avaliação de métodos de penalidade.

`goals/protocol_penalty_reevaluation.md` §C.2.

Existe separado de `split_mnist.py` por um motivo de pré-registro, não de
organização: `ewc_lambda` e `si_lambda` entram em `config_payload()` mesmo
quando o método não é usado, então mudar o default de `SplitMNISTConfig`
moveria o sha256 do protocolo congelado em `confirmatory_split_mnist.py`.
Os valores publicados vivem aqui e são aplicados só por este piloto.
"""

from __future__ import annotations

from typing import Any

__all__ = ["HSU_2018_CLASS_IL_REG_COEF", "PUBLISHED_PENALTY_STRENGTHS"]

#: `reg_coef` publicado por Hsu et al. (2018) para **Split-MNIST
#: class-incremental**, lido de `scripts/split_MNIST_incremental_class.sh` no
#: repositório GT-RIPL/Continual-Learning-Benchmark.
#:
#: Referência:
#:   Yen-Chang Hsu, Yen-Cheng Liu, Anita Ramasamy, Zsolt Kira.
#:   "Re-evaluating Continual Learning Scenarios: A Categorization and Case for
#:   Strong Baselines." NeurIPS Continual Learning Workshop, 2018.
#:   arXiv:1810.12488
#:
#: Cenário do script: MLP400, Adam `lr=1e-3`, batch 128, 4 épocas por tarefa,
#: 10 repetições — o mesmo cenário deste piloto.
#:
#: **EWC: 100, não 600.** Hsu publica os dois. `EWC_mnist` (600) guarda um
#: termo de regularização POR TAREFA (`online_reg = False`);
#: `EWC_online_mnist` (100) mantém um Fisher acumulado único
#: (`online_reg = True`). `consolidate_importance()` deste repo acumula sobre
#: um único dicionário (`decay * importance + estimate`), que é a variante
#: online — logo o número comparável é 100.
#:
#: Os valores do cenário **task-incremental** são outros (EWC 100, online 400,
#: SI 300) e não se aplicam aqui: este piloto é class-IL.
HSU_2018_CLASS_IL_REG_COEF: dict[str, float] = {
    "ewc": 100.0,   # EWC_online_mnist
    "si": 600.0,
    "mas": 1.0,
}

#: Os mesmos valores traduzidos para os campos de `SplitMNISTConfig`.
#:
#: **A tradução não é identidade, e confundir as duas convenções é o erro que
#: este comentário existe para impedir.** Hsu soma `reg_coef * Σ Ω (θ−θ*)²`,
#: sem o fator 1/2. Este runner soma:
#:
#:     EWC:  0.5 * ewc_lambda * Σ Ω (θ−θ*)²   -> k = 0.5 * ewc_lambda
#:     SI:         si_lambda  * Σ Ω (θ−θ*)²   -> k = si_lambda
#:     MAS:        mas_lambda * Σ Ω (θ−θ*)²   -> k = mas_lambda
#:
#: Logo `ewc_lambda = 200` produz `k = 100`, que é o `reg_coef` publicado.
#: Pinado por `tests/test_penalty_reevaluation_seeds.py`, que compara os `k`
#: efetivos via `_penalty_scale()` em vez de comparar os λ de olho.
#:
#: `decay = 1.0` é a soma acumulada clássica do EWC online, sem esquecimento
#: progressivo de tarefas antigas.
PUBLISHED_PENALTY_STRENGTHS: dict[str, Any] = {
    "ewc_lambda": 200.0,   # k = 100 = EWC_online_mnist de Hsu
    "ewc_decay": 1.0,
    "si_lambda": 600.0,    # k = 600
    "mas_lambda": 1.0,     # k = 1
    "mas_decay": 1.0,
}
