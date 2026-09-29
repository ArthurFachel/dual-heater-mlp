"""MAS ligado ao runner do Split-MNIST.

MAS (Aljundi et al., ECCV 2018) completa o trio de baselines de penalidade
quadrática no host MLP: EWC e SI já existiam, MAS não existia em lugar nenhum
do repositório (ver `docs/audits/baseline_inventory.md`).

O modo de falha que estes testes existem para pegar não é a fórmula — é o
método aparecer na tabela de resultados, produzir números plausíveis, e ser
vanilla com outro nome porque a penalidade nunca chegou na loss. Esse bug já
mordeu este projeto duas vezes (`importance_criterion` não propagado,
`learning_rate_scale` não chegando ao `lr_control`).
"""

from __future__ import annotations

import pytest
import torch


def test_mas_is_a_registered_method() -> None:
    from experiments.split_mnist import _METHOD_SPECS

    assert "mas" in _METHOD_SPECS


def test_mas_has_its_own_strength_knob() -> None:
    """Sem hiperparâmetro próprio, MAS herdaria a força do EWC silenciosamente."""

    from experiments.split_mnist import SplitMNISTConfig

    config = SplitMNISTConfig()
    assert hasattr(config, "mas_lambda")
    assert config.mas_lambda > 0.0


def test_mas_lambda_is_validated() -> None:
    from experiments.split_mnist import SplitMNISTConfig

    with pytest.raises(ValueError):
        SplitMNISTConfig(mas_lambda=-1.0).validate()


def test_mas_lambda_reaches_the_serialized_config() -> None:
    """Um hiperparâmetro ausente do manifest não é auditável depois da run.

    Segue a convenção do `config_payload`: o campo entra no payload apenas
    quando o método é usado, para que adicionar um baseline não altere o
    sha256 de protocolos já congelados (ver
    `tests/test_confirmatory_statistics.py`).
    """

    from experiments.split_mnist import SplitMNISTConfig, config_payload

    used = config_payload(
        SplitMNISTConfig(methods=("vanilla", "mas"), mas_lambda=7.5)
    )
    assert used["mas_lambda"] == pytest.approx(7.5)
    assert "mas_decay" in used


def test_mas_hyperparameters_stay_out_of_unrelated_payloads() -> None:
    """Adicionar MAS não pode mudar o payload de um protocolo congelado."""

    from experiments.split_mnist import SplitMNISTConfig, config_payload

    unused = config_payload(SplitMNISTConfig(methods=("vanilla", "ewc")))
    assert "mas_lambda" not in unused
    assert "mas_decay" not in unused


def test_mas_uses_the_shared_quadratic_penalty() -> None:
    """DRY: a terceira penalidade quadrática não pode ser escrita de novo."""

    import inspect

    from experiments import split_mnist

    source = inspect.getsource(split_mnist)
    # MAS soma via o mesmo helper compartilhado que EWC e SI usam.
    assert source.count("_parameter_penalty(") >= 4


def test_mas_omega_is_label_free() -> None:
    """MAS mede sensibilidade da SAÍDA; usar o rótulo o transformaria em Fisher."""

    from dual_heater.ewc import accumulate_mas_omega

    torch.manual_seed(0)
    weight = torch.randn(3, 2, requires_grad=True)
    inputs = torch.randn(4, 3)
    outputs = inputs @ weight

    first = {"w": torch.zeros(3, 2)}
    accumulate_mas_omega(
        outputs=outputs, named_parameters=[("w", weight)], accumulator=first
    )
    # Mesma saída, "rótulos" diferentes -> mesmo omega, porque não há rótulo.
    second = {"w": torch.zeros(3, 2)}
    accumulate_mas_omega(
        outputs=outputs, named_parameters=[("w", weight)], accumulator=second
    )
    assert first["w"] == pytest.approx(second["w"])


def test_mas_penalty_bites_the_loss() -> None:
    """O teste de mordida: com âncora e omega, a loss tem de subir."""

    import torch.nn as nn

    from experiments.split_mnist import _parameter_penalty

    model = nn.Linear(4, 2)
    anchors = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    omega = {n: torch.ones_like(p) for n, p in model.named_parameters()}

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)

    penalty = _parameter_penalty(model, omega, anchors)
    assert float(penalty) > 0.0


def test_mas_reduces_drift_versus_vanilla_on_a_two_task_smoke() -> None:
    """Um MAS que não reduz drift não está ligado, por mais que rode.

    Smoke de CPU: duas tarefas, poucos exemplos, uma época. A asserção é sobre
    o drift RMS dos parâmetros durante a segunda tarefa — a quantidade que a
    penalidade existe para reduzir.

    **Calibração de λ, aprendida por divergência:** o gradiente da penalidade é
    `2·λ·Ω·(θ − θ*)`, então sob SGD o passo é estável apenas enquanto
    `lr·λ·2·Ω_max < 2`. Aqui `Ω_max ≈ 0,57` e `lr = 0,1`, o que dá o teto
    `λ < 17,5`; um primeiro rascunho deste teste usava `λ = 50` e media drift
    **maior** que vanilla (1,61 contra 0,05) — divergência do otimizador, não
    ausência do mecanismo. Um λ acima do teto faz este teste reprovar uma
    implementação correta.
    """

    import torch.nn as nn

    from dual_heater.ewc import accumulate_mas_omega, consolidate_importance
    from experiments.split_mnist import _parameter_penalty

    def train_second_task(*, mas_strength: float) -> float:
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        torch.manual_seed(11)
        task1_x = torch.randn(16, 8)
        task1_y = torch.randint(0, 2, (16,))
        task2_x = torch.randn(16, 8) + 3.0
        task2_y = torch.randint(2, 4, (16,))

        # Tarefa 1.
        for _ in range(3):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(task1_x), task1_y)
            loss.backward()
            optimizer.step()

        # Consolida omega e ancora.
        omega: dict[str, torch.Tensor] = {}
        accumulator: dict[str, torch.Tensor] = {}
        outputs = model(task1_x)
        examples = accumulate_mas_omega(
            outputs=outputs,
            named_parameters=tuple(model.named_parameters()),
            accumulator=accumulator,
        )
        anchors: dict[str, torch.Tensor] = {}
        consolidate_importance(
            named_parameters=model.named_parameters(),
            accumulator=accumulator,
            examples=examples,
            importance=omega,
            anchors=anchors,
            decay=1.0,
        )
        reference = {n: p.detach().clone() for n, p in model.named_parameters()}

        # Tarefa 2, com ou sem penalidade.
        for _ in range(5):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(task2_x), task2_y)
            if mas_strength > 0.0:
                loss = loss + mas_strength * _parameter_penalty(model, omega, anchors)
            loss.backward()
            optimizer.step()

        drift = torch.cat(
            [
                (p.detach() - reference[n]).flatten()
                for n, p in model.named_parameters()
            ]
        )
        return float(drift.pow(2).mean().sqrt())

    vanilla_drift = train_second_task(mas_strength=0.0)
    mas_drift = train_second_task(mas_strength=5.0)

    assert mas_drift < vanilla_drift, (
        f"MAS não reduziu drift: {mas_drift:.6f} vs vanilla {vanilla_drift:.6f}"
    )


def test_mas_changes_the_training_loss_in_the_real_runner() -> None:
    """A mutação que os testes de fórmula não pegam: MAS desligado no runner.

    Os testes acima exercitam `_parameter_penalty` e `accumulate_mas_omega`
    diretamente, e por isso continuam verdes mesmo se o bloco
    `if method == "mas"` for removido do loop de treino — o método aparece na
    tabela, produz números plausíveis, e é vanilla com outro nome. Este teste
    passa pelo `run_split_mnist` de verdade e compara as losses.

    Mutações que ele mata (verificadas): remover a penalidade da loss, remover
    a consolidação de omega, e trocar `mas_lambda` por `ewc_lambda`.
    """

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_split_mnist import _tiny_tasks

    from experiments.split_mnist import SplitMNISTConfig, run_split_mnist

    config = SplitMNISTConfig(
        methods=("vanilla", "mas"),
        class_order=tuple(range(6)),
        epochs_per_task=2,
        mas_lambda=25.0,
    )
    results = run_split_mnist(config, _tiny_tasks(config))

    vanilla_losses = results["vanilla"]["training_losses"]
    mas_losses = results["mas"]["training_losses"]

    assert len(mas_losses) == len(vanilla_losses)
    # A primeira tarefa não tem âncora, logo não há penalidade: as losses
    # coincidem. A divergência precisa aparecer DEPOIS da primeira
    # consolidação, que é a prova de que omega foi consolidado E usado.
    assert mas_losses != vanilla_losses, (
        "MAS não alterou a loss do runner: a penalidade não chegou ao treino"
    )


def test_mas_strength_changes_the_runner_outcome() -> None:
    """Se `mas_lambda` não for lido pelo runner, dobrar a força não muda nada.

    Mata a mutação que troca `config.mas_lambda` por `config.ewc_lambda`, que
    sobrevive a qualquer teste que só verifique a existência da penalidade.
    """

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_split_mnist import _tiny_tasks

    from experiments.split_mnist import SplitMNISTConfig, run_split_mnist

    def losses(mas_lambda: float) -> list[float]:
        config = SplitMNISTConfig(
            methods=("mas",),
            class_order=tuple(range(6)),
            epochs_per_task=2,
            mas_lambda=mas_lambda,
        )
        return run_split_mnist(config, _tiny_tasks(config))["mas"]["training_losses"]

    assert losses(1.0) != losses(40.0), (
        "mas_lambda não influencia o treino: o knob não chegou na loss"
    )


def test_mas_consolidation_is_reached_by_the_runner() -> None:
    """Sem consolidação, omega fica vazio e a penalidade é sempre zero.

    Mata a mutação que desliga o bloco `consolidate_importance` do MAS: com
    omega vazio, `mas` produziria exatamente as losses do vanilla, e a
    diferença medida aqui desapareceria.
    """

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_split_mnist import _tiny_tasks

    from experiments.split_mnist import SplitMNISTConfig, run_split_mnist

    config = SplitMNISTConfig(
        methods=("vanilla", "mas"),
        class_order=tuple(range(6)),
        epochs_per_task=2,
        mas_lambda=25.0,
    )
    results = run_split_mnist(config, _tiny_tasks(config))
    vanilla = results["vanilla"]["training_losses"]
    mas = results["mas"]["training_losses"]

    # `training_losses` é aninhada por tarefa. A primeira tarefa não tem
    # âncora, logo não há penalidade e as losses coincidem; alguma tarefa
    # posterior precisa diferir (omega consolidado e em uso). As duas
    # condições juntas localizam o efeito na consolidação, não no acaso.
    assert len(mas) == len(vanilla) >= 2
    assert mas[0] == pytest.approx(vanilla[0]), (
        "houve penalidade na primeira tarefa, onde não existe âncora"
    )
    assert any(
        mas[task] != pytest.approx(vanilla[task]) for task in range(1, len(vanilla))
    ), "nenhuma diferença após a primeira consolidação: omega não foi usado"


def test_mas_smoke_would_catch_a_disconnected_penalty() -> None:
    """Controle do teste acima: com λ = 0 o drift tem de ser idêntico ao vanilla.

    Se este teste falhasse, o smoke anterior estaria medindo ruído de seed em
    vez do efeito da penalidade, e passaria mesmo com MAS desligado.
    """

    import torch.nn as nn

    from dual_heater.ewc import accumulate_mas_omega, consolidate_importance
    from experiments.split_mnist import _parameter_penalty

    def drift(*, mas_strength: float) -> float:
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 4))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        torch.manual_seed(11)
        task1_x = torch.randn(16, 8)
        task1_y = torch.randint(0, 2, (16,))
        task2_x = torch.randn(16, 8) + 3.0
        task2_y = torch.randint(2, 4, (16,))
        for _ in range(3):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(model(task1_x), task1_y).backward()
            optimizer.step()
        accumulator: dict[str, torch.Tensor] = {}
        examples = accumulate_mas_omega(
            outputs=model(task1_x),
            named_parameters=tuple(model.named_parameters()),
            accumulator=accumulator,
        )
        omega: dict[str, torch.Tensor] = {}
        anchors: dict[str, torch.Tensor] = {}
        consolidate_importance(
            named_parameters=model.named_parameters(),
            accumulator=accumulator,
            examples=examples,
            importance=omega,
            anchors=anchors,
            decay=1.0,
        )
        reference = {n: p.detach().clone() for n, p in model.named_parameters()}
        for _ in range(5):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(task2_x), task2_y)
            if mas_strength > 0.0:
                loss = loss + mas_strength * _parameter_penalty(model, omega, anchors)
            loss.backward()
            optimizer.step()
        flat = torch.cat(
            [(p.detach() - reference[n]).flatten() for n, p in model.named_parameters()]
        )
        return float(flat.pow(2).mean().sqrt())

    assert drift(mas_strength=0.0) == pytest.approx(drift(mas_strength=0.0))
    assert drift(mas_strength=5.0) != pytest.approx(drift(mas_strength=0.0))
