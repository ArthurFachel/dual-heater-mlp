# Aplicações de LoRA no projeto

Índice das quatro formas de LoRA implementadas no repositório, com o estado da
evidência de cada uma. Cada linha aponta para o documento que traz o contrato,
os testes e — quando existe — os números.

Última atualização: 25 de setembro de 2026 (run iso-plasticidade concluída).

| # | Aplicação | Onde | Unidade protegida | Exatidão | Estado |
|---|---|---|---|---|---|
| 1 | DualHeat-LoRA (legado) | `src/dual_heater/lora.py` | saída, via hook no `delta` | **nenhuma** — `A` compartilhada vaza | [lora_dualheat_legacy.md](lora_dualheat_legacy.md) |
| 2 | LoRA exato produtor-only | `src/dual_heater/bert.py:1055` | saída, com `A` congelada | exata por saída | [lora_exact_producer_only.md](lora_exact_producer_only.md) |
| 3 | Três mecanismos SlowHeat-em-LoRA | `src/dual_heater/lora_slowheat.py` | rank / saída / fatia | ver documento | [lora_slowheat_mechanisms.md](lora_slowheat_mechanisms.md) |
| 4 | Benchmark em Qwen2.5-0.5B (2 runs × 10 seeds) | `experiments/qwen_lora_sweep.py` | — | — | [lora_qwen_benchmark_results.md](lora_qwen_benchmark_results.md) |

## O problema comum às quatro

LoRA representa o update como `delta_weight = B @ A`, com
`A.shape = [rank, in_features]` e `B.shape = [out_features, rank]`. Para a
saída `i`:

```text
delta_weight[i, :] = B[i, :] @ A
```

Congelar `B[i, :]` não protege a saída `i`, porque `A` é compartilhada por
todas as linhas: um update induzido por qualquer outra saída altera `A` e com
isso altera `delta_weight[i, :]`. **Toda alegação de proteção em LoRA precisa
declarar qual unidade ela cobre**, e as quatro aplicações acima respondem isso
de formas diferentes.

## Resumo do que se sabe hoje

- A aplicação 1 **não** oferece a proteção que a sua própria docstring alega, e
  isso está documentado no módulo desde a origem.
- A aplicação 2 é a única com efeito estatisticamente sobrevivente: reduz
  forgetting em 10/10 seeds contra o controle de learning rate (−0,119,
  p=0,002, sobrevive a Holm), com o maior FAA de todos os braços.
- Os três mecanismos da aplicação 3 **falharam** nas duas runs. Sob pareamento
  de plasticidade eles ficam **abaixo do vanilla** e não superam
  significativamente o controle de LR.
- A primeira run da aplicação 4 estava **confundida por capacidade** (`E_eff` de
  0,062 a 0,923, com o FAA seguindo essa ordenação). A segunda run pareia todos
  os braços em `E = 0,85` e acrescenta o braço `lr_control`. O efeito da
  aplicação 2 **aumentou** sob o controle mais rigoroso.
- O `lr_control` é o pior braço de todos, pior que o vanilla: reduzir learning
  rate uniformemente atrapalha. Isso o valida como falsificador não-trivial.
  **Correção de 29/09/2026:** não replicou. Nas 10 seeds confirmatórias o sinal
  inverte e nada é significativo (FAA `lr_control − vanilla` = +0,0084, p = 0,754;
  forgetting −0,0096, p = 0,344). Leitura correta: **nula** — o controle empata
  com o vanilla, que é a posição neutra esperada de um falsificador.

## Referências

- [Índice da documentação](../README.md)
- [SlowHeat em Transformers, seção 12 (mascaramento LoRA)](../mechanisms/functional_slowheat_transformers.md)
- [Semântica do otimizador](../mechanisms/optimizer_semantics.md)
- [LoRA (Hu et al., ICLR 2022)](https://arxiv.org/abs/2106.09685)
