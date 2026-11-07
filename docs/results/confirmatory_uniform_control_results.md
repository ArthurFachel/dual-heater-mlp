# Confirmatório: SlowHeat vs lr uniforme pareado, Split-MNIST, 4 baselines (F11)

Protocolo: docs/protocols/confirmatory_uniform_control.md, congelado no commit e7a2ece antes da run.
Run: 07/11/2026 14:45-15:02, CPU do Lab (motoz), 8 threads, nice 10, 20/20 seeds, nenhuma descartada
nem rerodada. Runner scripts/run_confirm_uniform.py (lançado por scripts/launch_confirm.sh); análise
scripts/analyze_confirm_uniform.py com /home/fachel/novoenv/bin/python (scipy).
Artefatos brutos (gitignored): results/confirm_uniform_control/ (aggregate.csv, paired_differences.csv,
seed_*/summary.csv, analysis_preregistered.txt).

## Saída da análise pré-registrada (literal)

```
n seeds = 20
replay             SH-uniforme +0.62 pp IC95 [+0.25; +0.98] p 2.14e-03 Holm 4.27e-03 a favor 17/20 p sinal 0.0026 -> CONFIRMADO
derpp              SH-uniforme +1.91 pp IC95 [+1.39; +2.43] p 3.12e-07 Holm 9.35e-07 a favor 19/20 p sinal 0.0000 -> CONFIRMADO
replay_calibrated  SH-uniforme +1.49 pp IC95 [+1.21; +1.78] p 9.59e-10 Holm 3.84e-09 a favor 20/20 p sinal 0.0000 -> CONFIRMADO
er_ace             SH-uniforme +0.39 pp IC95 [+0.15; +0.64] p 3.33e-03 Holm 4.27e-03 a favor 15/20 p sinal 0.0414 -> CONFIRMADO

Secundários (descritivos):
replay             FAA B 0.7586 SH 0.7706 UN 0.7644 | esquec. B 0.2880 SH 0.2694 UN 0.2797
derpp              FAA B 0.8244 SH 0.8586 UN 0.8395 | esquec. B 0.2041 SH 0.1483 UN 0.1805
replay_calibrated  FAA B 0.8224 SH 0.8479 UN 0.8329 | esquec. B 0.2020 SH 0.1599 UN 0.1857
er_ace             FAA B 0.7109 SH 0.7083 UN 0.7044 | esquec. B 0.0435 SH 0.0487 UN 0.0463
```

## Leitura

Pela regra congelada, os 4 contrastes são CONFIRMADOS: a seleção por importância rende mais que a
mesma plasticidade média sem seleção em replay, DER++, replay calibrado e ER-ACE.

Ressalvas (descritivas, fora da família de teste):
- ER-ACE: SlowHeat+ER-ACE (0,7083) fica abaixo do ER-ACE puro (0,7109) e o esquecimento sobe
  (0,0487 vs 0,0435). Os dois braços de plasticidade reduzida pioram o ER-ACE; a seleção só perde menos.
- Efeitos pequenos (+0,4 a +1,9 pp), um benchmark (Split-MNIST, MLP), beta 30 e budget 0,25 herdados,
  não tunados por baseline.
- Custo de parede: ~5,0 s vs ~3,2 s por seed (SlowHeat vs baseline puro, CPU).
