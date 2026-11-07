#!/usr/bin/env python3
"""Pre-declared analysis for docs/protocols/confirmatory_bert_soft30.md (F12).

Usage: analyze_confirm_bert_soft30.py [ROOT]            -> ROOT/<T1>__<T2>/budget_0.5
       analyze_confirm_bert_soft30.py --exploratory     -> F5 + soft_2x2 data (validation only)
Needs scipy (/home/fachel/novoenv/bin/python).
"""
import json, math, statistics as st, sys
from scipy import stats

PAIRS = [("banking", "credit_cards"), ("auto_and_commute", "travel"), ("kitchen_and_dining", "home"),
         ("small_talk", "meta"), ("utility", "work")]
SH, UN, VA = "slowheat_magnitude_soft_b30", "slowheat_uniform_soft_b30", "vanilla"
PRIMARY = "task1_retention"
SECONDARY = ("final_average_accuracy", "task2_acquisition")

if "--exploratory" in sys.argv:
    path = lambda a, b: ("results/soft_2x2/budget_0.5" if (a, b) == PAIRS[0]
                         else f"results/domain_pairs/{a}__{b}/budget_0.5")
else:
    root = next((x for x in sys.argv[1:] if not x.startswith("--")), "results/confirm_bert_soft30")
    path = lambda a, b: f"{root}/{a}__{b}/budget_0.5"


def sign_p(d):
    pos, n = sum(x > 0 for x in d), sum(x != 0 for x in d)
    return stats.binomtest(pos, n).pvalue if n else 1.0


def ci(d):
    m = st.mean(d); h = stats.t.ppf(0.975, len(d) - 1) * st.stdev(d) / math.sqrt(len(d))
    return m, m - h, m + h


data, res = {}, []
for a, b in PAIRS:
    raw = json.load(open(f"{path(a, b)}/diagnostic_summary.json"))["raw_by_seed"]
    seeds = sorted(raw); data[(a, b)] = (raw, seeds)
    x = [raw[s][SH][PRIMARY] for s in seeds]; y = [raw[s][UN][PRIMARY] for s in seeds]
    d = [p - q for p, q in zip(x, y)]
    m, lo, hi = ci(d)
    res.append([f"{a} -> {b}", len(seeds), m, lo, hi, stats.ttest_rel(x, y).pvalue, sum(v > 0 for v in d), d])
order = sorted(range(len(res)), key=lambda i: res[i][5])
adj, run = [0.0] * len(res), 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(res) - rank) * res[i][5])); adj[i] = run

print(f"Endpoint primário: {PRIMARY}, SlowHeat soft b30 - uniforme soft b30 (t pareado, Holm sobre {len(res)} pares)")
nconf = 0
for (name, n, m, lo, hi, p, pos, d), pa in zip(res, adj):
    verdict = "CONFIRMADO" if (m > 0 and pa < 0.05) else ("REFUTADO" if lo <= 0 <= hi or m <= 0 else "inconclusivo")
    nconf += verdict == "CONFIRMADO"
    print(f"{name:32s} n {n:2d}  {100*m:+.2f} pp IC95 [{100*lo:+.2f}; {100*hi:+.2f}] p {p:.2e} Holm {pa:.2e} "
          f"a favor {pos}/{n} p sinal {sign_p(d):.4f} -> {verdict}")
print(f"\nPares confirmados: {nconf}/{len(res)}. Leitura geral (>= 4 de 5): {'CONFIRMA' if nconf >= 4 else 'NÃO CONFIRMA'}")

print("\nSecundários (descritivos):")
for (a, b), (raw, seeds) in data.items():
    mean = lambda arm, k: st.mean(raw[s][arm][k] for s in seeds)
    line = f"{a + ' -> ' + b:32s} ret T1 VA {mean(VA, PRIMARY):.4f} SH {mean(SH, PRIMARY):.4f} UN {mean(UN, PRIMARY):.4f}"
    for k in SECONDARY:
        m, lo, hi = ci([raw[s][SH][k] - raw[s][UN][k] for s in seeds])
        pos = sum(raw[s][SH][k] > raw[s][UN][k] for s in seeds)
        line += f" | {k.split('_')[0][:4]} SH-UN {100*m:+.2f} [{100*lo:+.2f}; {100*hi:+.2f}] {pos}/{len(seeds)}"
    print(line)
    print(f"{'':32s} FAA VA {mean(VA, 'final_average_accuracy'):.4f} SH {mean(SH, 'final_average_accuracy'):.4f} "
          f"UN {mean(UN, 'final_average_accuracy'):.4f} | SH-VA ret {100*(mean(SH, PRIMARY)-mean(VA, PRIMARY)):+.2f} pp, "
          f"UN-VA ret {100*(mean(UN, PRIMARY)-mean(VA, PRIMARY)):+.2f} pp")
