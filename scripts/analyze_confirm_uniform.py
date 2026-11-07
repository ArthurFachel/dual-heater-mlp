#!/usr/bin/env python3
"""Pre-declared analysis for docs/protocols/confirmatory_uniform_control.md."""
import csv, glob, math, statistics as st, sys
from scipy import stats

R = sys.argv[1] if len(sys.argv) > 1 else "results/confirm_uniform_control"
FAM = [
    ("replay", "slowheat_replay_hidden_beta_30_budget_0.25", "uniform_replay_hidden_beta_30_budget_0.25"),
    ("derpp", "slowheat_derpp_hidden_beta_30_budget_0.25", "uniform_derpp_hidden_beta_30_budget_0.25"),
    ("replay_calibrated", "slowheat_replay_hidden_beta_30_budget_0.25_calibrated",
     "uniform_replay_hidden_beta_30_budget_0.25_calibrated"),
    ("er_ace", "slowheat_er_ace_hidden_beta_30_budget_0.25", "uniform_er_ace_hidden_beta_30_budget_0.25"),
]
rows = {}
for f in sorted(glob.glob(f"{R}/seed_*/summary.csv")):
    seed = int(f.split("seed_")[1].split("/")[0])
    for r in csv.DictReader(open(f)):
        rows[(seed, r["method"])] = r
seeds = sorted({s for s, _ in rows})
print(f"n seeds = {len(seeds)}")
v = lambda s, m, k="final_average_accuracy": float(rows[(s, m)][k])
res = []
for B, SH, UN in FAM:
    d = [v(s, SH) - v(s, UN) for s in seeds]
    t = stats.ttest_rel([v(s, SH) for s in seeds], [v(s, UN) for s in seeds])
    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    h = stats.t.ppf(0.975, len(d) - 1) * se
    res.append([B, m, m - h, m + h, t.pvalue, sum(x > 0 for x in d), len(d), d])
order = sorted(range(len(res)), key=lambda i: res[i][4])
adj, run = [0.0] * len(res), 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(res) - rank) * res[i][4])); adj[i] = run
for (B, m, lo, hi, p, pos, n, d), pa in zip(res, adj):
    verdict = "CONFIRMADO" if (m > 0 and pa < 0.05) else ("REFUTADO" if lo <= 0 <= hi or m <= 0 else "inconclusivo")
    sp = stats.binomtest(pos, sum(x != 0 for x in d)).pvalue if any(d) else 1.0
    print(f"{B:18s} SH-uniforme {100*m:+.2f} pp IC95 [{100*lo:+.2f}; {100*hi:+.2f}] p {p:.2e} Holm {pa:.2e} "
          f"a favor {pos}/{n} p sinal {sp:.4f} -> {verdict}")
print("\nSecundários (descritivos):")
for B, SH, UN in FAM:
    f = lambda m, k: st.mean(v(s, m, k) for s in seeds)
    print(f"{B:18s} FAA B {f(B,'final_average_accuracy'):.4f} SH {f(SH,'final_average_accuracy'):.4f} "
          f"UN {f(UN,'final_average_accuracy'):.4f} | esquec. B {f(B,'average_forgetting'):.4f} "
          f"SH {f(SH,'average_forgetting'):.4f} UN {f(UN,'average_forgetting'):.4f}")
