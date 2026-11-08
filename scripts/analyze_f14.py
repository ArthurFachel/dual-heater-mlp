#!/usr/bin/env python3
"""Pre-declared analysis for docs/protocols/confirmatory_qwen_hard.md (F14).
Usage: analyze_f14.py [GLOB]   default results/f14_confirm_qwen_hard/seed*/manifest.json
Validation: analyze_f14.py "/mnt/B-SSD/fachel/dual-heater-mlp/results/qwen_iso_plasticity/confirm120_seed*/manifest.json"
Needs scipy (/home/fachel/novoenv/bin/python)."""
import glob, json, math, statistics as st, sys
from scipy import stats

pat = sys.argv[1] if len(sys.argv) > 1 else "results/f14_confirm_qwen_hard/seed*/manifest.json"
E = {}
for f in sorted(glob.glob(pat)):
    m = json.load(open(f))
    for arm in m["families"]["0.75"]["arms"]:
        if not arm["discarded"]:
            E[(arm["arm"]["name"], m["seed"])] = arm["endpoints"]
seeds = sorted({s for (a, s) in E if a == "hard_b0.75"} & {s for (a, s) in E if a == "reduced_lr"})
n = len(seeds)
print(f"manifests: {pat}\nseeds completas: {n} {seeds}")
if n < 2: sys.exit(0)


def contrast(a, b, k):
    d = [E[(a, s)][k] - E[(b, s)][k] for s in seeds]
    m = st.mean(d); h = stats.t.ppf(0.975, n - 1) * st.stdev(d) / math.sqrt(n)
    return d, m, m - h, m + h, stats.ttest_1samp(d, 0).pvalue


d, m, lo, hi, p = contrast("hard_b0.75", "reduced_lr", "retention_first_task")
pos = sum(x > 0 for x in d); sp = stats.binomtest(pos, sum(x != 0 for x in d)).pvalue if any(d) else 1.0
verdict = "CONFIRMADO" if (m > 0 and p < 0.05) else ("REFUTADO" if (lo <= 0 <= hi or m <= 0) else "inconclusivo")
print(f"\nPrimário retenção t0 hard_b0.75 - reduced_lr: {100*m:+.2f} pp IC95 [{100*lo:+.2f}; {100*hi:+.2f}] p {p:.2e} "
      f"a favor {pos}/{n} p sinal {sp:.4f} -> {verdict}")
print("\nSecundários (descritivos):")
for a, b, k in [("hard_b0.75", "reduced_lr", "final_average_accuracy"), ("hard_b0.75", "reduced_lr", "acquisition_last_task"),
                ("hard_b0.75", "vanilla", "retention_first_task"), ("reduced_lr", "vanilla", "retention_first_task")]:
    if all((x, s) in E for x in (a, b) for s in seeds):
        dd, mm, l, h, pp = contrast(a, b, k)
        print(f"  {a} - {b} {k}: {100*mm:+.2f} pp [{100*l:+.2f}; {100*h:+.2f}] {sum(x > 0 for x in dd)}/{n} p {pp:.3f}")
for a in ("hard_b0.75", "reduced_lr", "vanilla"):
    if all((a, s) in E for s in seeds):
        print(f"  média {a}: ret {st.mean(E[(a, s)]['retention_first_task'] for s in seeds):.4f} "
              f"FAA {st.mean(E[(a, s)]['final_average_accuracy'] for s in seeds):.4f}")
