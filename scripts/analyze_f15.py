#!/usr/bin/env python3
"""Pre-declared analysis for docs/protocols/confirmatory_qwen_hard_permuted.md (F15).
Usage: analyze_f15.py [GLOB]   default results/f15_confirm_qwen_hard_permuted/seed*/manifest.json
Validation (secondary only, no hard_permuted there): analyze_f15.py "results/f14_confirm_qwen_hard/seed*/manifest.json"
Needs scipy (/home/fachel/novoenv/bin/python)."""
import glob, json, math, statistics as st, sys
from scipy import stats

H, HP, R = "hard_b0.75", "hard_permuted_b0.75", "reduced_lr"
pat = sys.argv[1] if len(sys.argv) > 1 else "results/f15_confirm_qwen_hard_permuted/seed*/manifest.json"
E, B = {}, {}
for f in sorted(glob.glob(pat)):
    m = json.load(open(f))
    for arm in m["families"]["0.75"]["arms"]:
        if not arm["discarded"]:
            E[(arm["arm"]["name"], m["seed"])] = arm["endpoints"]
            B[(arm["arm"]["name"], m["seed"])] = arm["boundaries"]
print(f"manifests: {pat}")


def paired(a, b):
    return sorted({s for (x, s) in E if x == a} & {s for (x, s) in E if x == b})


def contrast(a, b, k, seeds):
    d = [E[(a, s)][k] - E[(b, s)][k] for s in seeds]
    n = len(d); m = st.mean(d); h = stats.t.ppf(0.975, n - 1) * st.stdev(d) / math.sqrt(n)
    return d, m, m - h, m + h, stats.ttest_1samp(d, 0).pvalue


seeds = paired(H, HP)
print(f"seeds completas (hard e hard_permuted): {len(seeds)} {seeds}")
if len(seeds) >= 2:
    bad = []
    for s in seeds:
        bh, bp = B[(H, s)], B[(HP, s)]
        if len(bh) != len(bp) or any(not x.get("permutation_reordered") for x in bp) or \
           [x["protected_units"] for x in bh] != [x["protected_units"] for x in bp] or \
           any(x["mask"] != "hard" for x in bp):
            bad.append(s)
    print("integridade (permutação real, mesma contagem protegida, máscara hard): " + ("OK" if not bad else f"FALHOU nas seeds {bad} -> RUN INVÁLIDA"))
    d, m, lo, hi, p = contrast(H, HP, "retention_first_task", seeds)
    pos = sum(x > 0 for x in d); nz = sum(x != 0 for x in d)
    sp = stats.binomtest(pos, nz).pvalue if nz else 1.0
    verdict = "CONFIRMADO" if (m > 0 and p < 0.05) else ("REFUTADO" if (lo <= 0 <= hi or m <= 0) else "inconclusivo")
    print(f"\nPrimário retenção t0 {H} - {HP}: {100*m:+.2f} pp IC95 [{100*lo:+.2f}; {100*hi:+.2f}] p {p:.2e} "
          f"a favor {pos}/{len(seeds)} p sinal {sp:.4f} -> {verdict}")
print("\nSecundários (descritivos):")
for a, b, k in [(H, HP, "final_average_accuracy"), (H, HP, "acquisition_last_task"),
                (H, R, "retention_first_task"), (HP, R, "retention_first_task"), (H, R, "final_average_accuracy")]:
    ss = paired(a, b)
    if len(ss) >= 2:
        dd, mm, l, h, pp = contrast(a, b, k, ss)
        print(f"  {a} - {b} {k}: {100*mm:+.2f} pp [{100*l:+.2f}; {100*h:+.2f}] {sum(x > 0 for x in dd)}/{len(ss)} p {pp:.3f}")
for a in (H, HP, R):
    ss = sorted(s for (x, s) in E if x == a)
    if ss:
        print(f"  média {a} (n {len(ss)}): ret {st.mean(E[(a, s)]['retention_first_task'] for s in ss):.4f} "
              f"FAA {st.mean(E[(a, s)]['final_average_accuracy'] for s in ss):.4f} "
              f"aq {st.mean(E[(a, s)]['acquisition_last_task'] for s in ss):.4f}")
