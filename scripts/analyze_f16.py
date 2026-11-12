#!/usr/bin/env python3
"""Pre-declared analysis for docs/protocols/confirmatory_qwen_hard_random_fixed.md (F16).
Usage: analyze_f16.py [GLOB]   default results/f16_confirm_qwen_hard_random_fixed/seed*/manifest.json
Validation (secondary only): analyze_f16.py "results/f15_confirm_qwen_hard_permuted/seed*/manifest.json"
Needs scipy (/home/fachel/novoenv/bin/python)."""
import glob, json, math, statistics as st, sys
from scipy import stats

H, F, P = "hard_b0.75", "hard_random_fixed_b0.75", "hard_permuted_b0.75"
pat = sys.argv[1] if len(sys.argv) > 1 else "results/f16_confirm_qwen_hard_random_fixed/seed*/manifest.json"
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


seeds = paired(H, F)
print(f"seeds completas (hard e fixo): {len(seeds)} {seeds}")
if len(seeds) >= 1:
    bad = []
    for s in seeds:
        bh, bf = B[(H, s)], B[(F, s)]
        counts = [x["protected_units"] for x in bh]
        ok = len(bh) == len(bf) and counts == [x["protected_units"] for x in bf]
        ok = ok and all(x["mask"] == "hard" for x in bh + bf)
        ok = ok and all(x.get("random_fixed") is True for x in bf)
        for i in range(1, len(bf)):
            same_count = bf[i]["protected_units"] == bf[i - 1]["protected_units"]
            if same_count and bf[i]["protected_digest"] != bf[i - 1]["protected_digest"]:
                ok = False
        ok = ok and all(x["protected_digest"] != y["protected_digest"] for x, y in zip(bh, bf))
        if (P, s) in B:
            bp = B[(P, s)]
            ok = ok and [x["protected_units"] for x in bp] == counts
            ok = ok and all(x.get("permutation_reordered") for x in bp)
        if not ok:
            bad.append(s)
    print("integridade (máscara hard, contagem igual, fixo constante e != hard, permutado ressorteado): "
          + ("OK" if not bad else f"FALHOU nas seeds {bad} -> RUN INVÁLIDA"))
if len(seeds) >= 2:
    d, m, lo, hi, p = contrast(H, F, "retention_first_task", seeds)
    pos = sum(x > 0 for x in d); nz = sum(x != 0 for x in d)
    sp = stats.binomtest(pos, nz).pvalue if nz else 1.0
    verdict = "CONFIRMADO" if (m > 0 and p < 0.05) else ("REFUTADO" if (lo <= 0 <= hi or m <= 0) else "inconclusivo")
    print(f"\nPrimário retenção t0 {H} - {F}: {100*m:+.2f} pp IC95 [{100*lo:+.2f}; {100*hi:+.2f}] p {p:.2e} "
          f"a favor {pos}/{len(seeds)} p sinal {sp:.4f} -> {verdict}")
print("\nSecundários (descritivos):")
for a, b, k in [(H, F, "final_average_accuracy"), (H, F, "acquisition_last_task"),
                (F, P, "retention_first_task"), (H, P, "retention_first_task")]:
    ss = paired(a, b)
    if len(ss) >= 2:
        dd, mm, l, h, pp = contrast(a, b, k, ss)
        print(f"  {a} - {b} {k}: {100*mm:+.2f} pp [{100*l:+.2f}; {100*h:+.2f}] {sum(x > 0 for x in dd)}/{len(ss)} p {pp:.3f}")
for a in (H, F, P):
    ss = sorted(s for (x, s) in E if x == a)
    if ss:
        print(f"  média {a} (n {len(ss)}): ret {st.mean(E[(a, s)]['retention_first_task'] for s in ss):.4f} "
              f"FAA {st.mean(E[(a, s)]['final_average_accuracy'] for s in ss):.4f} "
              f"aq {st.mean(E[(a, s)]['acquisition_last_task'] for s in ss):.4f}")
