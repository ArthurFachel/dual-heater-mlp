#!/usr/bin/env python3
"""Pre-declared analysis for docs/protocols/confirmatory_random_fixed_mlp.md (T3c).

Written and validated BEFORE the confirmatory run. Usage:
  analyze_t3c.py <split_dir> <permuted_dir>
Reads seed_*/summary.csv (endpoints) and seed_*/results.json (integrity).
"""
import csv
import glob
import json
import math
import statistics as st
import sys

from scipy import stats

SPLIT = sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else "results/t3c_random_fixed_mlp/split"
PERM = sys.argv[2] if len(sys.argv) > 2 and __name__ == "__main__" else "results/t3c_random_fixed_mlp/permuted"
TAILS = {
    "replay": "replay_hidden_beta_30_budget_0.25",
    "derpp": "derpp_hidden_beta_30_budget_0.25",
    "replay_calibrated": "replay_hidden_beta_30_budget_0.25_calibrated",
    "er_ace": "er_ace_hidden_beta_30_budget_0.25",
}
SPLIT_FAM = [(f"MLP-S {b}", b, f"slowheat_{t}", f"uniform_{t}", f"random_fixed_{t}")
             for b, t in TAILS.items()]
PERM_FAM = [("MLP-P hard", "vanilla", "hard_freeze", "uniform_hard_freeze",
             "random_fixed_hard_freeze")]
FAA = "final_average_accuracy"


def load(root):
    rows, raw = {}, {}
    for f in sorted(glob.glob(f"{root}/seed_*/summary.csv")):
        seed = int(f.split("seed_")[1].split("/")[0])
        for r in csv.DictReader(open(f)):
            rows[(seed, r["method"])] = r
        j = f.replace("summary.csv", "results.json")
        try:
            raw[seed] = json.load(open(j))
        except FileNotFoundError:
            raw[seed] = None
    return rows, raw


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    m = st.mean(d)
    se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    h = stats.t.ppf(0.975, len(d) - 1) * se
    p = stats.ttest_rel(a, b).pvalue if len(d) > 1 else float("nan")
    return m, m - h, m + h, p, sum(x > 0 for x in d), len(d), d


def integrity(raw, seeds, method_arm, random_arm):
    """G1 fixed order, G2 protected fraction matches the method arm, G3 not the selection."""
    problems = []
    for s in seeds:
        r = raw.get(s)
        if r is None or random_arm not in r:
            problems.append(f"seed {s}: results.json missing {random_arm}")
            continue
        hist = r[random_arm].get("random_fixed_history", [])
        cap_rf = r[random_arm].get("capacity_history", [])
        cap_m = r[method_arm].get("capacity_history", [])
        if not hist or len(hist) != len(cap_rf):
            problems.append(f"seed {s}: {len(hist)} reassignments vs {len(cap_rf)} consolidations")
            continue
        for k in range(1, len(hist)):
            for layer, (c0, c1) in enumerate(zip(hist[k - 1]["protected_counts"],
                                                 hist[k]["protected_counts"])):
                if c0 == c1 and hist[k - 1]["digests"][layer] != hist[k]["digests"][layer]:
                    problems.append(f"seed {s}: layer {layer} order changed at boundary {k}")
        for k, (a, b) in enumerate(zip(cap_rf, cap_m)):
            fa = st.mean(x["protected_fraction"] for x in a)
            fb = st.mean(x["protected_fraction"] for x in b)
            if abs(fa - fb) > 0.01:
                problems.append(f"seed {s}: protected fraction {fa:.4f} vs {fb:.4f} at boundary {k}")
        if any(h["overlap_with_own_selection"] >= 0.999 for h in hist):
            problems.append(f"seed {s}: random set equals own selection")
    return problems


def overlap_mean(raw, seeds, random_arm):
    vals = [h["overlap_with_own_selection"] for s in seeds if raw.get(s)
            for h in raw[s][random_arm].get("random_fixed_history", [])]
    return st.mean(vals) if vals else float("nan")


def main():
    results, secondary, problems_all = [], [], []
    for root, fam in ((SPLIT, SPLIT_FAM), (PERM, PERM_FAM)):
        rows, raw = load(root)
        seeds = sorted({s for s, _ in rows})
        print(f"{root}: n seeds = {len(seeds)} {seeds}")
        v = lambda s, m, k=FAA: float(rows[(s, m)][k])
        for label, B, M, U, R in fam:
            have = [s for s in seeds if all((s, a) in rows for a in (B, M, U, R))]
            if len(have) != len(seeds):
                problems_all.append(f"{label}: arms missing in {set(seeds) - set(have)}")
            prob = integrity(raw, have, M, R) if raw else ["no results.json"]
            problems_all += [f"{label}: {x}" for x in prob]
            res = paired([v(s, M) for s in have], [v(s, R) for s in have])
            results.append((label, *res))
            for name, a1, a2, k in (("shape R - U", R, U, FAA), ("plasticity U - B", U, B, FAA),
                                    ("M - U (replication)", M, U, FAA), ("M - B", M, B, FAA),
                                    ("forgetting M - R", M, R, "average_forgetting")):
                secondary.append((label, name, *paired([v(s, a1, k) for s in have],
                                                       [v(s, a2, k) for s in have])[:6]))
            secondary.append((label, "mean overlap of R with own selection",
                              overlap_mean(raw, have, R), None, None, None, None, None))

    print("\nINTEGRITY (checked before reading the primary)")
    if problems_all:
        for x in problems_all:
            print("  FAIL", x)
        print("VERDICT: INVALID (protocol section Integrity)")
    else:
        print("  OK: fixed order at every boundary, protected fraction matched (<= 0.01), "
              "random set != selection, all arms present")

    order = sorted(range(len(results)), key=lambda i: results[i][4])
    adj, run = [0.0] * len(results), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(results) - rank) * results[i][4]))
        adj[i] = run
    print("\nPRIMARY: selection = FAA(M) - FAA(R_fixed), paired t two-sided, Holm over 5, alpha 0.05")
    for (label, m, lo, hi, p, pos, n, d), pa in zip(results, adj):
        if m > 0 and pa < 0.05:
            verdict = "CONFIRMED"
        elif lo <= 0 <= hi or m <= 0:
            verdict = "REFUTED"
        else:
            verdict = "inconclusive"
        bounded = " (null bounded within +-0.5 pp)" if verdict == "REFUTED" and -0.005 < lo and hi < 0.005 else ""
        nz = sum(x != 0 for x in d)
        sp = stats.binomtest(pos, nz).pvalue if nz else 1.0
        print(f"  {label:22s} {100*m:+.2f} pp IC95 [{100*lo:+.2f}; {100*hi:+.2f}] p {p:.2e} "
              f"Holm {pa:.2e} {pos}/{n} sign p {sp:.4f} -> {verdict}{bounded}")

    print("\nSECONDARY (descriptive, no correction)")
    for label, name, m, lo, hi, p, pos, n in secondary:
        if lo is None:
            print(f"  {label:22s} {name}: {m:.3f}")
        else:
            print(f"  {label:22s} {name}: {100*m:+.2f} pp [{100*lo:+.2f}; {100*hi:+.2f}] {pos}/{n} p {p:.2e}")


if __name__ == "__main__":
    main()
