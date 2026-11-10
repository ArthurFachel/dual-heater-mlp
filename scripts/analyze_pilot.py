#!/usr/bin/env python3
"""Descriptive analysis for goals/piloto_tempo_da_mascara.md (EXPLORATORY pilot, seeds 500-509).
Reading declared before the run: endpoint FAA; retention T1 secondary; guard = acquisition of the last
task may drop at most 2 pp. Candidate for confirmatory if the FAA contrast is >= +2 pp in >= 8/10 seeds.
Needs scipy (/home/fachel/novoenv/bin/python)."""
import glob, json, math, statistics as st, sys
from scipy import stats

pat = sys.argv[1] if len(sys.argv) > 1 else "results/pilot_mask_timing/seed*/manifest.json"
E, B = {}, {}
for f in sorted(glob.glob(pat)):
    m = json.load(open(f))
    for arm in m["families"]["0.75"]["arms"]:
        if not arm["discarded"]:
            E[(arm["arm"]["name"], m["seed"])] = arm["endpoints"]
            B[(arm["arm"]["name"], m["seed"])] = arm["boundaries"]
arms = sorted({a for a, _ in E})
seeds = sorted({s for _, s in E if all((a, s) in E for a in arms)})
print(f"manifests: {pat}\nbraços: {arms}\nseeds completas: {len(seeds)} {seeds}")


def ci(d):
    n = len(d); m = st.mean(d); h = stats.t.ppf(0.975, n - 1) * st.stdev(d) / math.sqrt(n)
    return m, m - h, m + h, stats.ttest_1samp(d, 0).pvalue if st.stdev(d) > 0 else float("nan")


def line(label, d):
    m, lo, hi, p = ci(d)
    print(f"  {label:58s} {100*m:+6.2f} pp [{100*lo:+6.2f}; {100*hi:+6.2f}] a favor {sum(x > 0 for x in d)}/{len(d)} p {p:.3f}")
    return m, sum(x > 0 for x in d)


def diff(a, b, k):
    return [E[(a, s)][k] - E[(b, s)][k] for s in seeds]


# integrity of the freeze arms: frozen boundaries keep the digest of the last allowed boundary
for a in arms:
    if "freeze" in a:
        bad = []
        for s in seeds:
            bs = B[(a, s)]
            last = None
            for b in bs:
                if b["mask_frozen"] is False:
                    last = b["protected_digest"]
                elif b["protected_digest"] != last:
                    bad.append(s); break
        print(f"integridade {a}: " + ("OK" if not bad else f"FALHOU {bad}"))

print("\nMédias (n = %d):" % len(seeds))
for a in arms:
    ss = [E[(a, s)] for s in seeds]
    print(f"  {a:24s} FAA {st.mean(x['final_average_accuracy'] for x in ss):.4f}  ret T1 "
          f"{st.mean(x['retention_first_task'] for x in ss):.4f}  aq {st.mean(x['acquisition_last_task'] for x in ss):.4f}  "
          f"esq {st.mean(x['mean_forgetting'] for x in ss):.4f}")

H, R = "hard_b0.75", "reduced_lr"
print("\nFreeze (endpoint FAA; candidato se >= +2 pp e >= 8/10, guarda aquisição >= -2 pp):")
for a in [x for x in arms if "freeze" in x]:
    m, w = line(f"{a} - hard  FAA", diff(a, H, "final_average_accuracy"))
    line(f"{a} - hard  retenção T1 (secundário)", diff(a, H, "retention_first_task"))
    g, _ = line(f"{a} - hard  aquisição (guarda)", diff(a, H, "acquisition_last_task"))
    print(f"    -> {'CANDIDATO' if (m >= 0.02 and w >= 8 and g >= -0.02) else 'não candidato'}")

W = [x for x in arms if x.startswith("hard_warmup")]
RW = [x for x in arms if x.startswith("reduced_lr_warmup")]
if W and RW:
    w, rw = W[0], RW[0]
    print("\nWarmup (candidato se a diferença-em-diferenças de FAA >= +2 pp e >= 8/10, guarda ok):")
    line(f"{w} - hard  FAA", diff(w, H, "final_average_accuracy"))
    line(f"{rw} - reduced_lr  FAA", diff(rw, R, "final_average_accuracy"))
    did = [(E[(w, s)]["final_average_accuracy"] - E[(rw, s)]["final_average_accuracy"])
           - (E[(H, s)]["final_average_accuracy"] - E[(R, s)]["final_average_accuracy"]) for s in seeds]
    m, k = line("DiD FAA (warmup ajuda mais a máscara que o controle)", did)
    line(f"{w} - hard  retenção T1", diff(w, H, "retention_first_task"))
    g, _ = line(f"{w} - hard  aquisição (guarda)", diff(w, H, "acquisition_last_task"))
    print(f"    -> {'CANDIDATO' if (m >= 0.02 and k >= 8 and g >= -0.02) else 'não candidato'}")

print("\nReferência:")
line("hard - reduced_lr  FAA", diff(H, R, "final_average_accuracy"))
line("hard - reduced_lr  retenção T1", diff(H, R, "retention_first_task"))
