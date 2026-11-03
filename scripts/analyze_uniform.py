import json, math, statistics as st, sys

d = json.load(open(sys.argv[1]))
raw = d["raw_by_seed"]
seeds = sorted(raw, key=int)
arms = ["vanilla", "slowheat_random_hard", "slowheat_uniform_hard", "slowheat_magnitude_hard"]
NAME = {"vanilla": "vanilla (sem protecao)", "slowheat_random_hard": "mascara aleatoria",
        "slowheat_uniform_hard": "lr uniforme (mesma plasticidade)", "slowheat_magnitude_hard": "SlowHeat |z|"}
keys = ["task1_acquisition", "task1_retention", "task2_acquisition", "final_average_accuracy"]

def val(seed, arm, k):
    return float(raw[seed][arm][k])

def sign_p(diffs):
    pos = sum(x > 0 for x in diffs); neg = sum(x < 0 for x in diffs); n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)

print(f"n = {len(seeds)} seeds")
print(f"{'braco':36s}" + "".join(f"{k:>24s}" for k in keys))
for a in arms:
    row = f"{NAME[a]:36s}"
    for k in keys:
        v = [val(s, a, k) for s in seeds]
        row += f"{st.mean(v):>16.4f} +-{st.stdev(v):.4f}"
    print(row)
print("\ncontrastes pareados (pontos percentuais):")
for a, b in [("slowheat_magnitude_hard", "slowheat_uniform_hard"), ("slowheat_uniform_hard", "slowheat_random_hard"),
             ("slowheat_magnitude_hard", "slowheat_random_hard"), ("slowheat_uniform_hard", "vanilla")]:
    for k in ("task1_retention", "final_average_accuracy", "task2_acquisition"):
        dd = [100 * (val(s, a, k) - val(s, b, k)) for s in seeds]
        print(f"  {NAME[a]} - {NAME[b]:34s} {k:24s} {st.mean(dd):+7.2f} pp  {sum(x>0 for x in dd)}+/{sum(x<0 for x in dd)}-  p_sinal={sign_p(dd):.4f}")
