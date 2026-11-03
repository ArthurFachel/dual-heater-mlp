import json, math, statistics as st, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/budget_curve")
ARMS = {"slowheat_random_hard": "aleatoria", "slowheat_uniform_hard": "lr uniforme", "slowheat_magnitude_hard": "SlowHeat |z|"}
rows = {}  # (budget, arm, seed) -> endpoints
vanilla = {}
for f in sorted(root.glob("budget_*/diagnostic_summary.json")):
    b = float(f.parent.name.split("_")[1])
    raw = json.load(open(f))["raw_by_seed"]
    for seed, arms in raw.items():
        for arm, v in arms.items():
            if arm == "vanilla":
                vanilla.setdefault(seed, []).append(v["task1_retention"])
            else:
                rows[(b, arm, seed)] = v

def sign_p(d):
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d); n = pos + neg
    if not n: return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

budgets = sorted({k[0] for k in rows})
seeds = sorted({k[2] for k in rows}, key=int)
same = all(len(set(v)) == 1 for v in vanilla.values())
print(f"seeds={len(seeds)} budgets={budgets}  vanilla identico entre orcamentos: {same}")
print(f"\n{'orc.':>5s} {'braco':14s} {'ret T1':>8s} {'aquis T2':>9s} {'FAA':>7s}")
for b in budgets:
    for arm, name in ARMS.items():
        rs = [rows[(b, arm, s)] for s in seeds if (b, arm, s) in rows]
        if rs:
            print(f"{b:>5g} {name:14s} {st.mean(r['task1_retention'] for r in rs):8.4f} {st.mean(r['task2_acquisition'] for r in rs):9.4f} {st.mean(r['final_average_accuracy'] for r in rs):7.4f}  n={len(rs)}")
print("\nSlowHeat - lr uniforme, por orcamento (pp; sinais; p sinal)")
for b in budgets:
    out = []
    for k in ("task1_retention", "task2_acquisition", "final_average_accuracy"):
        d = [100 * (rows[(b, "slowheat_magnitude_hard", s)][k] - rows[(b, "slowheat_uniform_hard", s)][k])
             for s in seeds if (b, "slowheat_magnitude_hard", s) in rows and (b, "slowheat_uniform_hard", s) in rows]
        if d:
            out.append(f"{k[:9]} {st.mean(d):+6.2f} ({sum(x>0 for x in d)}+/{sum(x<0 for x in d)}-, p={sign_p(d):.3f})")
    print(f"  {b:>5g}  " + "   ".join(out))

# Pareto: for each SlowHeat point, interpolate the uniform curve's retention at the same T2 acquisition
print("\nRetencao do SlowHeat contra a curva do lr uniforme na MESMA aquisicao da T2 (interpolacao linear):")
def curve(arm):
    pts = sorted((st.mean(rows[(b, arm, s)]["task2_acquisition"] for s in seeds), st.mean(rows[(b, arm, s)]["task1_retention"] for s in seeds), b) for b in budgets)
    return pts
U = curve("slowheat_uniform_hard")
for acq, ret, b in curve("slowheat_magnitude_hard"):
    lo = [p for p in U if p[0] <= acq]; hi = [p for p in U if p[0] >= acq]
    if not lo or not hi:
        print(f"  orc {b:g}: aquisicao {acq:.4f} fora da faixa do lr uniforme ({U[0][0]:.4f}-{U[-1][0]:.4f})")
        continue
    a0, a1 = lo[-1], hi[0]
    r_u = a0[1] if a1[0] == a0[0] else a0[1] + (a1[1] - a0[1]) * (acq - a0[0]) / (a1[0] - a0[0])
    print(f"  orc {b:g}: aquis {acq:.4f}  SlowHeat ret {ret:.4f}  lr uniforme interpolado {r_u:.4f}  diferenca {100*(ret-r_u):+.2f} pp")
