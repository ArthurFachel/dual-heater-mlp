"""Validation of analyze_t3c.paired/load on F11 (must reproduce +0.62 / +1.91 / +1.49 / +0.39 pp)."""
import sys
sys.argv = ["x"]
from analyze_t3c import load, paired, TAILS, FAA
rows, _ = load("results/confirm_uniform_control")
seeds = sorted({s for s, _ in rows})
for b, t in TAILS.items():
    m = [float(rows[(s, f"slowheat_{t}")][FAA]) for s in seeds]
    u = [float(rows[(s, f"uniform_{t}")][FAA]) for s in seeds]
    r = paired(m, u)
    print(f"{b:18s} M-U {100*r[0]:+.2f} pp [{100*r[1]:+.2f}; {100*r[2]:+.2f}] {r[4]}/{r[5]} n={len(seeds)}")
