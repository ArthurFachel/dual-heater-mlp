"""Fase 3b confirmatory analysis, per goals/protocol_3b_confirmatory.md (commit deea8d2).
Reads results/3b_confirm/s<seed>/{main,lrc_exact,lrc_lorafa}/manifest.json."""
import json, math, os, sys
from scipy import stats
import numpy as np

ROOT = "/mnt/B-SSD/fachel/dh-qwen-f14/results/3b_confirm"
SEEDS = list(range(1000, 1010))
LORAFA_E = 2039808 / 3317760


def load(p):
    with open(p) as f:
        return json.load(f)


rows = {}
val = []
for s in SEEDS:
    m = load(f"{ROOT}/s{s}/main/manifest.json")
    le = load(f"{ROOT}/s{s}/lrc_exact/manifest.json")
    ll = load(f"{ROOT}/s{s}/lrc_lorafa/manifest.json")
    g = m["gates"]["G-disp"]
    r = {
        "ap": dict(g["ap_by_arm"]),
        "fgt": dict(g["forgetting_by_arm"]),
        "E": dict(g["surface_plasticity_by_arm"]),
    }
    for name, mm in (("lrc_exact", le), ("lrc_lorafa", ll)):
        gg = mm["gates"]["G-disp"]
        r["ap"][name] = gg["ap_by_arm"]["lr_control"]
        r["fgt"][name] = gg["forgetting_by_arm"]["lr_control"]
        # lr_control trains the same parameters as vanilla, so E = learning_rate_scale (pitfall 24)
        r["E"][name] = mm["protocol"]["learning_rate_scale"]
    gimp = m["gates"]["G-imp"]["exact"]
    van_aq = sum(1 for t in m["gates"]["G-aq"]["vanilla"]["per_task"] if t["passes"])
    v2e = abs(r["E"]["exact"] - r["E"]["lrc_exact"])
    v2l = abs(r["E"]["frozen_a_control"] - r["E"]["lrc_lorafa"])
    val.append((s, gimp, v2e, v2l, van_aq, r["fgt"]["vanilla"], m["git_commit"][:7]))
    rows[s] = r


def summ(x):
    x = np.asarray(x)
    n = len(x)
    mu = x.mean()
    sd = x.std(ddof=1)
    se = sd / math.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    t = mu / se if se > 0 else float("inf")
    p = 2 * stats.t.sf(abs(t), n - 1)
    pos = int((x > 0).sum())
    return mu, sd, mu - tc * se, mu + tc * se, p, pos, n


out = []
P = out.append
P("Fase 3b CONFIRMATORIO, SuperNI-15, Qwen2.5-0.5B LoRA, 1 epoca, seeds 1000-1009")
P("Protocolo: goals/protocol_3b_confirmatory.md, commit deea8d2. Analise escrita depois do CONFIRM DONE (11/11 11:57).")
P("")
P("VALIDADE")
P("seed  G-imp(exact)  |dE exact|  |dE lorafa|  G-aq vanilla  Fgt vanilla  commit")
for s, gimp, a, b, aq, fv, c in val:
    P(f"{s}  {str(gimp):>5}        {a:.2e}    {b:.2e}     {aq:>2}/15        {fv:.4f}      {c}")
nfail = sum(1 for v in val if v[1] is not True)
P(f"V1 G-imp falhas: {nfail} (invalida se > 2) -> {'OK' if nfail <= 2 else 'INVALIDO'}")
v2ok = all(v[2] <= 0.01 and v[3] <= 0.01 for v in val)
P(f"V2 pareamento E <= 0,01 em toda seed: {'OK' if v2ok else 'FALHA'}")
P(f"V3 G-aq vanilla >= 11/15: {sum(1 for v in val if v[4] >= 11)}/10 seeds; Fgt vanilla >= 0,05: {sum(1 for v in val if v[5] >= 0.05)}/10 (nao excluem)")
P("V4 relancamentos: nenhum (todos os 30 processos try1, exit=0)")
P("")

P("MEDIAS POR BRACO (AP / Fgt, n=10)")
arms = ["vanilla", "exact", "frozen_a_control", "lrc_exact", "lrc_lorafa"]
for a in arms:
    ap = np.mean([rows[s]["ap"][a] for s in SEEDS])
    fg = np.mean([rows[s]["fgt"][a] for s in SEEDS])
    P(f"  {a:18s} AP {ap:.4f}  Fgt {fg:.4f}")
P("")

pair = {"exact": "lrc_exact", "frozen_a_control": "lrc_lorafa"}
P("PRIMARIO: D_unmatched - D_matched = AP(lrc_X) - AP(vanilla), t pareado bilateral, Holm sobre 2")
prim = {}
for X, L in pair.items():
    d = [rows[s]["ap"][L] - rows[s]["ap"]["vanilla"] for s in SEEDS]
    prim[X] = summ(d)
ps = sorted(prim.items(), key=lambda kv: kv[1][4])
holm = {}
for i, (X, r) in enumerate(ps):
    holm[X] = min(1.0, max(r[4] * (2 - i), holm.get(ps[i - 1][0], 0) if i else 0))
for X, r in prim.items():
    mu, sd, lo, hi, p, pos, n = r
    du = np.mean([rows[s]["ap"][X] - rows[s]["ap"]["vanilla"] for s in SEEDS])
    dm = np.mean([rows[s]["ap"][X] - rows[s]["ap"][pair[X]] for s in SEEDS])
    P(f"  {X:18s} {mu:+.4f} (sd {sd:.4f}) IC95 [{lo:+.4f}; {hi:+.4f}] {pos}/{n} >0  p {p:.2e}  Holm {holm[X]:.2e}  "
      f"{'SIGNIFICATIVO' if holm[X] < 0.05 and mu > 0 else 'nao significativo'}")
    P(f"      D_unmatched medio {du:+.4f}, D_matched medio {dm:+.4f}, fracao D_matched/D_unmatched {dm/du:.3f}" if du else "")
P("")
P("S1: D_matched(X) = AP(X) - AP(lrc_X)")
s1 = {}
for X, L in pair.items():
    s1[X] = summ([rows[s]["ap"][X] - rows[s]["ap"][L] for s in SEEDS])
    mu, sd, lo, hi, p, pos, n = s1[X]
    P(f"  {X:18s} {mu:+.4f} (sd {sd:.4f}) IC95 [{lo:+.4f}; {hi:+.4f}] {pos}/{n} >0  p {p:.2e}  -> "
      f"{'IC acima de zero: selecao paga com E pareado' if lo > 0 else ('IC abaixo de zero' if hi < 0 else 'IC cruza zero: nao confirmado')}")
P("")
P("S2: com Fgt (menos e melhor; valores = Fgt(a) - Fgt(b), negativo favorece a)")
for X, L in pair.items():
    for lab, a, b in ((f"lrc_{X} - vanilla", L, "vanilla"), (f"{X} - lrc", X, L), (f"{X} - vanilla", X, "vanilla")):
        mu, sd, lo, hi, p, pos, n = summ([rows[s]["fgt"][a] - rows[s]["fgt"][b] for s in SEEDS])
        P(f"  {lab:30s} {mu:+.4f} IC95 [{lo:+.4f}; {hi:+.4f}] {n-pos}/{n} <0  p {p:.2e}")
P("")
P("S3: exact - frozen_a_control (E nao pareado, descritivo)")
for k in ("ap", "fgt"):
    mu, sd, lo, hi, p, pos, n = summ([rows[s][k]["exact"] - rows[s][k]["frozen_a_control"] for s in SEEDS])
    P(f"  {k.upper():3s} {mu:+.4f} IC95 [{lo:+.4f}; {hi:+.4f}] {pos}/{n} >0  p {p:.2e}")
P("")
P("S4: ordenacao por AP media")
order = sorted(arms, key=lambda a: -np.mean([rows[s]["ap"][a] for s in SEEDS]))
P("  " + " > ".join(order))
P("")
P("POR SEED (AP)")
P("seed  " + "  ".join(f"{a[:10]:>10}" for a in arms))
for s in SEEDS:
    P(f"{s}  " + "  ".join(f"{rows[s]['ap'][a]:10.4f}" for a in arms))

txt = "\n".join(out)
print(txt)
with open(f"{ROOT}/analysis_preregistered.txt", "w") as f:
    f.write(txt + "\n")
