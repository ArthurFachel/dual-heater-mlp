"""Fase 3c confirmatory analysis, per goals/protocol_3c_random_fixed.md.

Written BEFORE the run. Reads the 3b manifests (vanilla, exact, frozen_a_control,
lrc_exact, lrc_lorafa; seeds 1000-1009) and the new random_fixed manifests at
results/3c_random_fixed/s<seed>/manifest.json.

--dry-lorafa replaces random_fixed by 3b's frozen_a_control (no new run needed):
the primary must then reproduce 3b's S3 exactly (+0.0028, 5/10). This is the
validation of the script on existing data. Integrity checks that need the new
arm's fields are skipped in dry mode and say so.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path("/mnt/B-SSD/fachel/dh-qwen-f14")
B3 = REPO / "results" / "3b_confirm"
C3 = REPO / "results" / "3c_random_fixed"
SEEDS = list(range(1000, 1010))
E_TOL = 0.01          # V3, same tolerance as 3b
BOUND = 0.01          # declared margin: 1 pp of AP
MAX_INVALID = 2       # V1 / V3: more than 2 failing seeds invalidates

sys.path[:0] = [str(REPO), str(REPO / "src")]
from dual_heater.lora_slowheat import surface_plasticity  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def summ(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    mu, sd = x.mean(), x.std(ddof=1)
    se = sd / math.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    if se > 0:
        p = 2 * stats.t.sf(abs(mu / se), n - 1)
    else:  # identical arms (dry mode): no test is defined
        p = float("nan")
    return mu, sd, mu - tc * se, mu + tc * se, p, int((x > 0).sum()), n


def fmt(r, unit=1.0):
    mu, sd, lo, hi, p, pos, n = r
    return (f"{mu*unit:+.4f} (sd {sd*unit:.4f}) IC95 [{lo*unit:+.4f}; {hi*unit:+.4f}] "
            f"{pos}/{n} >0  p {p:.2e}")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dry-lorafa", action="store_true")
    a = ap_.parse_args()
    dry = a.dry_lorafa

    rows, val = {}, []
    for s in SEEDS:
        main3b = load(B3 / f"s{s}" / "main" / "manifest.json")
        by = {r["method"]: r for r in main3b["results"]}
        reference = int(by["vanilla"]["trainable_parameters"])
        lrc = load(B3 / f"s{s}" / "lrc_exact" / "manifest.json")["results"][0]
        lrl = load(B3 / f"s{s}" / "lrc_lorafa" / "manifest.json")["results"][0]
        if dry:
            rf = by["frozen_a_control"]
            rf_commit = main3b["git_commit"]
        else:
            m = load(C3 / f"s{s}" / "manifest.json")
            rfs = [r for r in m["results"] if r["method"] == "random_fixed"]
            assert len(rfs) == 1, f"seed {s}: random_fixed ausente"
            rf = rfs[0]
            rf_commit = m["git_commit"]
            assert m["protocol"]["seed"] == s
        e_rf = surface_plasticity(
            masked_mean=float(rf.get("masked_mean", 1.0)),
            masked_parameters=int(rf.get("masked_parameters", 0)),
            trainable_parameters=int(rf["trainable_parameters"]),
            reference_parameters=reference,
            lr_scale=float(rf.get("learning_rate_scale", 1.0)),
        )
        e_exact = float(by["exact"]["surface_plasticity"])
        r = {
            "ap": {"vanilla": by["vanilla"]["endpoints"]["ap"], "exact": by["exact"]["endpoints"]["ap"],
                   "frozen_a": by["frozen_a_control"]["endpoints"]["ap"], "lrc_exact": lrc["endpoints"]["ap"],
                   "lrc_lorafa": lrl["endpoints"]["ap"], "random_fixed": rf["endpoints"]["ap"]},
            "fgt": {"vanilla": by["vanilla"]["endpoints"]["forgetting"],
                    "exact": by["exact"]["endpoints"]["forgetting"],
                    "frozen_a": by["frozen_a_control"]["endpoints"]["forgetting"],
                    "lrc_exact": lrc["endpoints"]["forgetting"],
                    "lrc_lorafa": lrl["endpoints"]["forgetting"],
                    "random_fixed": rf["endpoints"]["forgetting"]},
        }
        gimp = rf.get("importance_live_all_stages")
        if dry:
            integrity, overlap = "n/a (dry)", float("nan")
        else:
            audits = [st.get("heat_reassignment") for st in rf["stages"]]
            ok = all(
                au is not None and au["protected_rows"] > 0
                and abs(au["heat_sum_after"] - au["heat_sum_before"]) <= 1e-6 * max(1.0, abs(au["heat_sum_before"]))
                for au in audits
            )
            overlap = float(np.mean([au["own_selection_overlap"] for au in audits if au]))
            ok = ok and overlap < 1.0
            integrity = "OK" if ok else "FALHA"
        val.append((s, gimp, e_rf, e_exact, abs(e_rf - e_exact), integrity, overlap, rf_commit[:7] if rf_commit else None))
        rows[s] = r

    out = []
    P = out.append
    P("Fase 3c CONFIRMATORIO: random_fixed (selecao) no SuperNI-15, Qwen2.5-0.5B LoRA, seeds 1000-1009")
    P("Protocolo: goals/protocol_3c_random_fixed.md. Script escrito ANTES do run." + (
        "  *** DRY: random_fixed = frozen_a_control do 3b ***" if dry else ""))
    P("")
    P("VALIDADE")
    P("seed  G-imp(rf)  E_rf     E_exact  |dE|     integridade  overlap_own  commit")
    for s, g, er, ee, de, integ, ov, c in val:
        P(f"{s}  {str(g):>6}     {er:.4f}   {ee:.4f}   {de:.4f}   {integ:>11}  {ov:.3f}        {c}")
    v1 = sum(1 for v in val if v[1] is not True)
    v3 = sum(1 for v in val if v[4] > E_TOL)
    v2 = sum(1 for v in val if v[5] == "FALHA")
    invalid = (not dry and (v1 > MAX_INVALID or v2 > 0)) or v3 > MAX_INVALID
    P(f"V1 G-imp falhas: {v1} (invalida se > {MAX_INVALID})" + ("  [dry: frozen_a nao tem tracker]" if dry else ""))
    P(f"V2 integridade da realocacao: {v2} seeds com falha (invalida se > 0)")
    P(f"V3 |E_rf - E_exact| > {E_TOL}: {v3} seeds (invalida se > {MAX_INVALID})")
    P(f"VEREDITO DE VALIDADE: {'INVALIDO' if invalid else 'OK'}")
    P("")

    arms = ["vanilla", "lrc_exact", "lrc_lorafa", "frozen_a", "random_fixed", "exact"]
    P("MEDIAS (AP / Fgt, n=10)")
    for k in arms:
        P(f"  {k:14s} AP {np.mean([rows[s]['ap'][k] for s in SEEDS]):.4f}  "
          f"Fgt {np.mean([rows[s]['fgt'][k] for s in SEEDS]):.4f}")
    P("")

    prim = summ([rows[s]["ap"]["exact"] - rows[s]["ap"]["random_fixed"] for s in SEEDS])
    mu, sd, lo, hi, p, pos, n = prim
    P("PRIMARIO: selecao = AP(exact) - AP(random_fixed), t pareado bilateral, alfa 0,05, teste unico")
    P(f"  {fmt(prim)}")
    if invalid:
        verdict = "NAO INTERPRETAVEL (validade falhou)"
    elif p < 0.05 and mu > 0:
        verdict = "CONFIRMADO: a selecao do exact paga sobre linhas aleatorias fixas"
    elif p < 0.05 and mu < 0:
        verdict = "INVERTIDO: linhas aleatorias fixas batem a selecao"
    elif -BOUND < lo and hi < BOUND:
        verdict = f"NULO E LIMITADO: IC95 dentro de +-{BOUND:.2f} AP (1 pp)"
    else:
        verdict = f"NULO NAO LIMITADO: IC95 cruza zero e sai de +-{BOUND:.2f}"
    P(f"  -> {verdict}")
    P("")
    P("SECUNDARIOS (sem correcao, todos reportados)")
    for lab, x, y in (("S1 forma: random_fixed - lrc_exact", "random_fixed", "lrc_exact"),
                      ("S2 random_fixed - frozen_a (E nao pareado)", "random_fixed", "frozen_a"),
                      ("S3 exact - frozen_a (repete o S3 do 3b)", "exact", "frozen_a")):
        P(f"  {lab:44s} AP  {fmt(summ([rows[s]['ap'][x] - rows[s]['ap'][y] for s in SEEDS]))}")
    for lab, x, y in (("S4 Fgt exact - random_fixed (neg favorece exact)", "exact", "random_fixed"),
                      ("S4 Fgt random_fixed - lrc_exact", "random_fixed", "lrc_exact"),
                      ("S4 Fgt random_fixed - frozen_a", "random_fixed", "frozen_a")):
        P(f"  {lab:44s} Fgt {fmt(summ([rows[s]['fgt'][x] - rows[s]['fgt'][y] for s in SEEDS]))}")
    if not dry:
        P(f"  S5 sobreposicao media das linhas protegidas com a propria selecao: "
          f"{np.mean([v[6] for v in val]):.3f}")
    P("")
    P("POR SEED (AP)")
    P("seed  " + "  ".join(f"{k[:12]:>12}" for k in arms))
    for s in SEEDS:
        P(f"{s}  " + "  ".join(f"{rows[s]['ap'][k]:12.4f}" for k in arms))

    txt = "\n".join(out)
    print(txt)
    if not dry:
        C3.mkdir(parents=True, exist_ok=True)
        (C3 / "analysis_preregistered.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
