"""T1.6 -- aggregate the ladder, score the pre-registered readings, test P8/P9.

Reads whatever rungs have completed, so it can be run mid-flight and re-run at
the end. No model calls.

WHAT IT DECIDES
---------------
T1.5 registered two readings and a fallback, and the CRITERIA ARE APPLIED AS
WRITTEN -- the point of pre-registering them is that they are not adjusted after
seeing the numbers:

  Reading 1  fails prior-family      rung A proj asym >= 0.4  AND  negeig >= 0.10
  Reading 2  passes prior-family     rung A asym <= 3x the P6 floor (~0.15)
                                     AND  negeig <= 0.02,  with D/E failing
  neither    report as inconclusive between the two and lead with the
             dose-response curve instead of a binary verdict

  P8  violation decreases in the T1.3 coordinate, pooled Spearman <= -0.4.
      A flat or non-monotone curve IS the headline and is reported as such.
  P9  prior-family trJ/n <= 0.9.

THE NON-DEGENERACY GATE IS APPLIED BEFORE ANY VERDICT
-----------------------------------------------------
E2.1: `J ~ I` passes A1/A2 vacuously and `J ~ 0` is instrument noise. Contexts
failing `||J-I||_F/||J||_F > 0.3` are counted and EXCLUDED from the verdict
means, and both the gated and ungated numbers are reported -- if a rung is mostly
degenerate, that is the finding, not a pass.

Writes results/t1_summary.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
RUNGS = ["A", "B", "C", "D", "E", "F"]
NONDEG = 0.3
P6_FLOOR_KEY = "proj_asym"


def load_rung(r):
    p = RES / f"t1_model_{r}.json"
    return json.load(open(p))["rows"] if p.exists() else None


def load_coord(r):
    p = RES / f"t1_coordinate_{r}.json"
    return {q["index"]: q["logml_per_obs"] for q in json.load(open(p))["rows"]} \
        if p.exists() else {}


def load_circ(r):
    p = RES / f"t1_circulation_{r}.json"
    return {q["index"]: q for q in json.load(open(p))["rows"]} if p.exists() else {}


def msd(v):
    v = np.asarray(v, float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def main():
    a3p = RES / "b1_a3_floors.json"
    a3 = json.load(open(a3p))["per_rung"] if a3p.exists() else {}
    ctrl_p = RES / "t1_controls.json"
    ctrl = json.load(open(ctrl_p)) if ctrl_p.exists() else None
    floors = {r: ctrl["per_rung"][r]["aggregate"]["noise_hyperprior"][P6_FLOOR_KEY]["mean"]
              for r in RUNGS} if ctrl else {}

    out = {"_config": {"nondeg_gate": NONDEG, "rungs": RUNGS}, "per_rung": {}}
    pooled = {"asym": [], "negeig": [], "coord": []}

    print("=" * 108)
    print("T1.6  ladder summary   (gated = non-degenerate contexts only, ||J-I||/||J|| > 0.3)")
    print("=" * 108)
    # Block 1.2: the ratio column is REMOVED. The class floor varies ~250x across
    # rungs (0.0002 on F to 0.0505 on D), so asym/floor ranked the two LEAST
    # violating rungs above the two worst. Violation and floor are separate
    # columns; the reader divides where the division is meaningful, which is
    # within a rung and not across them.
    print(f"  {'rung':<5}{'n':>4}{'nondeg':>8}{'asym (gated)':>20}{'negeig (gated)':>20}"
          f"{'trJ/n':>16}{'A1 floor':>10}{'A3 R2':>9}{'A3 floor':>10}{'A3 verdict':>13}")

    for r in RUNGS:
        rows = load_rung(r)
        if not rows:
            continue
        coord = load_coord(r)
        circ = load_circ(r)
        keep = [q for q in rows if q["J_minus_I"] > NONDEG]
        e = {"n": len(rows), "n_nondegenerate": len(keep),
             "frac_nondegenerate": len(keep) / len(rows)}
        for k in ("asym", "negeig", "trJ_over_n", "J_minus_I", "a3_r2",
                  "row_residual", "col_residual", "n_neg_diag", "skew_fro"):
            m, s = msd([q[k] for q in rows])
            e[f"{k}_all_mean"], e[f"{k}_all_sd"] = m, s
            if keep:
                m2, s2 = msd([q[k] for q in keep])
                e[f"{k}_mean"], e[f"{k}_sd"] = m2, s2
        e["frac_negeig_zero"] = float(np.mean([q["negeig"] == 0 for q in rows]))
        if r in floors:
            e["class_floor_proj_asym"] = floors[r]
            # kept in the JSON for within-rung use only; NOT printed and NOT
            # comparable across rungs -- see the note above the header
            e["asym_over_floor_WITHIN_RUNG_ONLY"] = (
                e.get("asym_mean", e["asym_all_mean"]) / floors[r]) if floors[r] > 0 else None
        if coord:
            e["coord_n"] = len(coord)
            e["coord_mean"], e["coord_sd"] = msd(list(coord.values()))
            for q in rows:
                if q["index"] in coord:
                    pooled["asym"].append(q["asym"])
                    pooled["negeig"].append(q["negeig"])
                    pooled["coord"].append(coord[q["index"]])
        if circ:
            cv = [c["circulation_fro"] for c in circ.values()]
            e["circulation_mean"], e["circulation_sd"] = msd(cv)
            rr = [c["ratio_C_over_FD"] for c in circ.values()
                  if c.get("ratio_C_over_FD")]
            if rr:
                e["ratio_C_over_FD_mean"], e["ratio_C_over_FD_sd"] = msd(rr)
        out["per_rung"][r] = e

        fl = f"{floors[r]:.4f}" if r in floors else "-"
        a3f = a3.get(r, {})
        a3fl = (f"{a3f['worst_floor_mean']:.4f}" if a3f else "-")
        a3v = a3f.get("verdict", "-")
        e["a3_floor_mean"] = a3f.get("worst_floor_mean")
        e["a3_verdict"] = a3v
        print(f"  {r:<5}{e['n']:>4}{e['n_nondegenerate']:>8}"
              f"{e.get('asym_mean', float('nan')):>12.4f} +-{e.get('asym_sd', 0):<6.3f}"
              f"{e.get('negeig_mean', float('nan')):>12.4f} +-{e.get('negeig_sd', 0):<6.3f}"
              f"{e['trJ_over_n_all_mean']:>10.3f} +-{e['trJ_over_n_all_sd']:<5.3f}"
              f"{fl:>10}{e['a3_r2_all_mean']:>9.4f}{a3fl:>10}{a3v:>13}")

    # ------------------------------------------------------------- the readings
    A = out["per_rung"].get("A")
    if A and A["n"] >= 10:
        a_as = A.get("asym_mean", A["asym_all_mean"])
        a_ne = A.get("negeig_mean", A["negeig_all_mean"])
        floor = floors.get("A", 0.05)
        r1 = a_as >= 0.4 and a_ne >= 0.10
        r2 = a_as <= 3 * floor and a_ne <= 0.02
        out["readings"] = {"rung_A_asym_gated": a_as, "rung_A_negeig_gated": a_ne,
                           "P6_floor_A": floor, "reading_1": bool(r1),
                           "reading_2": bool(r2),
                           "verdict": ("Reading 1 -- fails prior-family" if r1 else
                                       "Reading 2 -- passes prior-family" if r2 else
                                       "NEITHER -- inconclusive between the two; "
                                       "lead with the dose-response curve")}
        print("\n" + "=" * 108)
        print("T1.5 READINGS, criteria applied as pre-registered")
        print("=" * 108)
        print(f"  rung A gated   asym {a_as:.4f}   negeig {a_ne:.4f}   "
              f"(P6 floor {floor:.4f}, 3x = {3*floor:.4f})")
        print(f"  Reading 1 (asym>=0.4 and negeig>=0.10) : {r1}")
        print(f"  Reading 2 (asym<=3xfloor and negeig<=0.02): {r2}")
        print(f"  -> {out['readings']['verdict']}")
        print(f"\n  NON-DEGENERACY on rung A: {A['n_nondegenerate']}/{A['n']} pass the gate; "
              f"negeig is exactly zero on {A['frac_negeig_zero']*100:.0f}% of contexts")

    # ------------------------------------------------------------------- P8, P9
    if len(pooled["coord"]) >= 10:
        sa = stats.spearmanr(pooled["coord"], pooled["asym"])
        sn = stats.spearmanr(pooled["coord"], pooled["negeig"])
        out["P8"] = {"n": len(pooled["coord"]),
                     "spearman_coord_vs_asym": float(sa.statistic),
                     "p_asym": float(sa.pvalue),
                     "spearman_coord_vs_negeig": float(sn.statistic),
                     "p_negeig": float(sn.pvalue),
                     "gate": -0.4,
                     "PASS": bool(sa.statistic <= -0.4)}
        print(f"\n  P8  pooled n={len(pooled['coord'])}   "
              f"Spearman(coord, asym) = {sa.statistic:+.4f} (p={sa.pvalue:.2e})   "
              f"Spearman(coord, negeig) = {sn.statistic:+.4f}")
        print(f"      gate <= -0.40 : {'PASS' if sa.statistic <= -0.4 else 'MISS'}"
              f"   {'(a flat/non-monotone curve IS the headline)' if sa.statistic > -0.4 else ''}")

    if A:
        p9 = A["trJ_over_n_all_mean"] <= 0.9
        out["P9"] = {"rung_A_trJ_over_n_mean": A["trJ_over_n_all_mean"],
                     "rung_A_trJ_over_n_max": None, "gate": 0.9, "PASS": bool(p9)}
        p9_msg = ("PASS" if p9 else
                  "*** MISS -- the model interpolates its own prior's contexts "
                  "and the non-degeneracy gate fails at home ***")
        print(f"\n  P9  rung A trJ/n mean = {A['trJ_over_n_all_mean']:.4f}   "
              f"gate <= 0.9 : {p9_msg}")

    json.dump(out, open(RES / "t1_summary.json", "w"), indent=2, default=float)
    print(f"\nSaved: {RES / 't1_summary.json'}")


if __name__ == "__main__":
    main()
