"""Block 1.1 -- A3 control floors on every rung. CPU only.

WHY THIS BLOCKS THE A3 CLAIM
----------------------------
Phase 1 section 5.0 established that A3's wrapped form carries an artifact: a
hierarchical GP, which satisfies the cross-channel law EXACTLY by construction,
reads R^2 as low as 0.675 through the normaliser and fails the 0.90 gate on 3 of
60 real contexts. Six rungs of TabICL A3 failure (R^2 0.068-0.213) are therefore
unquotable until the same floor is measured on the same rungs.

Two controls, both pushed through the IDENTICAL A3 regression the model gets --
the ambient wrapped Jacobian diagonal against the control's own predictive
variance, `ols(1 + J_ii, s2)`:

  hiergp            mixes LENGTHSCALE at fixed sigma^2. Satisfies A3 exactly.
  noise_hyperprior  mixes SIGMA^2. Also satisfies A3 exactly (Brown applies to
                    any prior), and is the class a reviewer invokes.

DECISION RULE, fixed here rather than after seeing the numbers:
  floor R^2 >= 0.90                      -> the rung's A3 measurement has power;
                                            a model R^2 below the gate is a FAILURE
  floor R^2 <  0.90                      -> the wrapper artifact alone can fail the
                                            gate on this rung, so A3 is UNEVALUABLE
                                            there and is reported as such, not as a
                                            model failure
  floor R^2 within 0.15 of the model R^2 -> UNEVALUABLE regardless of the above;
                                            the control and the model are not
                                            separated

Units: for a wrapped map the recovered slope targets sigma^2 * s_y^2, not
sigma^2 (Phase 1 section 5.7), so slope is reported against that target.

Writes results/b1_a3_floors.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

from controls import NoiseHyperpriorGP                                  # noqa: E402
from experiments.phase_1.core.controls import GP_SIGMA                   # noqa: E402
from experiments.phase_1.core.surrogates import HierarchicalGP           # noqa: E402

RES = HERE.parent / "results"
RUNGS = ["A", "B", "C", "D", "E", "F"]
GATE = 0.90
SEPARATION = 0.15


def ols(x, y):
    Xd = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(coef[1]), float(coef[0]), (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def wrap_J(G, y):
    """Ambient Jacobian of the wrapped map from the inner G (N1), closed form."""
    n = len(y)
    yb, sd = float(np.mean(y)), float(np.std(y)) + 1e-8
    u = (y - yb) / sd
    g = G @ u
    M = np.eye(n) - np.ones((n, n)) / n - np.outer(u, u) / n
    return np.ones((n, n)) / n + np.outer(g, u) / n + G @ M


def a3_for(obj, y):
    """The identical A3 regression the model gets, on a control."""
    sd = float(np.std(y)) + 1e-8
    u = (y - np.mean(y)) / sd
    J = wrap_J(obj.jacobian(u), y)
    s2 = (sd ** 2) * obj.predictive_variance(u)          # un-normalised, Phase 1 5.7
    slope, icept, r2 = ols(1.0 + np.diag(J), s2)
    return {"slope": slope, "intercept": icept, "r2": r2,
            "slope_target": (GP_SIGMA ** 2) * sd ** 2,
            "cv_s": float(np.std(s2) / np.mean(s2)),
            "cv_J": float(np.std(np.diag(J)) / abs(np.mean(np.diag(J))))}


def main():
    z = np.load(RES / "t1_contexts.npz")
    model = {r: json.load(open(RES / f"t1_model_{r}.json"))["rows"]
             for r in RUNGS if (RES / f"t1_model_{r}.json").exists()}
    out = {"_config": {"gate": GATE, "separation": SEPARATION, "rungs": RUNGS,
                       "controls": ["hiergp", "noise_hyperprior"]}, "per_rung": {}}

    print("=" * 104)
    print("B1.1  A3 control floors -- the identical regression, on maps that satisfy A3 EXACTLY")
    print("=" * 104)
    print(f"  {'rung':<5}{'model R2':>11}{'hierGP floor':>22}{'noise-hyper floor':>24}"
          f"{'verdict':>18}")

    for r in RUNGS:
        rows = {"hiergp": [], "noise_hyperprior": []}
        k = 0
        while f"{r}__X__{k}" in z:
            X, y = z[f"{r}__X__{k}"], z[f"{r}__y__{k}"]
            rows["hiergp"].append(a3_for(HierarchicalGP(X, sigma=GP_SIGMA), y))
            rows["noise_hyperprior"].append(a3_for(NoiseHyperpriorGP(X), y))
            k += 1
        agg = {}
        for nm, rs in rows.items():
            v = np.array([q["r2"] for q in rs])
            agg[nm] = {"r2_mean": float(v.mean()), "r2_sd": float(v.std(ddof=1)),
                       "r2_min": float(v.min()), "r2_max": float(v.max()),
                       "n_below_gate": int(np.sum(v < GATE)), "n": len(v)}
        mr2 = (float(np.mean([q["a3_r2"] for q in model[r]]))
               if r in model else float("nan"))

        worst_floor = min(agg["hiergp"]["r2_mean"], agg["noise_hyperprior"]["r2_mean"])
        floor_min = min(agg["hiergp"]["r2_min"], agg["noise_hyperprior"]["r2_min"])
        has_power = worst_floor >= GATE
        separated = abs(worst_floor - mr2) > SEPARATION
        verdict = ("FAILURE" if (has_power and separated) else "UNEVALUABLE")
        out["per_rung"][r] = {"model_a3_r2_mean": mr2, "controls": agg,
                              "worst_floor_mean": worst_floor,
                              "worst_floor_min": floor_min,
                              "floor_has_power": bool(has_power),
                              "separated_from_model": bool(separated),
                              "verdict": verdict}
        print(f"  {r:<5}{mr2:>11.4f}"
              f"{agg['hiergp']['r2_mean']:>13.5f} +-{agg['hiergp']['r2_sd']:<7.5f}"
              f"{agg['noise_hyperprior']['r2_mean']:>15.5f} "
              f"+-{agg['noise_hyperprior']['r2_sd']:<7.5f}{verdict:>18}")

    print("\n" + "=" * 104)
    print("READING")
    print("=" * 104)
    fails = [r for r, e in out["per_rung"].items() if e["verdict"] == "FAILURE"]
    unev = [r for r, e in out["per_rung"].items() if e["verdict"] == "UNEVALUABLE"]
    print(f"  A3 is a genuine FAILURE on rungs : {fails or 'none'}")
    print(f"  A3 is UNEVALUABLE on rungs       : {unev or 'none'}")
    print(f"\n  gate {GATE}; a rung is a failure only if BOTH controls clear the gate")
    print(f"  AND the model sits more than {SEPARATION} below the worse control.")
    out["failure_rungs"] = fails
    out["unevaluable_rungs"] = unev

    p = RES / "b1_a3_floors.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
