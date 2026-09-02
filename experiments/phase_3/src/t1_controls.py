"""T1.4 -- the control battery on every rung. Tests P5, P6, P7. CPU only.

Six controls, all through the IDENTICAL pipeline the model gets: the same
normaliser wrapper, the same seeded Q, the same asym/negeig, the same row and
column profiling.

  exactgp_unwrapped     Phase 1 control. Brown applies directly; asym ~ 0.
  exactgp_wrapped       Phase 1 control. The projection floor.
  hiergp                Phase 1 control. Mixes LENGTHSCALE at fixed sigma^2, so
                        Brown still applies and asym stays ~1e-8. This is the
                        control the paper currently has, and P6's whole point is
                        that it is the WRONG floor for A1.
  noise_hyperprior      NEW. Mixes sigma^2. T3.1 proves its Jacobian is symmetric
                        PSD plus a rank-one term along y, so it is a genuine
                        Bayesian object with genuinely nonzero raw asymmetry.
                        THE CLASS-LEVEL A1 FLOOR.          -> P5, P6
  loo_smoother          NEW. J = D^-1 (K - diag K). Row-scaled symmetric by
                        construction, live A2-failure candidate.    -> P7
  targeted_imitator     Phase 1 control. Bayes-impossible by construction; must
                        survive the projection.

Everything is analytic, so the Jacobian is taken in closed form and pushed
through the same metrics rather than finite-differenced -- the finite-difference
agreement was already established in Phase 1's regression tests and in T0.3 D1.

Writes results/t1_controls.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

from controls import LOOSmoother, NoiseHyperpriorGP, SIGMA2_GRID       # noqa: E402
from experiments.phase_1.core.controls import (                         # noqa: E402
    GP_SIGMA, TargetedImitator,
)
from experiments.phase_1.core.metrics import (                          # noqa: E402
    asym, get_Q, negeig, profile_jacobian,
)
from experiments.phase_1.core.surrogates import ExactGP, HierarchicalGP  # noqa: E402

RUNGS = ["A", "B", "C", "D", "E", "F"]
Q_SEED = 0
NAMES = ["exactgp_unwrapped", "exactgp_wrapped", "hiergp",
         "noise_hyperprior", "loo_smoother", "targeted_imitator"]


def wrap_J(G, y):
    """Ambient Jacobian of the normaliser-wrapped map, from the inner G.

    N1:  J = (1/n) 1 1' + (1/n) g u' + G M,   M = I - (1/n)11' - (1/n)uu'
    Built in closed form so the control is exact rather than finite-differenced.
    """
    n = len(y)
    yb, sd = float(np.mean(y)), float(np.std(y)) + 1e-8
    u = (y - yb) / sd
    g = G @ u
    M = np.eye(n) - np.ones((n, n)) / n - np.outer(u, u) / n
    return np.ones((n, n)) / n + np.outer(g, u) / n + G @ M


def metrics(J, Jr):
    return {"ambient_asym": float(asym(J)), "ambient_negeig": float(negeig(J)),
            "proj_asym": float(asym(Jr)), "proj_negeig": float(negeig(Jr)),
            "normF": float(np.linalg.norm(J, "fro"))}


def run_context(X, y):
    """All six controls on one context."""
    Q = get_Q(y, seed=Q_SEED)
    out = {}

    egp = ExactGP(X, sigma=GP_SIGMA)
    Ju = egp.W
    out["exactgp_unwrapped"] = metrics(Ju, Q.T @ Ju @ Q)
    Jw = wrap_J(egp.W, y)
    out["exactgp_wrapped"] = metrics(Jw, Q.T @ Jw @ Q)

    u = (y - np.mean(y)) / (np.std(y) + 1e-8)
    for name, obj in [("hiergp", HierarchicalGP(X, sigma=GP_SIGMA)),
                      ("noise_hyperprior", NoiseHyperpriorGP(X))]:
        G = obj.jacobian(u)
        J = wrap_J(G, y)
        out[name] = metrics(J, Q.T @ J @ Q)

    smo = LOOSmoother(X)
    Js = wrap_J(smo.W, y)
    r = out["loo_smoother"] = metrics(Js, Q.T @ Js @ Q)
    for mode in ("row", "column"):
        pr = profile_jacobian(smo.W, mode)          # profile the INNER map
        r[f"{mode}_residual"] = float(pr["residual_rel"])
        r[f"{mode}_post_asym"] = float(pr["asym"])

    ti = TargetedImitator(X, y, seed=0)
    Jt = wrap_J(ti.inner_jacobian(u), y)
    out["targeted_imitator"] = metrics(Jt, Q.T @ Jt @ Q)
    return out


def main():
    z = np.load(HERE.parent / "results" / "t1_contexts.npz")
    out = {"_config": {"rungs": RUNGS, "Q_seed": Q_SEED, "gp_sigma": GP_SIGMA,
                       "sigma2_grid": SIGMA2_GRID.tolist(),
                       "controls": NAMES}, "per_rung": {}}

    print("=" * 100)
    print("T1.4  control battery, all six rungs -- identical pipeline")
    print("=" * 100)

    for rung in RUNGS:
        t0 = time.time()
        rows = []
        k = 0
        while f"{rung}__X__{k}" in z:
            X = z[f"{rung}__X__{k}"]; y = z[f"{rung}__y__{k}"]
            rows.append(run_context(X, y))
            k += 1
        agg = {}
        for nm in NAMES:
            keys = rows[0][nm].keys()
            agg[nm] = {kk: {"mean": float(np.mean([r[nm][kk] for r in rows])),
                            "sd": float(np.std([r[nm][kk] for r in rows], ddof=1)),
                            "max": float(np.max([r[nm][kk] for r in rows])),
                            "min": float(np.min([r[nm][kk] for r in rows]))}
                       for kk in keys}
        out["per_rung"][rung] = {"n": len(rows), "aggregate": agg, "rows": rows}
        print(f"\n  RUNG {rung}   ({len(rows)} contexts, {time.time()-t0:.0f}s)")
        print(f"    {'control':<20}{'proj asym':>22}{'proj negeig':>22}{'amb asym':>12}")
        for nm in NAMES:
            a = agg[nm]
            print(f"    {nm:<20}"
                  f"{a['proj_asym']['mean']:>10.3e} +-{a['proj_asym']['sd']:<9.2e}"
                  f"{a['proj_negeig']['mean']:>10.3e} +-{a['proj_negeig']['sd']:<9.2e}"
                  f"{a['ambient_asym']['mean']:>12.4f}")

    # ------------------------------------------------------------- P5, P6, P7
    print("\n" + "=" * 100)
    print("REGISTERED PREDICTIONS")
    print("=" * 100)
    nh = {r: out["per_rung"][r]["aggregate"]["noise_hyperprior"] for r in RUNGS}
    lo = {r: out["per_rung"][r]["aggregate"]["loo_smoother"] for r in RUNGS}

    p5_max = max(nh[r]["proj_negeig"]["max"] for r in RUNGS)
    p5 = p5_max < 1e-10
    print(f"\n  P5  noise-hyperprior GP projected negeig < 1e-10, every rung")
    for r in RUNGS:
        print(f"        rung {r}  max {nh[r]['proj_negeig']['max']:.3e}")
    print(f"      -> worst {p5_max:.3e}   {'PASS' if p5 else '*** FAIL ***'}")

    means = [nh[r]["proj_asym"]["mean"] for r in RUNGS]
    maxes = [nh[r]["proj_asym"]["max"] for r in RUNGS]
    amb_a = [nh[r]["ambient_asym"]["mean"] for r in RUNGS]
    amb_n = [nh[r]["ambient_negeig"]["mean"] for r in RUNGS]
    p6_mean = all(0.02 <= m <= 0.12 for m in means)
    p6_max = max(maxes) <= 0.45
    p6_amb_a = all(0.15 <= v <= 0.36 for v in amb_a)
    p6_amb_n = all(0.07 <= v <= 0.62 for v in amb_n)
    print(f"\n  P6  noise-hyperprior GP is the honest A1 floor")
    print(f"      {'rung':<6}{'proj asym mean':>16}{'proj asym max':>15}"
          f"{'amb asym':>11}{'amb negeig':>13}")
    for r in RUNGS:
        print(f"      {r:<6}{nh[r]['proj_asym']['mean']:>16.4f}"
              f"{nh[r]['proj_asym']['max']:>15.4f}"
              f"{nh[r]['ambient_asym']['mean']:>11.4f}"
              f"{nh[r]['ambient_negeig']['mean']:>13.4f}")
    print(f"      proj mean in [0.02,0.12] : {'PASS' if p6_mean else 'MISS'}")
    print(f"      proj max  <= 0.45        : {'PASS' if p6_max else 'MISS'}  ({max(maxes):.4f})")
    print(f"      amb asym in [0.15,0.36]  : {'PASS' if p6_amb_a else 'MISS'}")
    print(f"      amb negeig in [0.07,0.62]: {'PASS' if p6_amb_n else 'MISS'}")

    p7_neg = all(lo[r]["proj_negeig"]["mean"] >= 0.15 for r in RUNGS)
    p7_row = max(lo[r]["row_residual"]["max"] for r in RUNGS) <= 1e-10
    print(f"\n  P7  LOO smoother")
    print(f"      {'rung':<6}{'proj negeig mean':>18}{'row residual max':>20}")
    for r in RUNGS:
        print(f"      {r:<6}{lo[r]['proj_negeig']['mean']:>18.4f}"
              f"{lo[r]['row_residual']['max']:>20.3e}")
    print(f"      proj negeig >= 0.15 : {'PASS' if p7_neg else 'MISS'}")
    print(f"      row residual <= 1e-10: {'PASS' if p7_row else 'MISS'}")

    out["predictions"] = {
        "P5": {"worst_proj_negeig": p5_max, "gate": 1e-10, "PASS": bool(p5)},
        "P6": {"proj_asym_mean_by_rung": dict(zip(RUNGS, means)),
               "proj_asym_max_by_rung": dict(zip(RUNGS, maxes)),
               "ambient_asym_mean_by_rung": dict(zip(RUNGS, amb_a)),
               "ambient_negeig_mean_by_rung": dict(zip(RUNGS, amb_n)),
               "proj_mean_in_range": bool(p6_mean), "proj_max_ok": bool(p6_max),
               "ambient_asym_in_range": bool(p6_amb_a),
               "ambient_negeig_in_range": bool(p6_amb_n)},
        "P7": {"proj_negeig_mean_by_rung": {r: lo[r]["proj_negeig"]["mean"] for r in RUNGS},
               "row_residual_max_by_rung": {r: lo[r]["row_residual"]["max"] for r in RUNGS},
               "negeig_PASS": bool(p7_neg), "row_PASS": bool(p7_row)}}

    p = HERE.parent / "results" / "t1_controls.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
