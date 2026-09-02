"""P1, third attempt: is FD-at-y unrepresentative of the sphere the loop averages over?

WHERE THE ARGUMENT STANDS
-------------------------
  P1   loop / FD skew ratio on TabICL = 0.166. Falsifier fired.
  D1   on a wrapped TargetedImitator, FD reproduces the ANALYTIC inner Jacobian
       exactly and the loop recovers it to 11.5%. Both estimators are correct.
  D2   same on a wrapped nonlinear tanh^3 map, loop within 20.3%.
  D3   using loop planes that CONTAIN y does not close the gap (0.119-0.126,
       no better than the 0.133-0.202 of perpendicular planes).

So the estimators are sound and "the plane misses y" is not the explanation.

WHAT IS LEFT, AND WHY IT IS FORCED
----------------------------------
For a normaliser-wrapped map, a constant-ybar constant-r loop sends u around the
SAME circle of radius sqrt(n) whatever r is (established in the P1 radius sweep,
and derived: m = ybar*1 + s_y*g(u) with both statistics constant on the loop).
So the loop CANNOT be made local -- it always averages the antisymmetric 2-form
over a great circle of the u-sphere. FD, by contrast, is a point value at u_actual.

If TabICL's antisymmetry is much larger at the actual data configuration than at
a generic point of the same sphere, both estimators are right and simply measure
different things. This script tests that directly, without using the loop at all:

    measure the SAME finite-difference skew that Phase 1 measures, at K random
    points on the same constant-(ybar, r) sphere, and compare to the value at y.

PREDICTION, stated before running: if the locality reading is right, the
sphere-average FD skew should be roughly 6x smaller than FD at y, i.e. close to
the loop's ~0.63, and the loop and FD are reconciled. If the sphere-average FD
skew is instead close to FD-at-y (~4.2-4.7), locality is refuted too, the two
estimators disagree on the same object, and halt H2 fires for real.

Writes results/t0_3_sphere_locality.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np


class _MockAnalytics:
    def __getattr__(self, name):
        return lambda *a, **k: None


sys.modules["analytics"] = _MockAnalytics()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

import torch                                                            # noqa: E402
from models import load                                                 # noqa: E402
from experiments.phase_1.core.context import generate_audit_context      # noqa: E402
from experiments.phase_1.core.metrics import get_Q, reduced_jacobian     # noqa: E402

SEEDS = [42, 100, 200]
K_POINTS = 6          # random points on the sphere, per context
T = 1e-3


def sphere_point(y, rng):
    """A random point with the SAME mean and the same centred radius as y."""
    yb = float(np.mean(y))
    r = float(np.linalg.norm(y - yb))
    v = rng.standard_normal(len(y))
    v -= v.mean()
    v *= r / np.linalg.norm(v)
    return yb + v


def skew_at(predict, y_point, t=T):
    """Phase 1's own quantity: ||Q'JQ - (Q'JQ)'||_F, at y_point.

    Q is built from y_point, so this is the same construction Phase 1 applies at
    y -- the projection always removes span{1, u} of the point being probed.
    """
    Q = get_Q(y_point, seed=0)
    J, _, _ = reduced_jacobian(predict, y_point, Q, t)
    return float(np.linalg.norm(J - J.T, "fro")), float(np.linalg.norm(J, "fro"))


def main():
    model = load("tabicl_v2", task="regression", device="cuda")
    out = {"_config": {"seeds": SEEDS, "k_points": K_POINTS, "t": T,
                       "quantity": "||Q'JQ - (Q'JQ)'||_F at points on the "
                                   "constant-(ybar, r) sphere"},
           "rows": []}
    print("=" * 96)
    print("Is FD-at-y representative of the sphere the loop averages over?")
    print(f"  {K_POINTS} random points per context on the same constant-(ybar, r) sphere")
    print("=" * 96)

    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)

        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            return model.estimator.predict(X)

        t0 = time.time()
        at_y, nrm_y = skew_at(predict, y)
        rng = np.random.default_rng(7000 + s)
        sph, nrms = [], []
        for k in range(K_POINTS):
            sk, nk = skew_at(predict, sphere_point(y, rng))
            sph.append(sk); nrms.append(nk)
        sph = np.asarray(sph)
        row = {"seed": s, "skew_at_y": at_y, "normF_at_y": nrm_y,
               "skew_sphere_mean": float(sph.mean()),
               "skew_sphere_sd": float(sph.std(ddof=1)),
               "skew_sphere_min": float(sph.min()),
               "skew_sphere_max": float(sph.max()),
               "normF_sphere_mean": float(np.mean(nrms)),
               "ratio_sphere_over_y": float(sph.mean() / at_y),
               "seconds": time.time() - t0}
        out["rows"].append(row)
        print(f"\n  seed {s}")
        print(f"    FD skew AT y                    {at_y:.5f}   (||J||_F {nrm_y:.4f})")
        print(f"    FD skew on the sphere, {K_POINTS} pts   "
              f"{sph.mean():.5f} +- {sph.std(ddof=1):.5f}  "
              f"[{sph.min():.5f}, {sph.max():.5f}]   (||J||_F {np.mean(nrms):.4f})")
        print(f"    sphere / at-y                   {sph.mean()/at_y:.3f}"
              f"     ({time.time()-t0:.0f}s)")

    r = out["rows"]
    mean_ratio = float(np.mean([q["ratio_sphere_over_y"] for q in r]))
    out["mean_ratio_sphere_over_y"] = mean_ratio
    # the loop's own answer, from P1
    p1 = json.load(open(HERE.parent / "results" / "t0_3_reconcile_tabicl.json"))
    loop_by_seed = {q["seed"]: q["loop_skew_fro"] for q in p1["rows"]}
    out["loop_vs_sphere"] = [
        {"seed": q["seed"], "loop": loop_by_seed[q["seed"]],
         "fd_sphere_mean": q["skew_sphere_mean"],
         "loop_over_fd_sphere": loop_by_seed[q["seed"]] / q["skew_sphere_mean"]}
        for q in r]

    print("\n" + "=" * 96)
    print("RECONCILIATION")
    print("=" * 96)
    print(f"  {'seed':<6}{'FD at y':>10}{'FD sphere':>12}{'loop':>10}"
          f"{'loop/FD_sphere':>16}{'loop/FD_at_y':>14}")
    for q, lv in zip(r, out["loop_vs_sphere"]):
        print(f"  {q['seed']:<6}{q['skew_at_y']:>10.4f}{q['skew_sphere_mean']:>12.4f}"
              f"{lv['loop']:>10.4f}{lv['loop_over_fd_sphere']:>16.3f}"
              f"{lv['loop']/q['skew_at_y']:>14.3f}")
    lr = float(np.mean([q["loop_over_fd_sphere"] for q in out["loop_vs_sphere"]]))
    out["mean_loop_over_fd_sphere"] = lr
    print(f"\n  sphere / at-y            {mean_ratio:.3f}")
    print(f"  loop / FD-on-sphere      {lr:.3f}     <- 1.0 would reconcile the two estimators")
    resolved = 0.5 <= lr <= 2.0
    out["locality_reading_supported"] = bool(resolved)
    out["halt_H2"] = bool(not resolved)
    verdict = ("LOCALITY READING SUPPORTED -- estimators reconciled, H2 does not fire"
               if resolved else
               "LOCALITY REFUTED -- the estimators disagree on one object; HALT H2")
    print(f"\n  {verdict}")

    del model
    torch.cuda.empty_cache()
    p = HERE.parent / "results" / "t0_3_sphere_locality.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
