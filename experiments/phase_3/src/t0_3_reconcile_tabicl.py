"""T0.3 part 2 -- P1. Reconcile the circulation estimator against Phase 1's Jacobians.

This is the whole point of T0.3: two independent estimators of the same quantity,
on the same contexts, one of which (circulation) needs no derivatives, no
projection, no dither and no probe-amplitude choice. It is also the independent
cross-check Phase 2's E0.4 asked for and could not get without autograd hooks.

WHAT IS COMPARED
----------------
Phase 1 measured Q^T J Q by central differences at t=1e-3 and stored it in
    phase_1/tier0_instrument/exp4_tabicl_reduced.npz   key tabicl_t1e-03__J__<seed>
The comparison quantity is the skew Frobenius norm of that block,
    ||J_red - J_red'||_F .

The loop is run with planes drawn orthogonal to BOTH 1 and the centred y, so it
lives in exactly the subspace Q spans, and the two numbers are estimates of the
same thing.

P1 ACCEPTANCE: agreement within 20%. Falsifier: disagreement > 2x, at which point
one estimator is wrong and finding out which becomes the task (halt H2).

The loop estimate carries Monte Carlo error that scales as 1/(2*sqrt(n_planes))
-- measured in T0.3 part 1 and confirmed against theory -- so a bootstrap CI over
the plane samples is reported beside the point estimate. Comparing a point
estimate to a point estimate without that CI would be the wrong test.

A NOTE ON WHAT AGREEMENT MEANS HERE. TabICL is not affine, so the loop returns
the area-average of the antisymmetric 2-form over the enclosed disc (Stokes)
while the finite difference returns a point value at y. They are the same
quantity only to the extent the field is flat over the disc. A radius sweep is
therefore run on one seed: if the loop value moves with r, the two estimators are
measuring genuinely different things and that must be said rather than averaged
over.
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

import torch                                                          # noqa: E402
from models import load                                               # noqa: E402
from circulation import FROBENIUS_CONST, centred_basis, circulation_asym  # noqa: E402
from experiments.phase_1.core.context import generate_audit_context    # noqa: E402

SEEDS = [42, 100, 200, 300, 400]
N_PLANES = 64
N_LOOP = 32
PH1_NPZ = HERE.parents[1] / "phase_1" / "tier0_instrument" / "exp4_tabicl_reduced.npz"
PH1_KEY = "tabicl_t1e-03__J__{}"


def boot_ci(vals, m_dim, n_boot=4000, seed=0):
    """Bootstrap CI for the Frobenius estimate over the plane samples."""
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    ests = np.sqrt(np.maximum(m_dim * (m_dim - 1)
                              * (v[idx] ** 2).mean(axis=1) / FROBENIUS_CONST, 0.0))
    return float(np.quantile(ests, 0.025)), float(np.quantile(ests, 0.975))


def loop_frobenius(predict, y, n_planes, n_loop, seed, r_mult=1.0):
    n = len(y)
    yb = float(np.mean(y))
    r = r_mult * float(np.linalg.norm(y - yb))
    rng = np.random.default_rng(seed)
    m_dim = n - 2                       # planes are orthogonal to 1 AND to centred y
    vals = np.empty(n_planes)
    for i in range(n_planes):
        a, b = centred_basis(n, rng, exclude=y)
        vals[i] = circulation_asym(predict, yb, r, a, b, N=n_loop)
    est = float(np.sqrt(max(m_dim * (m_dim - 1) * np.mean(vals ** 2)
                            / FROBENIUS_CONST, 0.0)))
    return est, vals, m_dim, r


def main():
    z = np.load(PH1_NPZ)
    model = load("tabicl_v2", task="regression", device="cuda")
    out = {"_config": {"n_planes": N_PLANES, "n_loop": N_LOOP, "seeds": SEEDS,
                       "phase1_npz": str(PH1_NPZ), "phase1_key": PH1_KEY,
                       "frobenius_const": FROBENIUS_CONST,
                       "planes_orthogonal_to": ["1", "y - mean(y)"]},
           "rows": []}

    print("=" * 92)
    print("P1  circulation vs Phase 1 finite differences -- TabICL v2, audit contexts")
    print(f"    {N_PLANES} planes x {N_LOOP} loop points = {N_PLANES*N_LOOP} forward passes per context")
    print("=" * 92)
    print(f"  {'seed':<6}{'FD skew_F':>12}{'loop skew_F':>13}{'ratio':>8}"
          f"{'loop 95% CI':>24}{'sec':>7}")

    for s in SEEDS:
        t0 = time.time()
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        Jred = z[PH1_KEY.format(s)]
        fd = float(np.linalg.norm(Jred - Jred.T, "fro"))

        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            return model.estimator.predict(X)

        est, vals, m_dim, r = loop_frobenius(predict, y, N_PLANES, N_LOOP, seed=s)
        lo, hi = boot_ci(vals, m_dim, seed=s)
        row = {"seed": s, "fd_skew_fro": fd, "loop_skew_fro": est,
               "ratio_loop_over_fd": est / fd, "ci95": [lo, hi],
               "fd_in_ci": bool(lo <= fd <= hi), "r": r, "m_dim": m_dim,
               "rel_diff": abs(est - fd) / fd, "seconds": time.time() - t0,
               "plane_vals_mean_abs": float(np.mean(np.abs(vals)))}
        out["rows"].append(row)
        print(f"  {s:<6}{fd:>12.5f}{est:>13.5f}{row['ratio_loop_over_fd']:>8.3f}"
              f"   [{lo:.5f}, {hi:.5f}]{'*' if row['fd_in_ci'] else ' '}"
              f"{row['seconds']:>7.0f}")

    rels = [r["rel_diff"] for r in out["rows"]]
    ratios = [r["ratio_loop_over_fd"] for r in out["rows"]]
    worst = max(rels)
    within20 = worst <= 0.20
    falsified = max(max(ratios), 1.0 / min(ratios)) > 2.0
    out["worst_rel_diff"] = worst
    out["mean_ratio"] = float(np.mean(ratios))
    out["n_fd_in_ci"] = sum(r["fd_in_ci"] for r in out["rows"])
    out["P1_within_20pct"] = bool(within20)
    out["P1_falsified_2x"] = bool(falsified)
    out["halt_H2"] = bool(falsified)

    print(f"\n  worst relative difference {worst:.3f}   mean ratio {out['mean_ratio']:.3f}"
          f"   FD inside loop CI on {out['n_fd_in_ci']}/5 seeds")
    print(f"  P1 (within 20%): {'PASS' if within20 else 'MISS'}"
          f"    falsifier (>2x): {'FIRED -- HALT H2' if falsified else 'not fired'}")

    # ---- radius sweep on one seed: are the two estimators measuring the same object?
    print("\n" + "=" * 92)
    print("Radius sweep, seed 42 -- loop is an area-average (Stokes); FD is a point value")
    print("=" * 92)
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    Jred = z[PH1_KEY.format(42)]
    fd = float(np.linalg.norm(Jred - Jred.T, "fro"))

    def predict(y_eval):
        model.estimator.fit(X, y_eval)
        return model.estimator.predict(X)

    sweep = []
    for rm in [0.1, 0.3, 1.0, 2.0]:
        est, vals, m_dim, r = loop_frobenius(predict, y, 24, N_LOOP, seed=42, r_mult=rm)
        sweep.append({"r_mult": rm, "r": r, "loop_skew_fro": est,
                      "ratio_to_fd": est / fd})
        print(f"  r = {rm:>4.1f} x ||y-ybar||  ({r:7.3f})   loop {est:.5f}   "
              f"ratio to FD {est/fd:.3f}")
    out["radius_sweep_seed42"] = {"fd_skew_fro": fd, "rows": sweep}
    sr = [q["ratio_to_fd"] for q in sweep]
    out["radius_sweep_spread"] = float(max(sr) - min(sr))
    print(f"  ratio spread across two decades of radius: {out['radius_sweep_spread']:.3f}")

    del model
    torch.cuda.empty_cache()
    p = HERE.parent / "results" / "t0_3_reconcile_tabicl.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
