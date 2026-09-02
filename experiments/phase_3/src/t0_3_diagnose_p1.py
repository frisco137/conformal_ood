"""P1 fired its falsifier. Which estimator is wrong?

P1 measured, on TabICL's audit contexts:
    FD skew ||J-J'||_F  ~ 4.67      loop skew ~ 0.63      ratio ~ 0.166

Halt condition H2 fires only if *neither estimator can be shown correct*. This
script tries to show one of them correct, on a map where the answer is known.

THE HYPOTHESIS
--------------
The loop and the finite difference are evaluated at DIFFERENT PLACES.

  FD    is a point value at y.
  loop  integrates over a circle of radius r about ybar*1, inside the plane
        span{a,b}. Because P1 draws a, b orthogonal to the centred y (so the
        loop is blind to the T3.1 noise-scale contamination), THE LOOP'S PLANE
        DOES NOT CONTAIN y. Every point on it is at distance sqrt(2)*r from y.

By Stokes the loop returns the area-average of the antisymmetric 2-form over
that disc. For an affine map that equals the point value -- verified to 5e-15 in
T0.3 part 1. For a nonlinear map it need not, and TabICL is not affine.

The radius sweep already showed the loop value is EXACTLY r-independent
(0.59793 at r = 1.4, 4.2, 14.0, 28.0). That is not a bug and not evidence of
linearity: for a normaliser-wrapped map m(y) = ybar*1 + s_y*g(u), a constant-ybar
constant-r loop sends u around the SAME circle of radius sqrt(n) whatever r is,
and m scales linearly in r, so I/(pi r^2) is structurally r-invariant. It does
mean the radius knob cannot probe locality, so this script uses the plane instead.

THE THREE TESTS
---------------
  D1  Wrapped TargetedImitator -- a map with a KNOWN analytic inner Jacobian and
      genuine asymmetry, wearing the same normaliser TabICL wears. If the loop
      recovers its skew Frobenius, the instrument is correct on a nonlinear
      wrapped map and the TabICL gap is a property of TabICL.
  D2  Same, for a wrapped nonlinear GP-plus-cubic map with no closed form, loop
      vs FD. Isolates 'nonlinear' from 'constructed'.
  D3  TabICL with planes CONTAINING the centred y, so the loop passes through y
      itself. If the ratio moves toward 1, the gap is geometric -- the model's
      asymmetry is concentrated near the data configuration and the two
      estimators are measuring different regions, not disagreeing about one.

Writes results/t0_3_diagnose_p1.json.
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

from circulation import FROBENIUS_CONST, centred_basis, circulation_asym  # noqa: E402
from experiments.phase_1.core.context import generate_audit_context     # noqa: E402
from experiments.phase_1.core.controls import (                          # noqa: E402
    GP_SIGMA, TargetedImitator, wrap,
)
from experiments.phase_1.core.metrics import central_jacobian, get_Q     # noqa: E402
from experiments.phase_1.core.surrogates import ExactGP                  # noqa: E402

SEEDS = [42, 100, 200]
N_PLANES = 96
N_LOOP = 32


def loop_fro(predict, y, n_planes, seed, exclude_y=True, along_y=False, r_mult=1.0):
    n = len(y)
    yb = float(np.mean(y))
    r = r_mult * float(np.linalg.norm(y - yb))
    rng = np.random.default_rng(seed)
    yc = y - yb
    yhat = yc / np.linalg.norm(yc)
    vals = []
    if along_y:
        # planes CONTAINING the centred y: a := yhat, b _|_ {1, y}
        m_dim = n - 1
        for _ in range(n_planes):
            _a, b = centred_basis(n, rng, exclude=y)
            vals.append(circulation_asym(predict, yb, r, yhat, b, N=N_LOOP))
    else:
        m_dim = n - 2 if exclude_y else n - 1
        for _ in range(n_planes):
            a, b = centred_basis(n, rng, exclude=y if exclude_y else None)
            vals.append(circulation_asym(predict, yb, r, a, b, N=N_LOOP))
    vals = np.asarray(vals)
    est = float(np.sqrt(max(m_dim * (m_dim - 1) * np.mean(vals ** 2)
                            / FROBENIUS_CONST, 0.0)))
    return est, vals, m_dim


def fd_skew_in_subspace(J, y, exclude_y=True):
    """||J - J'||_F restricted to the subspace the loop planes live in."""
    n = len(y)
    B = [np.ones(n)] + ([y - y.mean()] if exclude_y else [])
    Qb, _ = np.linalg.qr(np.column_stack(B))
    P = np.eye(n) - Qb @ Qb.T
    return float(np.linalg.norm(P @ (J - J.T) @ P, "fro"))


def main():
    out = {"_config": {"n_planes": N_PLANES, "n_loop": N_LOOP, "seeds": SEEDS}}

    # =================================================================== D1, D2
    print("=" * 96)
    print("D1/D2  does the loop recover the skew of a WRAPPED NONLINEAR map with a known answer?")
    print("=" * 96)
    print(f"  {'map':<26}{'seed':<6}{'FD skew_F':>12}{'loop skew_F':>13}"
          f"{'ratio':>8}{'analytic':>12}")
    rows = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        u = (y - np.mean(y)) / (np.std(y) + 1e-8)

        # --- D1: wrapped targeted imitator, closed form available
        ti = TargetedImitator(X, y, seed=s)
        G = ti.inner_jacobian(u)
        ana = fd_skew_in_subspace(G, y, exclude_y=True)
        Jfd = central_jacobian(ti.predict, y, h=1e-4)
        fd = fd_skew_in_subspace(Jfd, y, exclude_y=True)
        est, vals, md = loop_fro(ti.predict, y, N_PLANES, seed=s)
        rows.append({"map": "wrapped_targeted_imitator", "seed": s, "fd": fd,
                     "loop": est, "ratio": est / fd, "analytic": ana})
        print(f"  {'wrapped TargetedImitator':<26}{s:<6}{fd:>12.5f}{est:>13.5f}"
              f"{est/fd:>8.3f}{ana:>12.5f}")

        # --- D2: wrapped nonlinear map, no closed form
        W = ExactGP(X, sigma=GP_SIGMA).W
        rng = np.random.default_rng(s)
        M = rng.standard_normal((100, 100)); M = (M - M.T) / 2
        M *= 0.3 * np.linalg.norm(W, "fro") / np.linalg.norm(M, "fro")

        def inner(uu, W=W, M=M):
            v = W @ uu + M @ uu
            return v + 0.15 * np.tanh(v) ** 3          # genuinely nonlinear
        mp = wrap(inner)
        Jfd2 = central_jacobian(mp, y, h=1e-4)
        fd2 = fd_skew_in_subspace(Jfd2, y, exclude_y=True)
        est2, _, _ = loop_fro(mp, y, N_PLANES, seed=s)
        rows.append({"map": "wrapped_nonlinear", "seed": s, "fd": fd2,
                     "loop": est2, "ratio": est2 / fd2, "analytic": None})
        print(f"  {'wrapped nonlinear (tanh^3)':<26}{s:<6}{fd2:>12.5f}{est2:>13.5f}"
              f"{est2/fd2:>8.3f}{'-':>12}")
    out["D1_D2"] = rows

    for nm in ("wrapped_targeted_imitator", "wrapped_nonlinear"):
        rs = [r for r in rows if r["map"] == nm]
        worst = max(abs(r["ratio"] - 1) for r in rs)
        print(f"  -> {nm:<28} worst |ratio-1| = {worst:.3f}  "
              f"{'INSTRUMENT OK' if worst <= 0.25 else '*** INSTRUMENT SUSPECT ***'}")
        out[f"verdict_{nm}"] = {"worst_abs_ratio_minus_1": worst,
                                "instrument_ok": bool(worst <= 0.25)}

    # ======================================================================= D3
    print("\n" + "=" * 96)
    print("D3  TabICL: does the gap close when the loop plane CONTAINS y?")
    print("=" * 96)
    from models import load                                              # noqa: E402
    import torch                                                         # noqa: E402
    model = load("tabicl_v2", task="regression", device="cuda")
    z = np.load(HERE.parents[1] / "phase_1" / "tier0_instrument" / "exp4_tabicl_reduced.npz")
    d3 = []
    print(f"  {'seed':<6}{'FD skew(perp)':>15}{'loop perp':>11}{'ratio':>8}"
          f"{'FD skew(incl y)':>17}{'loop along y':>14}{'ratio':>8}")
    for s in SEEDS:
        t0 = time.time()
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)

        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            return model.estimator.predict(X)

        Jred = z[f"tabicl_t1e-03__J__{s}"]
        fd_perp = float(np.linalg.norm(Jred - Jred.T, "fro"))
        loop_perp, _, _ = loop_fro(predict, y, 48, seed=s, exclude_y=True)
        # planes containing y -- compare against the skew on span{1}^perp
        Jamb = np.load(HERE.parents[1] / "phase_1" / "tier0_instrument"
                       / "exp12_ambient_jacobians.npz")[f"tabicl_v2__J__{s}"]
        fd_incl = fd_skew_in_subspace(Jamb, y, exclude_y=False)
        loop_y, _, _ = loop_fro(predict, y, 48, seed=s, along_y=True)
        d3.append({"seed": s, "fd_perp": fd_perp, "loop_perp": loop_perp,
                   "ratio_perp": loop_perp / fd_perp, "fd_incl_y": fd_incl,
                   "loop_along_y": loop_y, "ratio_along_y": loop_y / fd_incl,
                   "seconds": time.time() - t0})
        print(f"  {s:<6}{fd_perp:>15.5f}{loop_perp:>11.5f}{loop_perp/fd_perp:>8.3f}"
              f"{fd_incl:>17.5f}{loop_y:>14.5f}{loop_y/fd_incl:>8.3f}")
    out["D3"] = d3
    del model
    torch.cuda.empty_cache()

    p = HERE.parent / "results" / "t0_3_diagnose_p1.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
