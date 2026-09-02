"""T0.3 part 1 -- circulation instrument, analytic controls. P2, P3, P4. CPU only.

Validates the instrument on maps whose answer is known before it is pointed at a
model. Part 2 (`t0_3_reconcile_tabicl.py`) does the P1 reconciliation against the
stored Phase 1 Jacobians, which needs a GPU.

  IDENTITY  a'(J-J')b recovered exactly on an affine map, and the SIGN convention
            pinned. Not a registered prediction, but if this fails nothing else
            in T0.3 means anything.
  P2        Exact GP BEHIND THE NORMALISER, no projection: |I|/(pi r^2) <= 1e-8.
            This is the load-bearing claim -- it is what lets the loop drop the
            projection entirely.
  P3        Loop estimate changes <= 5% as an artificial output quantum sweeps
            1e-5 -> 1e-2, in a regime where central differences collapse.
  P4        Frobenius recovery from 40 random planes within 20% of the true
            centred ||J-J'||_F on analytic maps.

Writes results/t0_3_controls.json.
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

from circulation import (                                            # noqa: E402
    FROBENIUS_CONST, centred_basis, circulation_asym, frobenius_from_planes,
)
from experiments.phase_1.core.context import generate_audit_context  # noqa: E402
from experiments.phase_1.core.controls import GP_SIGMA, wrap         # noqa: E402
from experiments.phase_1.core.metrics import central_jacobian, get_Q  # noqa: E402
from experiments.phase_1.core.surrogates import (                     # noqa: E402
    ExactGP, HierarchicalGP, NadarayaWatson,
)

SEEDS = [42, 100, 200, 300, 400]
N_LOOP = 64
N_PLANES = 40
R_FRAC = 1.0          # loop radius as a multiple of ||y - ybar||


def quantize(f, delta):
    return lambda y: np.round(np.asarray(f(y)).ravel() / delta) * delta


def centred_skew_fro(J, n, exclude_y=None):
    """||J - J'||_F restricted to the subspace the loops actually live in."""
    B = [np.ones(n)]
    if exclude_y is not None:
        e = exclude_y - exclude_y.mean()
        B.append(e)
    Qb, _ = np.linalg.qr(np.column_stack(B))
    P = np.eye(n) - Qb @ Qb.T
    A = P @ (J - J.T) @ P
    return float(np.linalg.norm(A, "fro"))


def main():
    out = {"_config": {"N_loop": N_LOOP, "n_planes": N_PLANES, "seeds": SEEDS,
                       "r_frac": R_FRAC, "frobenius_const": FROBENIUS_CONST,
                       "gp_sigma": GP_SIGMA}}
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- IDENTITY
    print("=" * 78)
    print("IDENTITY  loop recovers a'(J-J')b on an affine map, sign pinned")
    print("=" * 78)
    n = 60
    Jr = rng.standard_normal((n, n)) / np.sqrt(n)
    pred_aff = lambda y: Jr @ y + 3.7                              # noqa: E731
    rows = []
    for k in range(5):
        a, b = centred_basis(n, rng)
        ana = float(a @ (Jr - Jr.T) @ b)
        loop = circulation_asym(pred_aff, 0.4, 1.3, a, b, N=N_LOOP)
        rows.append({"analytic": ana, "loop": loop,
                     "rel_err": abs(loop - ana) / abs(ana)})
        print(f"  plane {k}  analytic {ana:+.12f}  loop {loop:+.12f}  "
              f"rel {rows[-1]['rel_err']:.2e}")
    out["identity"] = {"rows": rows,
                       "max_rel_err": max(r["rel_err"] for r in rows),
                       "PASS": max(r["rel_err"] for r in rows) < 1e-10}
    print(f"  -> max rel err {out['identity']['max_rel_err']:.2e}   "
          f"{'PASS' if out['identity']['PASS'] else 'FAIL'}")

    # ---------------------------------------------------------------------- P2
    print("\n" + "=" * 78)
    print("P2  exact GP BEHIND THE NORMALISER, no projection: |I|/(pi r^2) <= 1e-8")
    print("=" * 78)
    p2 = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        egp = ExactGP(X, sigma=GP_SIGMA)
        m_wrapped = wrap(egp.predict)
        yb = float(np.mean(y))
        r = R_FRAC * float(np.linalg.norm(y - yb))
        vals = []
        for _ in range(8):
            a, b = centred_basis(100, rng)
            vals.append(abs(circulation_asym(m_wrapped, yb, r, a, b, N=N_LOOP)))
        p2.append({"seed": s, "max_abs": float(np.max(vals)),
                   "mean_abs": float(np.mean(vals)), "r": r})
        print(f"  seed {s:<5} max |I|/(pi r^2) over 8 planes = {p2[-1]['max_abs']:.3e}")
    worst2 = max(r["max_abs"] for r in p2)
    out["P2"] = {"rows": p2, "worst": worst2, "gate": 1e-8, "PASS": worst2 <= 1e-8}
    print(f"  -> worst {worst2:.3e}  gate 1e-8   "
          f"{'PASS' if out['P2']['PASS'] else '*** FAIL ***'}")

    # ---------------------------------------------------------------------- P3
    print("\n" + "=" * 78)
    print("P3  quantisation immunity: loop vs central differences under a sweep")
    print("=" * 78)
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    egp = ExactGP(X, sigma=GP_SIGMA)
    # a map with genuine, known asymmetry so there is something to degrade
    W = egp.W
    M = rng.standard_normal((100, 100))
    M = (M - M.T) / 2
    M *= 0.3 * np.linalg.norm(W, "fro") / np.linalg.norm(M, "fro")
    Jtrue = W + M
    base = lambda yy: Jtrue @ yy                                   # noqa: E731
    yb = float(np.mean(y))
    r = R_FRAC * float(np.linalg.norm(y - yb))
    a, b = centred_basis(100, rng)
    truth = float(a @ (Jtrue - Jtrue.T) @ b)

    p3 = []
    for delta in [0.0, 1e-5, 1e-4, 1e-3, 1e-2]:
        f = base if delta == 0 else quantize(base, delta)
        loop = circulation_asym(f, yb, r, a, b, N=N_LOOP)
        Jfd = central_jacobian(f, y, h=1e-3)
        fd = float(a @ (Jfd - Jfd.T) @ b)
        p3.append({"delta": delta, "loop": loop, "fd": fd,
                   "loop_rel_err": abs(loop - truth) / abs(truth),
                   "fd_rel_err": abs(fd - truth) / abs(truth),
                   "fd_exactly_zero": bool(fd == 0.0)})
        print(f"  delta {delta:<8.0e} loop {loop:+.6f} (rel {p3[-1]['loop_rel_err']:.2e})   "
              f"FD {fd:+.6f} (rel {p3[-1]['fd_rel_err']:.2e})"
              f"{'  <- FD exactly 0' if fd == 0.0 else ''}")
    swept = [q for q in p3 if q["delta"] > 0]
    loop_drift = max(q["loop_rel_err"] for q in swept)
    out["P3"] = {"truth": truth, "rows": p3, "loop_max_rel_err": loop_drift,
                 "gate": 0.05, "PASS": loop_drift <= 0.05,
                 "fd_collapsed": any(q["fd_exactly_zero"] for q in swept)}
    print(f"  -> loop worst deviation {loop_drift:.3e}  gate 5e-2   "
          f"{'PASS' if out['P3']['PASS'] else '*** FAIL ***'}"
          f"   FD collapsed to exactly 0: {out['P3']['fd_collapsed']}")

    # ---------------------------------------------------------------------- P4
    print("\n" + "=" * 78)
    print(f"P4  Frobenius recovery from {N_PLANES} random planes, analytic maps")
    print("=" * 78)
    p4 = []
    for name, mk in [("exactgp_wrapped", lambda X: (wrap(ExactGP(X, sigma=GP_SIGMA).predict),
                                                    ExactGP(X, sigma=GP_SIGMA).W)),
                     ("nadaraya_watson", lambda X: (NadarayaWatson(X).predict,
                                                    NadarayaWatson(X).W)),
                     ("asym_affine", lambda X: (lambda yy: Jtrue @ yy, Jtrue))]:
        for s in SEEDS[:3]:
            X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
            f, J = mk(X)
            yb = float(np.mean(y))
            r = R_FRAC * float(np.linalg.norm(y - yb))
            est, vals, m_dim = frobenius_from_planes(
                f, yb, r, 100, n_planes=N_PLANES, N=N_LOOP, seed=s,
                return_samples=True)
            true = centred_skew_fro(J, 100)
            rel = abs(est - true) / true if true > 1e-12 else (0.0 if est < 1e-10 else np.inf)
            p4.append({"map": name, "seed": s, "est": est, "true": true,
                       "rel_err": float(rel), "m_dim": m_dim})
            print(f"  {name:<18} seed {s:<5} est {est:.6f}  true {true:.6f}  "
                  f"rel {rel:.3f}")
    finite = [q for q in p4 if np.isfinite(q["rel_err"]) and q["true"] > 1e-12]
    worst4 = max(q["rel_err"] for q in finite)
    out["P4"] = {"rows": p4, "worst_rel_err": worst4, "gate": 0.20,
                 "PASS": worst4 <= 0.20, "n_planes": N_PLANES}
    print(f"  -> worst relative error {worst4:.3f}  gate 0.20   "
          f"{'PASS' if out['P4']['PASS'] else '*** FAIL ***'}")

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 78)
    for k in ("identity", "P2", "P3", "P4"):
        print(f"  {k:<10} {'PASS' if out[k]['PASS'] else '*** FAIL ***'}")
    print("=" * 78)

    p = HERE.parent / "results" / "t0_3_controls.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
