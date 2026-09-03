"""T0.3 -- does the antisymmetric part CHANGE DIRECTION across the sphere?

RECONSTRUCTED 2026-09-03. This measurement was originally run as an inline
heredoc and its script was never saved -- exactly the failure mode Phase 2 was
cleaned up for. The T0.3 verdict rests on it, so it is on disk now and the
verdict must be re-cited from this file, not from the transcript.

WHAT IT SETTLES
---------------
P1 found loop/FD = 0.166 on TabICL. Three explanations were eliminated:
  D1/D2  both estimators are correct on wrapped NONLINEAR maps with known answers
  D3     loop planes CONTAINING y do not close the gap
  sphere FD-at-y is representative of the sphere (sphere/at-y = 1.096), so
         locality of MAGNITUDE is refuted

That leaves one possibility: the magnitude is uniform but the DIRECTION varies,
so a great-circle average cancels while a point value does not.

THE MEASUREMENT, AND THE ONE THING THAT MAKES IT VALID
------------------------------------------------------
Reduced Jacobians at y and at K random points of the same constant-(ybar, r)
sphere, all projected through ONE FIXED BASIS Q -- Q built from y and reused at
every point. This is the crux: if Q were rebuilt per point (as `skew_at` in
t0_3_sphere_locality.py does, correctly, for a different purpose) the matrices
would live in different coordinate systems and their alignment would be
meaningless. Here we want to compare directions, so the frame must be shared.

Reported: ||A_i||_F per point, and the pairwise alignment
    <A_i, A_j> / (||A_i|| ||A_j||)
Near-zero off-diagonal alignment with comparable magnitudes is the signature that
explains the loop/FD gap; alignment near 1 would refute it.

Writes results/t0_3_direction_decorrelation.json.
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

SEEDS = [42, 100]
K_SPHERE = 3
T = 1e-3
ALIGN_GATE = 0.35        # below this, directions are decorrelated


def main():
    model = load("tabicl_v2", task="regression", device="cuda")
    out = {"_config": {"seeds": SEEDS, "k_sphere": K_SPHERE, "t": T,
                       "fixed_basis": "Q = get_Q(y, seed=0), reused at every point",
                       "align_gate": ALIGN_GATE},
           "rows": []}

    print("=" * 88)
    print("T0.3  direction decorrelation of the antisymmetric part across the sphere")
    print("      ONE fixed basis Q so the matrices are comparable")
    print("=" * 88)

    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        n = len(y)
        yb, r = float(np.mean(y)), float(np.linalg.norm(y - np.mean(y)))
        Q = get_Q(y, seed=0)                       # THE fixed frame

        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            return model.estimator.predict(X)

        rng = np.random.default_rng(11 + s)
        pts = [y]
        for _ in range(K_SPHERE):
            v = rng.standard_normal(n); v -= v.mean()
            v *= r / np.linalg.norm(v)
            pts.append(yb + v)

        print(f"\n  seed {s}")
        As, norms = [], []
        for i, pt in enumerate(pts):
            t0 = time.time()
            J, _, _ = reduced_jacobian(predict, pt, Q, T)     # same Q every time
            A = J - J.T
            As.append(A)
            norms.append(float(np.linalg.norm(A, "fro")))
            print(f"    point {i} ({'y' if i == 0 else 'sphere'}):  "
                  f"||A||_F = {norms[-1]:.4f}   ({time.time()-t0:.0f}s)", flush=True)

        M = np.zeros((len(As), len(As)))
        for i in range(len(As)):
            for j in range(len(As)):
                M[i, j] = float(np.sum(As[i] * As[j])
                                / (np.linalg.norm(As[i], "fro")
                                   * np.linalg.norm(As[j], "fro")))
        off = [M[i, j] for i in range(len(As)) for j in range(i + 1, len(As))]
        print("    alignment matrix:")
        for i in range(len(As)):
            print("      " + "".join(f"{M[i, j]:>9.3f}" for j in range(len(As))))
        print(f"    mean off-diagonal alignment = {np.mean(off):+.4f}")
        out["rows"].append({"seed": s, "normF": norms,
                            "alignment_matrix": M.tolist(),
                            "off_diagonal": [float(x) for x in off],
                            "mean_off_diagonal": float(np.mean(off)),
                            "max_abs_off_diagonal": float(np.max(np.abs(off)))})

    allo = [x for q in out["rows"] for x in q["off_diagonal"]]
    mean_all = float(np.mean(allo))
    max_all = float(np.max(np.abs(allo)))
    decorr = max_all < ALIGN_GATE
    out["mean_off_diagonal_all"] = mean_all
    out["max_abs_off_diagonal_all"] = max_all
    out["directions_decorrelated"] = bool(decorr)

    print("\n" + "=" * 88)
    print(f"  pooled mean off-diagonal alignment {mean_all:+.4f}   "
          f"max |alignment| {max_all:.4f}   (gate {ALIGN_GATE})")
    verdict = ("DIRECTIONS DECORRELATED -- cancellation explains the loop/FD gap"
               if decorr else
               "directions are STABLE -- cancellation does NOT explain it")
    print(f"  -> {verdict}")

    del model
    torch.cuda.empty_cache()
    p = HERE.parent / "results" / "t0_3_direction_decorrelation.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
