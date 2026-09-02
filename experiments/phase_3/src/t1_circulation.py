"""T1 -- sphere-averaged circulation across the ladder. The D7 second quantity.

Decision D7: after the T0.3 reconciliation failed, circulation and finite
differences are kept as SEPARATE registered quantities rather than one replacing
the other. Both were shown correct on wrapped nonlinear maps with known answers;
they measure different things.

    asym_FD   pointwise antisymmetry of Q'JQ at y
    C         sphere-averaged circulation -- the great-circle average of the
              antisymmetric 2-form over the constant-(ybar, r) sphere

and their RATIO C / skew_FD is itself reported, because T0.3 established what it
measures: how fast the antisymmetry DIRECTION decorrelates over the sphere. On
TabICL's audit contexts the antisymmetric parts at different sphere points are
orthogonal to three decimals and the ratio is ~0.15; on a map whose antisymmetric
part is a constant matrix the ratio is ~1. So a rung-by-rung ratio is a
measurement of structure, not a failed reconciliation.

POWER, per decision D8
----------------------
32 planes x 24 loop points = 768 forward passes per context, on N_SUBSET = 20 of
the 60 contexts per rung. Full power on all 360 would be ~51 GPU-hours against a
40 h budget for T0-T2. Monte Carlo SE of the Frobenius estimate is
1/(2*sqrt(32)) ~ 8.8%, measured and confirmed in T0.3 part 1, and a bootstrap CI
over the plane samples is reported with every value.

Usage:  python t1_circulation.py [rung ...]
Writes results/t1_circulation_<rung>.json.
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
from circulation import FROBENIUS_CONST, centred_basis, circulation_asym  # noqa: E402

ORDER = ["A", "D", "B", "C", "E", "F"]
N_SUBSET = 20
N_PLANES = 32
N_LOOP = 24


def boot_ci(vals, m_dim, n_boot=3000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    e = np.sqrt(np.maximum(m_dim * (m_dim - 1) * (v[idx] ** 2).mean(axis=1)
                           / FROBENIUS_CONST, 0.0))
    return float(np.quantile(e, 0.025)), float(np.quantile(e, 0.975))


def main():
    rungs = sys.argv[1:] or ORDER
    z = np.load(HERE.parent / "results" / "t1_contexts.npz")
    model = load("tabicl_v2", task="regression", device="cuda")

    for rung in rungs:
        # FD skew from the ladder, for the ratio
        fdp = HERE.parent / "results" / f"t1_model_{rung}.json"
        fd = {r["index"]: r["skew_fro"] for r in json.load(open(fdp))["rows"]} \
            if fdp.exists() else {}

        t0, rows = time.time(), []
        print("=" * 90, flush=True)
        print(f"CIRCULATION  rung {rung}   {N_SUBSET} contexts x {N_PLANES} planes "
              f"x {N_LOOP} points", flush=True)
        print("=" * 90, flush=True)
        for k in range(N_SUBSET):
            key = f"{rung}__X__{k}"
            if key not in z:
                break
            X, y = z[key], z[f"{rung}__y__{k}"]
            n = len(y)

            def predict(y_eval):
                model.estimator.fit(X, y_eval)
                return model.estimator.predict(X)

            yb = float(np.mean(y))
            r = float(np.linalg.norm(y - yb))
            rng = np.random.default_rng(6000 + k)
            m_dim = n - 2
            vals = np.empty(N_PLANES)
            for i in range(N_PLANES):
                a, b = centred_basis(n, rng, exclude=y)
                vals[i] = circulation_asym(predict, yb, r, a, b, N=N_LOOP)
            C = float(np.sqrt(max(m_dim * (m_dim - 1) * np.mean(vals ** 2)
                                  / FROBENIUS_CONST, 0.0)))
            lo, hi = boot_ci(vals, m_dim, seed=k)
            row = {"index": k, "circulation_fro": C, "ci95": [lo, hi], "r": r}
            if k in fd:
                row["fd_skew_fro"] = fd[k]
                row["ratio_C_over_FD"] = C / fd[k] if fd[k] > 0 else None
            rows.append(row)
            if k % 5 == 0:
                extra = (f"  FD {fd[k]:.4f}  C/FD {C/fd[k]:.3f}"
                         if k in fd and fd[k] > 0 else "")
                print(f"  ctx {k:>3}  C {C:.5f}  CI [{lo:.4f}, {hi:.4f}]"
                      f"{extra}   ({time.time()-t0:.0f}s)", flush=True)

        C = np.array([r["circulation_fro"] for r in rows])
        agg = {"circulation_mean": float(C.mean()),
               "circulation_sd": float(C.std(ddof=1))}
        rr = [r["ratio_C_over_FD"] for r in rows if r.get("ratio_C_over_FD")]
        if rr:
            agg["ratio_C_over_FD_mean"] = float(np.mean(rr))
            agg["ratio_C_over_FD_sd"] = float(np.std(rr, ddof=1))
        out = {"_config": {"rung": rung, "n_subset": len(rows),
                           "n_planes": N_PLANES, "n_loop": N_LOOP,
                           "mc_se_frac": 1 / (2 * np.sqrt(N_PLANES))},
               "aggregate": agg, "rows": rows}
        json.dump(out, open(HERE.parent / "results" / f"t1_circulation_{rung}.json", "w"),
                  indent=2, default=float)
        print(f"\n  RUNG {rung}  C {C.mean():.5f} +- {C.std(ddof=1):.5f}"
              + (f"   C/FD {agg['ratio_C_over_FD_mean']:.3f} "
                 f"+- {agg['ratio_C_over_FD_sd']:.3f}" if rr else "")
              + f"   ({time.time()-t0:.0f}s)", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
