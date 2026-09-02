"""T1.2 -- the TabICL audit across the six-rung ladder. GPU.

Per context: the A1 / A2 / non-degeneracy block from the reduced Jacobian, the A3
cross-channel regression, and a value-channel row so the structural verdict can
never be confused with a performance verdict (plan section 9).

A1's primary instrument is finite differences, per decision D7 -- circulation is
a SEPARATE registered quantity and runs in `t1_circulation.py` on a subset.

Rungs are run in the order A, D, B, C, E, F so the key contrast (prior-family vs
the existing Phase 1-2 audit family) lands first.

Usage:  python t1_model.py [rung ...]      default: all six, in that order
Writes results/t1_model_<rung>.json and results/t1_model_jacobians_<rung>.npz.
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
from experiments.phase_1.core.metrics import (                          # noqa: E402
    asym, extract_variance, get_Q, negeig, profile_jacobian, reduced_jacobian,
)

ORDER = ["A", "D", "B", "C", "E", "F"]
Q_SEED = 0
T = 1e-3                     # TabICL's Phase 1 amplitude; float32 output
LEVELS = np.linspace(1e-4, 1 - 1e-4, 9999)


def ols(x, y):
    Xd = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(coef[1]), float(coef[0]), (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def one_context(model, X, y, f):
    est = model.estimator
    n = len(y)
    Q = get_Q(y, seed=Q_SEED)

    def predict(y_eval):
        est.fit(X, y_eval)
        return est.predict(X)

    # ---- reduced Jacobian: A1, A2, non-degeneracy
    Jr, _, m_base = reduced_jacobian(predict, y, Q, T)

    # ---- ambient diagonal for A3 and tr J, from the same probe
    Ja = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = T
        Ja[:, i] = (np.asarray(predict(y + e)).ravel()
                    - np.asarray(predict(y - e)).ravel()) / (2 * T)

    # ---- variance channel (A3) and the value channel, one fit
    est.fit(X, y)
    m = np.asarray(est.predict(X)).ravel()
    Qq = np.asarray(est.predict(X, output_type="quantiles", alphas=list(LEVELS)))
    if Qq.ndim == 2 and Qq.shape[0] == len(LEVELS):
        Qq = Qq.T
    s2 = np.asarray(extract_variance(Qq, LEVELS)).ravel()

    Jd = np.diag(Ja)
    slope, icept, r2 = ols(1.0 + Jd, s2)
    sy = float(np.std(y))
    row = {
        # A1 / A2
        "asym": float(asym(Jr)), "negeig": float(negeig(Jr)),
        "ambient_asym": float(asym(Ja)), "ambient_negeig": float(negeig(Ja)),
        "skew_fro": float(np.linalg.norm(Jr - Jr.T, "fro")),
        # non-degeneracy (E2.1)
        "J_minus_I": float(np.linalg.norm(Ja - np.eye(n), "fro")
                           / np.linalg.norm(Ja, "fro")),
        "normF_over_sqrtn": float(np.linalg.norm(Ja, "fro") / np.sqrt(n)),
        "trJ": float(np.trace(Ja)), "trJ_over_n": float(np.trace(Ja) / n),
        "n_neg_diag": int(np.sum(Jd < 0)),
        # A3
        "a3_slope": slope, "a3_intercept": icept, "a3_r2": r2,
        "a3_sigma2_hat": slope / sy ** 2,
        "cv_s": float(np.std(s2) / np.mean(s2)),
        "cv_J": float(np.std(Jd) / abs(np.mean(Jd))),
        # mechanism profiling (ambient, per Phase 1 section 6.1)
        "row_residual": float(profile_jacobian(Ja, "row")["residual_rel"]),
        "col_residual": float(profile_jacobian(Ja, "column")["residual_rel"]),
        # value channel
        "mse_vs_y": float(np.mean((m - y) ** 2)),
        "s_y": sy,
    }
    if f is not None:
        row["mse_vs_f"] = float(np.mean((m - f) ** 2))
        row["shrinkage"] = float(np.mean((m - f) ** 2) / np.mean((y - f) ** 2))
    return row, Jr


def main():
    rungs = sys.argv[1:] or ORDER
    z = np.load(HERE.parent / "results" / "t1_contexts.npz")
    model = load("tabicl_v2", task="regression", device="cuda")

    for rung in rungs:
        t0 = time.time()
        rows, store = [], {}
        k = 0
        print("=" * 92, flush=True)
        print(f"RUNG {rung}", flush=True)
        print("=" * 92, flush=True)
        while f"{rung}__X__{k}" in z:
            X = z[f"{rung}__X__{k}"]; y = z[f"{rung}__y__{k}"]
            fk = f"{rung}__f__{k}"
            f = z[fk] if fk in z else None
            r, Jr = one_context(model, X, y, f)
            r["index"] = k
            rows.append(r)
            store[f"J__{k}"] = Jr.astype(np.float32)
            if k % 10 == 0 or k < 3:
                print(f"  ctx {k:>3}  asym {r['asym']:.4f}  negeig {r['negeig']:.4f}  "
                      f"trJ/n {r['trJ_over_n']:.3f}  ||J-I|| {r['J_minus_I']:.3f}  "
                      f"A3 R2 {r['a3_r2']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            k += 1

        agg = {kk: {"mean": float(np.mean([r[kk] for r in rows])),
                    "sd": float(np.std([r[kk] for r in rows], ddof=1)),
                    "min": float(np.min([r[kk] for r in rows])),
                    "max": float(np.max([r[kk] for r in rows]))}
               for kk in rows[0] if kk != "index" and isinstance(rows[0][kk], (int, float))}
        out = {"_config": {"rung": rung, "t": T, "Q_seed": Q_SEED,
                           "model": "tabicl_v2", "n_contexts": len(rows)},
               "aggregate": agg, "rows": rows}
        json.dump(out, open(HERE.parent / "results" / f"t1_model_{rung}.json", "w"),
                  indent=2, default=float)
        np.savez_compressed(
            HERE.parent / "results" / f"t1_model_jacobians_{rung}.npz", **store)
        print(f"\n  RUNG {rung} DONE  {len(rows)} contexts in {time.time()-t0:.0f}s")
        print(f"    asym      {agg['asym']['mean']:.4f} +- {agg['asym']['sd']:.4f}")
        print(f"    negeig    {agg['negeig']['mean']:.4f} +- {agg['negeig']['sd']:.4f}")
        print(f"    trJ/n     {agg['trJ_over_n']['mean']:.4f} "
              f"[{agg['trJ_over_n']['min']:.3f}, {agg['trJ_over_n']['max']:.3f}]")
        print(f"    ||J-I||   {agg['J_minus_I']['mean']:.4f}")
        print(f"    A3 R2     {agg['a3_r2']['mean']:.4f}")
        print(f"    row resid {agg['row_residual']['mean']:.4f}", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
