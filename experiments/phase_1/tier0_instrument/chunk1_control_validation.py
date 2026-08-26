"""Chunk 1 — fix the metric definitions, recompute every control through them.

Controls only. No model calls. Everything here is pure NumPy and regenerates
from scratch, which is necessary because no Jacobian was ever persisted to disk
(see CONTROL_VALIDATION.md, 0.4).

Standing configuration, stated once:
    context   generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
    seeds     [42, 100, 200, 300, 400]
    basis     core.metrics.get_Q(y, seed=0)  -- seeded, shape (100, 98)
    ExactGP   sigma=0.5, lengthscale=1.0
    HierGP    sigma=0.5, 8 lengthscales logspace(0.2, 5.0)
    delta     6.87e-4  (TabPFN's measured median output jump)
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.surrogates import ExactGP, HierarchicalGP
from experiments.phase_1.core.metrics import (
    asym, negeig, negfrac, get_Q, reduced_jacobian, profile_jacobian,
    second_difference_norm,
)
# The control maps used to be defined in this file. They now live in
# experiments/core/controls.py as the single definition -- TargetedImitator's
# closed-form Jacobian had been wrong here (-c v v^T for a map applying
# -(c/2) v v^T) and was worked around inline in three other scripts. Re-exported
# below so that `from chunk1_control_validation import wrap, TargetedImitator,
# GP_SIGMA` keeps working for every existing caller.
from experiments.phase_1.core.controls import (           # noqa: F401  (re-export)
    GP_SIGMA, DELTA, wrap, quantize, ImitatorWrapped, TargetedImitator,
)

SEEDS = [42, 100, 200, 300, 400]
Q_SEED = 0


# ---------------------------------------------------------------- control maps
#
# wrap / quantize / ImitatorWrapped / TargetedImitator moved to
# experiments/core/controls.py and are re-exported at the top of this file.
# Only the local quadratic calibration map remains here.

def quadratic(y_eval):
    """m(y) = y + 0.1 y^2, elementwise. Known second derivative."""
    y_eval = np.asarray(y_eval).flatten()
    return y_eval + 0.1 * y_eval**2


# ------------------------------------------------------------------- reporting

def metrics_of(J):
    return {"asym": float(asym(J)), "negeig": float(negeig(J)),
            "negfrac": float(negfrac(J)),
            "norm_F": float(np.linalg.norm(J, 'fro')),
            "norm_2": float(np.linalg.norm(J, 2))}


def summarise(name, per_seed, keys):
    print(f"\n  {name}")
    for k in keys:
        v = np.array([d[k] for d in per_seed], dtype=float)
        vals = "  ".join(f"{x:.6e}" for x in v)
        print(f"    {k:<9} [{vals}]   mean {v.mean():.6e}  std {v.std():.6e}")


# ------------------------------------------------------------------------ 1A

def run_1a():
    print("=" * 78)
    print("1A  negeig / asym / negfrac for every control, canonical definitions")
    print("=" * 78)

    out = {}
    configs = [
        ("exactgp_unwrapped",          1e-3, False, None),
        ("exactgp_wrapped",            1e-3, False, None),
        ("imitator_wrapped",           1e-3, False, None),
        ("targeted_imitator",          1e-3, False, None),
        ("quantized_exactgp_nodither", 1e-2, False, None),
        ("quantized_exactgp_dither_halfdelta", 1e-2, True, DELTA / 2),
        ("quantized_exactgp_dither_fulldelta", 1e-2, True, DELTA),
    ]

    for name, t, dither, hw in configs:
        per_seed, per_seed_analytic = [], []
        for s in SEEDS:
            X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
            Q = get_Q(y, seed=Q_SEED)
            egp = ExactGP(X, sigma=GP_SIGMA)

            analytic_G = None
            if name == "exactgp_unwrapped":
                predict, analytic_G = egp.predict, egp.jacobian(y)
            elif name == "exactgp_wrapped":
                predict, analytic_G = wrap(egp.predict), egp.jacobian(y)
            elif name == "imitator_wrapped":
                m = ImitatorWrapped(X, y)
                u = (y - np.mean(y)) / (np.std(y) + 1e-8)
                predict, analytic_G = m.predict, m.inner_jacobian(u)
            elif name == "targeted_imitator":
                m = TargetedImitator(X, y, seed=s)
                u = (y - np.mean(y)) / (np.std(y) + 1e-8)
                predict, analytic_G = m.predict, m.inner_jacobian(u)
            else:
                predict = quantize(wrap(egp.predict), DELTA)

            J, _, _ = reduced_jacobian(predict, y, Q, t, dither=dither, N=10,
                                       dither_halfwidth=hw, seed=s)
            per_seed.append(metrics_of(J))
            if analytic_G is not None:
                # N1(ii): Q^T J_wrapped Q = Q^T G Q exactly.
                per_seed_analytic.append(metrics_of(Q.T @ analytic_G @ Q))

        cfg = f"t={t:.0e}, dither={dither}" + (f", halfwidth={hw:.3e}, N=10" if dither else "")
        summarise(f"{name}   [{cfg}]  MEASURED", per_seed,
                  ["asym", "negeig", "negfrac", "norm_F", "norm_2"])
        out[name] = {"config": cfg, "measured": per_seed}
        if per_seed_analytic:
            summarise(f"{name}   ANALYTIC (Q^T G Q)", per_seed_analytic,
                      ["asym", "negeig", "negfrac", "norm_F"])
            out[name]["analytic"] = per_seed_analytic
    return out


# ------------------------------------------------------------------------ 1B

def curvature_of(predict, y, Q, t, dither=False, hw=None, seed=0):
    _, curvs, _ = reduced_jacobian(predict, y, Q, t, dither=dither, N=10,
                                   dither_halfwidth=hw, seed=seed)
    return float(np.mean(curvs))


def run_1b():
    print("\n" + "=" * 78)
    print("1B  true curvature: ||m+ - 2m0 + m-|| / (t^2 ||m0||)")
    print("=" * 78)

    maps = ["exactgp_unwrapped", "exactgp_wrapped",
            "hiergp_unwrapped", "hiergp_wrapped"]
    ts = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
    out = {"sweep": {m: {} for m in maps}, "per_seed": {}}

    # cache the surrogates per seed; HierGP construction is 8 x inv(100x100)
    cache = {}
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        cache[s] = (y, get_Q(y, seed=Q_SEED),
                    ExactGP(X, sigma=GP_SIGMA), HierarchicalGP(X, sigma=GP_SIGMA))

    def predict_for(name, egp, hgp):
        return {"exactgp_unwrapped": egp.predict,
                "exactgp_wrapped": wrap(egp.predict),
                "hiergp_unwrapped": hgp.predict,
                "hiergp_wrapped": wrap(hgp.predict)}[name]

    print("\n  1B.1 / 1B.2  amplitude sweep, per-seed, mean over the 98 probes")
    for name in maps:
        print(f"\n  {name}")
        for t in ts:
            vals = [curvature_of(predict_for(name, *cache[s][2:]),
                                 cache[s][0], cache[s][1], t, seed=s) for s in SEEDS]
            v = np.array(vals)
            print(f"    t={t:<6.0e} [" + "  ".join(f"{x:.6e}" for x in v) +
                  f"]   mean {v.mean():.6e}  std {v.std():.6e}")
            out["sweep"][name][f"{t:.0e}"] = vals

    print("\n  1B.2  HierarchicalGP / ExactGP curvature ratio at matched t")
    ratios = {}
    for t in ts:
        k = f"{t:.0e}"
        for tag, hg, eg in [("unwrapped", "hiergp_unwrapped", "exactgp_unwrapped"),
                            ("wrapped", "hiergp_wrapped", "exactgp_wrapped")]:
            h = np.mean(out["sweep"][hg][k])
            e = np.mean(out["sweep"][eg][k])
            r = h / e if e > 0 else float("inf")
            ratios.setdefault(tag, {})[k] = r
            print(f"    t={t:<6.0e} {tag:<10} hier {h:.4e} / exact {e:.4e} = {r:.4e}")
    out["ratios"] = ratios

    # ---- 1B.4 analytic calibration on m(y) = y + 0.1 y^2 -------------------
    print("\n  1B.4  analytic calibration, m(y) = y + 0.1 y^2 elementwise")
    print("        second difference is exactly 0.2 t^2 q*q, so the true")
    print("        curvature ||0.2 q*q|| / ||m(y)|| is independent of t.")
    cal = {}
    for t in ts:
        errs, meas_all, ana_all = [], [], []
        for s in SEEDS:
            y, Q = cache[s][0], cache[s][1]
            m0 = quadratic(y)
            norm_m0 = np.linalg.norm(m0)
            meas, ana = [], []
            for j in range(Q.shape[1]):
                q = Q[:, j]
                meas.append(second_difference_norm(
                    quadratic(y + t * q), m0, quadratic(y - t * q), t))
                ana.append(np.linalg.norm(0.2 * q * q) / norm_m0)
            meas, ana = np.array(meas), np.array(ana)
            errs.append(np.max(np.abs(meas - ana) / ana))
            meas_all.append(meas.mean())
            ana_all.append(ana.mean())
        cal[f"{t:.0e}"] = {"measured_mean": meas_all, "analytic_mean": ana_all,
                           "max_rel_err": errs}
        print(f"    t={t:<6.0e} measured {np.mean(meas_all):.8e}  "
              f"analytic {np.mean(ana_all):.8e}  "
              f"max rel err over 98 probes x 5 seeds {max(errs):.3e}")
    out["calibration_quadratic"] = cal

    # Jacobian instrument on the same known nonlinear map: J = diag(1 + 0.2 y)
    print("\n        Jacobian check on the same map, analytic J = diag(1 + 0.2 y)")
    jerr = {}
    for t in [1e-3, 1e-2, 1e-1]:
        e = []
        for s in SEEDS:
            y, Q = cache[s][0], cache[s][1]
            J_meas, _, _ = reduced_jacobian(quadratic, y, Q, t)
            J_ana = Q.T @ np.diag(1 + 0.2 * y) @ Q
            e.append(float(np.linalg.norm(J_meas - J_ana, 'fro')
                           / np.linalg.norm(J_ana, 'fro')))
        jerr[f"{t:.0e}"] = e
        print(f"    t={t:<6.0e} relative Frobenius error [" +
              "  ".join(f"{x:.3e}" for x in e) + "]")
    out["calibration_quadratic_jacobian"] = jerr

    # ---- 1B.3 drift diagnostic, reported under its own name ---------------
    print("\n  1B.3  normaliser drift diagnostic (NOT curvature):")
    print("        (s_y(y+tq) - 2 s_y(y) + s_y(y-tq)) / (t^2 s_y(y))")
    drift = {}
    for t in [1e-3, 1e-2, 1e-1]:
        vals = []
        for s in SEEDS:
            y, Q = cache[s][0], cache[s][1]
            sb = np.std(y)
            d = [abs((np.std(y + t * Q[:, j]) - 2 * sb + np.std(y - t * Q[:, j]))
                     / (t**2 * sb)) for j in range(Q.shape[1])]
            vals.append(float(np.mean(d)))
        drift[f"{t:.0e}"] = vals
        print(f"    t={t:<6.0e} [" + "  ".join(f"{x:.6e}" for x in vals) + "]")
    out["drift_diagnostic"] = drift
    return out


# ------------------------------------------------------------- profile residual

def run_profile_residual():
    print("\n" + "=" * 78)
    print("Profile residual, now actually computed (item3.py:95 always gave 0.0)")
    print("=" * 78)
    out = {}
    for name in ["exactgp_wrapped", "targeted_imitator"]:
        rows = []
        for s in SEEDS:
            X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
            Q = get_Q(y, seed=Q_SEED)
            egp = ExactGP(X, sigma=GP_SIGMA)
            predict = (wrap(egp.predict) if name == "exactgp_wrapped"
                       else TargetedImitator(X, y, seed=s).predict)
            J, _, _ = reduced_jacobian(predict, y, Q, 1e-3)
            r = {"raw_asym": float(asym(J))}
            for mode in ("column", "row"):
                p = profile_jacobian(J, mode)
                r[f"{mode}_asym"] = p["asym"]
                r[f"{mode}_resid_rel"] = p["residual_rel"]
                r[f"{mode}_pairs"] = f"{p['pairs_used']}/{p['pairs_total']}"
                r[f"{mode}_rank"] = p["design_rank"]
            rows.append(r)
        print(f"\n  {name}")
        for k in ["raw_asym", "column_asym", "column_resid_rel",
                  "row_asym", "row_resid_rel"]:
            v = np.array([d[k] for d in rows], dtype=float)
            print(f"    {k:<18} [" + "  ".join(f"{x:.6e}" for x in v) +
                  f"]   mean {v.mean():.6e}")
        print(f"    pairs used        {rows[0]['column_pairs']}   "
              f"design rank {rows[0]['column_rank']} of 98")
        out[name] = rows
    return out


if __name__ == "__main__":
    results = {"_config": {"seeds": SEEDS, "delta": DELTA, "Q_seed": Q_SEED,
                           "gp_sigma": GP_SIGMA, "n": 100, "d": 5,
                           "context_sigma": 1.0}}
    results["1A"] = run_1a()
    results["1B"] = run_1b()
    results["profile_residual"] = run_profile_residual()
    p = Path(__file__).resolve().parent / "chunk1_results.json"
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {p}")
