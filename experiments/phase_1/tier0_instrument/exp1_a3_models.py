"""SUPERSEDED FOR OUTPUT, RETAINED FOR ITS SPECIFICATION.

FINAL_NUMBERS.md section 9.3: "new, superseded by exp12_...; partial log only;
not a source of any number here." Do not read a result off this script.

It is NOT quarantined because FINAL_NUMBERS.md section 5 cites it as SOURCE
CODE, not as data -- it quotes the A3 ordinary-least-squares specification at
exp1_a3_models.py:52-58, called at :137. Moving the file would break a live
citation in a frozen document.

The A3 numbers themselves come from exp12_ambient_jacobians.py ->
exp12_ambient_jacobians.npz -> exp1_a3_final.json.
"""

"""Experiment 1 — A3 cross-channel regression on the models. First time run.

    A3:  s^2(x_i) = sigma^2 (1 + J_ii)      fresh observation at x_i

Ambient Jacobian diagonal, as Chunk 2 established: (Q^T J Q)_ii has no
cross-channel meaning because Q mixes context points.

Variance channel, per model, stated explicitly:

  TabICL v2  no closed-form predictive variance is exposed. output_type="variance"
             is raw_quantiles.var(dim=-1) (tabicl/_model/tabicl.py:582), the
             spread of the quantile grid, which the plan's E2.5 "required fixes"
             rule out. Used instead: output_type="quantiles" on a 9999-level
             uniform grid, integrated by core/metrics.py::extract_variance:
                 mean = trapz(Q(p), p);  E[X^2] = trapz(Q(p)^2, p);  var = E[X^2]-mean^2
             Validated against the exact GP at 0.03% worst-case (exp1_extractor_validation).

  TabPFN v2  FullSupportBarDistribution.variance(logits) = mean_of_square - mean^2,
             tabpfn/architectures/base/bar_distribution.py:606 and :412. This is the
             integrated second moment over the full predictive distribution, with
             half-normal tails on the outer bins -- exactly what E2.5 requires.

  TabSwift   Linear(384,1) point head, has_predictive_distribution=False in
             models/registry.py. NOT COMPUTABLE.
"""
import sys
import json
import numpy as np
from pathlib import Path


class _MockAnalytics:
    def __getattr__(self, name):
        return lambda *a, **k: None


sys.modules['analytics'] = _MockAnalytics()
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.metrics import extract_variance

SEEDS = [42, 100, 200, 300, 400]
DELTA = 6.87e-4
LEVELS = np.linspace(1e-4, 1 - 1e-4, 9999)


def ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(coef[1]), float(coef[0]), (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def ambient_diag(predict, y, h, dither=False, N=10, halfwidth=DELTA, seed=0):
    """Diagonal of the ambient Jacobian by central differences, one column at a time."""
    n = len(y)
    diag = np.empty(n)
    for i in range(n):
        ep = np.zeros(n); ep[i] = h
        if dither:
            rng = np.random.RandomState(seed + i)
            noise = rng.uniform(-halfwidth, halfwidth, (N, n))
            mp = np.mean([predict(y + noise[k] + ep) for k in range(N)], axis=0)
            mm = np.mean([predict(y + noise[k] - ep) for k in range(N)], axis=0)
        else:
            mp, mm = predict(y + ep), predict(y - ep)
        diag[i] = (np.asarray(mp).ravel()[i] - np.asarray(mm).ravel()[i]) / (2 * h)
    return diag


def tabicl_variance(est, X):
    Q = np.asarray(est.predict(X, output_type="quantiles", alphas=list(LEVELS)))
    return np.asarray(extract_variance(Q, LEVELS)).ravel()


def tabpfn_variance(est, X):
    full = est.predict(X, output_type="full")
    crit, lg = full["criterion"], full["logits"]
    lg = lg if torch.is_tensor(lg) else torch.as_tensor(lg)
    lg = lg.to(next(crit.buffers()).device).float()
    return crit.variance(lg).detach().cpu().numpy().ravel()


def run(model_id, h, dither, N, tag):
    print(f"\n{'='*78}\n{tag}\n{'='*78}", flush=True)
    model = load(model_id, task="regression", device="cuda")
    rows = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)

        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            if model_id == "tabpfn_v2":
                return model.estimator.predict(X, output_type="mean")
            return model.estimator.predict(X)

        Jd = ambient_diag(predict, y, h, dither=dither, N=N, seed=s)
        model.estimator.fit(X, y)
        s2 = tabpfn_variance(model.estimator, X) if model_id == "tabpfn_v2" \
            else tabicl_variance(model.estimator, X)

        slope, icept, r2 = ols(1.0 + Jd, s2)
        s_y = float(np.std(y))
        cv_s = float(np.std(s2) / np.mean(s2))
        cv_J = float(np.std(Jd) / abs(np.mean(Jd)))
        rows.append({"seed": s, "slope": slope, "intercept": icept, "r2": r2,
                     "sigma2_hat": slope / s_y**2, "s_y": s_y,
                     "cv_s": cv_s, "cv_J": cv_J,
                     "J_min": float(Jd.min()), "J_max": float(Jd.max()),
                     "J_mean": float(Jd.mean()), "J_std": float(Jd.std()),
                     "s2_min": float(s2.min()), "s2_max": float(s2.max()),
                     "s2_mean": float(s2.mean())})
        r = rows[-1]
        print(f"  seed {s:<5} slope {slope:+.8f}  icept {icept:+.6f}  R^2 {r2:+.8f}  "
              f"sigma2_hat {r['sigma2_hat']:.6f}  s_y {s_y:.4f}  "
              f"cv_s {cv_s:.5f}  cv_J {cv_J:.5f}", flush=True)
        print(f"             J_ii in [{r['J_min']:+.5f}, {r['J_max']:+.5f}]  "
              f"s2 in [{r['s2_min']:.5f}, {r['s2_max']:.5f}]", flush=True)
    for k in ["slope", "r2", "sigma2_hat", "cv_s", "cv_J"]:
        v = np.array([r[k] for r in rows])
        print(f"  mean {k:<11} {v.mean():+.8f}   std {v.std():.3e}", flush=True)
    return rows


if __name__ == "__main__":
    out = {"_config": {"seeds": SEEDS, "levels_n": len(LEVELS),
                       "levels_range": [float(LEVELS[0]), float(LEVELS[-1])],
                       "delta": DELTA, "n": 100, "d": 5, "context_sigma": 1.0}}
    out["tabicl_v2"] = run("tabicl_v2", 1e-3, False, 1,
                           "TabICL v2   h=1e-3, no dither, ambient diagonal")
    json.dump(out, open(Path(__file__).resolve().parent / "exp1_a3_models.json", "w"), indent=2)

    out["tabpfn_v2_nodither"] = run("tabpfn_v2", 1e-1, False, 1,
                                    "TabPFN v2   h=1e-1, no dither, ambient diagonal")
    json.dump(out, open(Path(__file__).resolve().parent / "exp1_a3_models.json", "w"), indent=2)

    out["tabpfn_v2_dither"] = run("tabpfn_v2", 1e-1, True, 10,
                                  "TabPFN v2   h=1e-1, dither halfwidth=delta N=10, ambient diagonal")
    out["tabswift"] = "NOT COMPUTABLE - Linear(384,1) point head, no predictive distribution"

    p = Path(__file__).resolve().parent / "exp1_a3_models.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")
