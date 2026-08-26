"""Experiment 1.1 — validate the variance extractor before using it.

Gate 1 part (a): extract_variance must recover the exact GP's closed-form
predictive variance to < 5% through the identical code path.

The extractor is core/metrics.py::extract_variance, which integrates the
second moment over a quantile grid:

    mean = trapz(Q(p), p)        E[X^2] = trapz(Q(p)^2, p)
    var  = E[X^2] - mean^2

It integrates over [p_first, p_last], not [0, 1], so a grid that stops short of
the tails is biased low. That bias is exactly what this measures. No Gaussian
mock: the GP's own closed-form quantiles are pushed through the same call.

Also runs the wrapped-GP A3 control at h = 1e-1, the step Experiment 1 uses on
TabPFN. Chunk 2 validated A3 through a normaliser only at h = 1e-3.
"""
import sys
import json
import numpy as np
from pathlib import Path
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.core.context import generate_audit_context
from experiments.core.surrogates import ExactGP
from experiments.core.metrics import central_jacobian, extract_variance

SEEDS = [42, 100, 200, 300, 400]
GP_SIGMA = 0.5


def wrap(g):
    def predict(y_eval):
        mu, sd = np.mean(y_eval), np.std(y_eval) + 1e-8
        return mu + sd * np.asarray(g((y_eval - mu) / sd)).flatten()
    return predict


def ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(coef[1]), float(coef[0]), (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


GRIDS = {
    "999 levels, 0.001..0.999": np.linspace(0.001, 0.999, 999),
    "999 levels, 1e-4..1-1e-4": np.linspace(1e-4, 1 - 1e-4, 999),
    "9999 levels, 1e-5..1-1e-5": np.linspace(1e-5, 1 - 1e-5, 9999),
    "9 levels, 0.1..0.9 (TabICL default)": np.linspace(0.1, 0.9, 9),
}

out = {}
print("=" * 78)
print("1.1  extract_variance vs the exact GP closed form")
print("=" * 78)
print("  closed form : diag(K - K(K+s2 I)^-1 K) + s2   (core/surrogates.py:19)")
print("  extractor   : core/metrics.py::extract_variance on Gaussian quantiles")
print("                Q(p) = mu_i + sqrt(v_i) * Phi^-1(p), v_i the closed form\n")

for name, levels in GRIDS.items():
    errs = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        egp = ExactGP(X, sigma=GP_SIGMA)
        v_true = egp.predictive_variance(y)              # closed form
        mu = egp.predict(y)
        # the model-side object: a quantile grid, exactly what TabICL returns
        Q = mu[:, None] + np.sqrt(v_true)[:, None] * norm.ppf(levels)[None, :]
        v_hat = extract_variance(Q, levels)
        errs.append(float(np.max(np.abs(v_hat - v_true) / v_true)))
    e = np.array(errs)
    flag = "PASS" if e.max() < 0.05 else "FAIL"
    print(f"  {name:38}  max rel err per seed [" +
          "  ".join(f"{x:.4f}" for x in e) + f"]  worst {e.max():.4f}  {flag}")
    out[name] = errs

print("\n" + "=" * 78)
print("EXTRA  wrapped-GP A3 control at the step Experiment 1 uses on TabPFN")
print("=" * 78)
print("  Chunk 2 validated A3 through a normaliser at h=1e-3 only (R^2 0.99999115).")
print("  TabPFN will be probed at h=1e-1, so the control must be re-run there.\n")
ctrl = {}
for h in [1e-3, 1e-2, 1e-1]:
    rows = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        egp = ExactGP(X, sigma=GP_SIGMA)
        J = central_jacobian(wrap(egp.predict), y, h=h)
        s_y = np.std(y)
        s2 = (s_y ** 2) * egp.predictive_variance(y)
        slope, icept, r2 = ols(1.0 + np.diag(J), s2)
        rows.append({"h": h, "seed": s, "slope": slope, "intercept": icept, "r2": r2,
                     "s_y": float(s_y), "target": float(GP_SIGMA**2 * s_y**2),
                     "rel_err": abs(slope - GP_SIGMA**2 * s_y**2) / (GP_SIGMA**2 * s_y**2)})
    r2s = np.array([r["r2"] for r in rows]); re = np.array([r["rel_err"] for r in rows])
    print(f"  h={h:.0e}  R^2 per seed [" + "  ".join(f"{x:.8f}" for x in r2s) +
          f"]  mean {r2s.mean():.8f}")
    print(f"          slope rel err vs sigma^2*s_y^2 [" +
          "  ".join(f"{x:.3e}" for x in re) + f"]  worst {re.max():.3e}")
    ctrl[f"{h:.0e}"] = rows
out["wrapped_gp_control"] = ctrl

p = Path(__file__).resolve().parent / "exp1_extractor_validation.json"
json.dump(out, open(p, "w"), indent=2)
print(f"\nSaved: {p}")
