"""Chunk 2 — A3 cross-channel controls. Never run before.

A3:  s^2(x_i) = sigma^2 (1 + J_ii)     for a fresh observation at x_i.

The two sides come from independent code paths on purpose:
  J_ii   measured by finite differences through the same probe the audit uses
  s^2_i  from the surrogate's closed-form posterior covariance + noise

A3 lives on the AMBIENT Jacobian diagonal. The reduced basis Q mixes context
points, so (Q^T J Q)_ii has no cross-channel interpretation -- the same reason
Chunk 1 found reduced-basis profiling has no power. Everything here is ambient.

Controls only. No model calls.
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.core.context import generate_audit_context
from experiments.core.surrogates import ExactGP, HierarchicalGP
from experiments.core.metrics import central_jacobian

SEEDS = [42, 100, 200, 300, 400]
GP_SIGMA = 0.5
H = 1e-3


def wrap(g):
    def predict(y_eval):
        mu, sd = np.mean(y_eval), np.std(y_eval) + 1e-8
        return mu + sd * np.asarray(g((y_eval - mu) / sd)).flatten()
    return predict


def ols(x, y):
    """s^2 = intercept + slope * (1 + J_ii). Returns slope, intercept, R^2."""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coef[1]), float(coef[0]), r2


def cvs(s2, Jdiag):
    cv_s = float(np.std(s2) / np.mean(s2))
    cv_J = float(np.std(Jdiag) / abs(np.mean(Jdiag)))
    return cv_s, cv_J


def row(tag, s2, Jdiag, sigma2_true):
    slope, icept, r2 = ols(1.0 + Jdiag, s2)
    cv_s, cv_J = cvs(s2, Jdiag)
    return {"tag": tag, "slope": slope, "intercept": icept, "r2": r2,
            "slope_rel_err": abs(slope - sigma2_true) / sigma2_true,
            "intercept_over_sigma2": icept / sigma2_true,
            "cv_s": cv_s, "cv_J": cv_J, "cv_holds": bool(cv_s <= cv_J),
            "Jdiag_min": float(Jdiag.min()), "Jdiag_max": float(Jdiag.max()),
            "s2_min": float(s2.min()), "s2_max": float(s2.max())}


def show(title, rows, sigma2_true):
    print(f"\n  {title}     (true sigma^2 = {sigma2_true})")
    hdr = f"    {'seed':<6} {'slope':>12} {'rel err':>9} {'intercept':>12} " \
          f"{'icept/s2':>9} {'R^2':>12} {'cv_s':>9} {'cv_J':>9}  cv_s<=cv_J"
    print(hdr)
    for s, r in zip(SEEDS, rows):
        print(f"    {s:<6} {r['slope']:>12.8f} {r['slope_rel_err']:>9.2e} "
              f"{r['intercept']:>12.4e} {r['intercept_over_sigma2']:>9.2e} "
              f"{r['r2']:>12.9f} {r['cv_s']:>9.5f} {r['cv_J']:>9.5f}  "
              f"{'yes' if r['cv_holds'] else 'NO'}")
    for k in ["slope", "r2", "cv_s", "cv_J"]:
        v = np.array([r[k] for r in rows])
        print(f"    mean {k:<10} {v.mean():.9f}   std {v.std():.3e}")


def main():
    out = {}
    sigma2 = GP_SIGMA ** 2

    # ---------------------------------------------------------------- 2.1/2.2/2.3
    print("=" * 78)
    print("2.1 - 2.3  Exact GP through the full pipeline")
    print("=" * 78)
    print("  J_ii  : central_jacobian(predict, y, h=1e-3), ambient, diagonal")
    print("  s^2_i : diag(K - K(K+s2 I)^-1 K) + s2, closed form, independent path")

    rows_gp, t2_err = [], []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        egp = ExactGP(X, sigma=GP_SIGMA)
        J = central_jacobian(egp.predict, y, h=H)
        s2 = egp.predictive_variance(y)
        rows_gp.append(row("exactgp", s2, np.diag(J), sigma2))
        # T2 itself: sigma^2 J == Cov(f|y), entrywise
        post_cov = egp.K - egp.K @ egp.inv @ egp.K
        t2_err.append(float(np.linalg.norm(sigma2 * J - post_cov, 'fro')
                            / np.linalg.norm(post_cov, 'fro')))
    show("2.2 / 2.3  ExactGP, unwrapped", rows_gp, sigma2)
    print(f"\n    T2 check ||sigma^2 J - Cov(f|y)||_F / ||Cov(f|y)||_F  per seed:")
    print("      [" + "  ".join(f"{e:.3e}" for e in t2_err) + "]")
    out["exactgp"] = {"rows": rows_gp, "t2_rel_err": t2_err}

    # ------------------------------------------------------------------- 2.4
    print("\n" + "=" * 78)
    print("2.4  Hierarchical GP -- the law holds for every prior")
    print("=" * 78)
    rows_h, t4_err = [], []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        hgp = HierarchicalGP(X, sigma=GP_SIGMA)
        J = central_jacobian(hgp.predict, y, h=H)
        s2 = hgp.predictive_variance(y)
        rows_h.append(row("hiergp", s2, np.diag(J), sigma2))
        pc = hgp.posterior_covariance(y)
        t4_err.append(float(np.linalg.norm(sigma2 * J - pc, 'fro')
                            / np.linalg.norm(pc, 'fro')))
    show("2.4  HierarchicalGP, unwrapped", rows_h, sigma2)
    print(f"\n    T4 check ||sigma^2 J - (E[C] + Cov(mu))||_F / ||.||_F  per seed:")
    print("      [" + "  ".join(f"{e:.3e}" for e in t4_err) + "]")
    out["hiergp"] = {"rows": rows_h, "t4_rel_err": t4_err}

    # ------------------------------------------------------------------- 2.5
    print("\n" + "=" * 78)
    print("2.5  Deliberately broken variance heads -- does the test have power?")
    print("=" * 78)
    print("  Same exact-GP mean map and same J. Only s^2 is corrupted.")
    print("    constant   s^2_i := mean(s^2)                 (no variation at all)")
    print("    scaled     s^2_i := s^2_i * lognormal(0, 0.5) (variation, wrong pattern)")
    print("    permuted   s^2_i := s^2_{pi(i)}               (cv_s EXACTLY preserved)")

    broken = {k: [] for k in ["constant", "scaled", "permuted"]}
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        egp = ExactGP(X, sigma=GP_SIGMA)
        Jd = np.diag(central_jacobian(egp.predict, y, h=H))
        s2 = egp.predictive_variance(y)
        rng = np.random.RandomState(s)
        broken["constant"].append(row("constant", np.full_like(s2, s2.mean()), Jd, sigma2))
        broken["scaled"].append(row("scaled", s2 * np.exp(rng.normal(0, 0.5, len(s2))), Jd, sigma2))
        broken["permuted"].append(row("permuted", s2[rng.permutation(len(s2))], Jd, sigma2))
    for k in broken:
        show(f"2.5  broken: {k}", broken[k], sigma2)
    out["broken"] = broken

    # -------------------------------------------- extra: does A3 survive a wrapper?
    print("\n" + "=" * 78)
    print("EXTRA  Normaliser-wrapped exact GP -- all three audited models wear one")
    print("=" * 78)
    print("  mean map   m(y) = ybar + s_y * g(u),  g = exact GP posterior mean")
    print("  variance   s^2_wrapped = s_y^2 * s^2_inner, the natural un-normalisation")
    rows_w = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        egp = ExactGP(X, sigma=GP_SIGMA)
        J = central_jacobian(wrap(egp.predict), y, h=H)
        s_y = np.std(y)
        s2 = (s_y ** 2) * egp.predictive_variance(y)
        rows_w.append(row("wrapped", s2, np.diag(J), sigma2 * s_y ** 2))
    print("\n  (slope is compared against sigma^2 * s_y^2, the wrapper-adjusted target)")
    for s, r in zip(SEEDS, rows_w):
        print(f"    seed {s:<5} slope {r['slope']:>11.6f}  rel err {r['slope_rel_err']:>8.2e}  "
              f"R^2 {r['r2']:>11.8f}  cv_s {r['cv_s']:.5f}  cv_J {r['cv_J']:.5f}  "
              f"cv holds: {'yes' if r['cv_holds'] else 'NO'}")
    v = np.array([r["r2"] for r in rows_w])
    print(f"    mean R^2 {v.mean():.8f}   std {v.std():.3e}")
    out["wrapped"] = rows_w

    p = Path(__file__).resolve().parent / "chunk2_results.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
