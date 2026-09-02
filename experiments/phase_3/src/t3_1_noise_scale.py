"""T3.1 -- unknown noise scale. Numerical verification of the derivation.

WHY THIS BLOCKS EVERY A1 NUMBER WE QUOTE
----------------------------------------
Assumption 1 fixes sigma^2 KNOWN. The prior these models were trained on does not
fix it. So the Bayesian class a reviewer will invoke is posterior means under a
JOINT prior on (f, sigma^2), which is strictly larger than the class Brown's
identity applies to -- and for that larger class the posterior mean's Jacobian
need not be symmetric at all.

THE DERIVATION (proved in t3_1_noise_scale.md; verified here)
-------------------------------------------------------------
With p(y) = int phi_{sigma^2}(y - f) dP(f, sigma^2), define

    h(y) = E[sigma^-2 f | y]        w(y) = E[sigma^-2 | y]
    g(y) = E[sigma^-2 (y - f) | y] = w(y) y - h(y)

Differentiating under the integral gives, in order:

    (1)  grad log p(y) = -g(y)                       generalised Tweedie
    (2)  Cov(sigma^-2 (y-f) | y) = w(y) I + Hess log p(y)     symmetric, PSD
    (3)  J_h = Cov(sigma^-2 (y-f) | y) + y grad w(y)'

So the precision-weighted map h is a symmetric PSD object plus a contamination
that is RANK ONE ALONG y. Since span{1, u} = span{1, y}, the audit's projection Q
annihilates it on the left (Q'y = 0), and a tangent-sphere loop with planes drawn
orthogonal to both 1 and y never moves along it either.

WHAT THIS SCRIPT CHECKS, on an exact sigma^2-mixing posterior mean:
  V1  identity (1), grad log p = -g
  V2  identity (2), the covariance is symmetric PSD and equals w I + Hess log p
  V3  identity (3), J_h - Cov is rank one and its range is spanned by y
  V4  mu*(y) = E[f|y] under sigma^2 mixing has O(0.1) AMBIENT asymmetry -- i.e.
      the honest class-level A1 floor is nowhere near 1e-3
  V5  that asymmetry is strongly SUPPRESSED in planes orthogonal to span{1,y}
      relative to planes containing y -- the claim that makes both the projection
      and the loop the right instruments for a second, independent reason

Writes results/t3_1_noise_scale.json. CPU only.
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

from circulation import centred_basis                                 # noqa: E402
from experiments.phase_1.core.context import generate_audit_context    # noqa: E402

N = 60
SIGMA2_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 9))   # log-uniform
SEEDS = [0, 1, 2, 3, 4]


class SigmaMixingPosterior:
    """Exact posterior under a joint prior: f ~ N(0, K), sigma^2 ~ log-uniform grid.

    Everything is closed form. For each grid point,
        mu_k(y)      = K (K + s_k I)^-1 y
        log p(y|s_k) = -0.5 (y' C_k^-1 y + logdet C_k + n log 2pi),  C_k = K + s_k I
    and the posterior over sigma^2 is the normalised likelihood times the (uniform
    on the grid) prior.
    """

    def __init__(self, X, sigma2_grid=SIGMA2_GRID, lengthscale=1.0):
        self.n = len(X)
        d2 = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
        self.K = np.exp(-d2 / (2 * lengthscale ** 2))
        self.s2 = np.asarray(sigma2_grid, float)
        self.C, self.Ci, self.W, self.logdet = [], [], [], []
        for s in self.s2:
            C = self.K + s * np.eye(self.n)
            Ci = np.linalg.inv(C)
            self.C.append(C); self.Ci.append(Ci)
            self.W.append(self.K @ Ci)
            self.logdet.append(np.linalg.slogdet(C)[1])

    def log_post_weights(self, y):
        lp = np.array([-0.5 * (y @ Ci @ y + ld + self.n * np.log(2 * np.pi))
                       for Ci, ld in zip(self.Ci, self.logdet)])
        lp -= lp.max()
        w = np.exp(lp)
        return w / w.sum()

    def log_p(self, y):
        lp = np.array([-0.5 * (y @ Ci @ y + ld + self.n * np.log(2 * np.pi))
                       for Ci, ld in zip(self.Ci, self.logdet)])
        m = lp.max()
        return m + np.log(np.mean(np.exp(lp - m)))

    def mu_star(self, y):
        """E[f | y] under the joint prior -- the object a reviewer calls Bayesian."""
        w = self.log_post_weights(y)
        return sum(w[k] * (self.W[k] @ y) for k in range(len(self.s2)))

    def w_fn(self, y):
        """w(y) = E[sigma^-2 | y]."""
        wt = self.log_post_weights(y)
        return float(np.sum(wt / self.s2))

    def h_fn(self, y):
        """h(y) = E[sigma^-2 f | y]."""
        wt = self.log_post_weights(y)
        return sum(wt[k] / self.s2[k] * (self.W[k] @ y) for k in range(len(self.s2)))

    def g_fn(self, y):
        """g(y) = E[sigma^-2 (y - f) | y] = w(y) y - h(y)."""
        return self.w_fn(y) * y - self.h_fn(y)

    def cov_prec_resid(self, y):
        """Cov(sigma^-2 (y - f) | y), by the law of total covariance over the grid.

        Given sigma^2 = s_k, (y - f) | y, s_k ~ N(y - mu_k, Cov_k) with
        Cov_k = K - K C_k^-1 K, so sigma^-2 (y-f) has mean (y - mu_k)/s_k and
        covariance Cov_k / s_k^2.
        """
        wt = self.log_post_weights(y)
        means = [(y - self.W[k] @ y) / self.s2[k] for k in range(len(self.s2))]
        gbar = sum(wt[k] * means[k] for k in range(len(self.s2)))
        out = np.zeros((self.n, self.n))
        for k in range(len(self.s2)):
            Ck = self.K - self.K @ self.Ci[k] @ self.K
            d = means[k] - gbar
            out += wt[k] * (Ck / self.s2[k] ** 2 + np.outer(d, d))
        return out


def fd_jac(f, y, h=1e-5):
    n = len(y)
    J = np.empty((n, n))
    for j in range(n):
        yp = y.copy(); yp[j] += h
        ym = y.copy(); ym[j] -= h
        J[:, j] = (np.asarray(f(yp)).ravel() - np.asarray(f(ym)).ravel()) / (2 * h)
    return J


def fd_grad(f, y, h=1e-5):
    n = len(y)
    g = np.empty(n)
    for j in range(n):
        yp = y.copy(); yp[j] += h
        ym = y.copy(); ym[j] -= h
        g[j] = (f(yp) - f(ym)) / (2 * h)
    return g


def asym_ratio(J):
    return float(np.linalg.norm(J - J.T, "fro") / np.linalg.norm(J, "fro"))


def main():
    out = {"_config": {"n": N, "sigma2_grid": SIGMA2_GRID.tolist(), "seeds": SEEDS},
           "rows": []}
    print("=" * 88)
    print("T3.1  unknown noise scale -- numerical verification at n = 60")
    print(f"      sigma^2 grid: log-uniform, {len(SIGMA2_GRID)} points in "
          f"[{SIGMA2_GRID[0]:.2f}, {SIGMA2_GRID[-1]:.2f}]")
    print("=" * 88)

    for s in SEEDS:
        X, y, _ = generate_audit_context(n=N, d=5, sigma=1.0, seed=42 + s)
        P = SigmaMixingPosterior(X)

        # V1 -- generalised Tweedie
        g = P.g_fn(y)
        gl = fd_grad(P.log_p, y)
        v1 = float(np.max(np.abs(gl + g)) / max(np.max(np.abs(g)), 1e-30))

        # V2 -- Cov = w I + Hess log p, symmetric PSD
        Cov = P.cov_prec_resid(y)
        Hess = fd_jac(lambda yy: fd_grad(P.log_p, yy, h=1e-4), y, h=1e-4)
        Hess = (Hess + Hess.T) / 2
        pred = P.w_fn(y) * np.eye(N) + Hess
        v2 = float(np.linalg.norm(Cov - pred, "fro") / np.linalg.norm(Cov, "fro"))
        v2_sym = asym_ratio(Cov)
        v2_psd = float(np.min(np.linalg.eigvalsh((Cov + Cov.T) / 2)))

        # V3 -- J_h - Cov is rank one, range spanned by y
        Jh = fd_jac(P.h_fn, y)
        D = Jh - Cov
        sv = np.linalg.svd(D, compute_uv=False)
        rank1 = float(sv[1] / sv[0]) if sv[0] > 0 else 0.0
        U = np.linalg.svd(D)[0][:, 0]
        align = float(abs(U @ (y / np.linalg.norm(y))))

        # V4 -- ambient asymmetry of mu*(y) = E[f|y]
        Jmu = fd_jac(P.mu_star, y)
        amb = asym_ratio(Jmu)

        # V5 -- skew in planes orthogonal to span{1,y} vs planes containing y
        rng = np.random.default_rng(100 + s)
        A = Jmu - Jmu.T
        perp, contain = [], []
        for _ in range(300):
            a, b = centred_basis(N, rng, exclude=y)          # a,b _|_ {1, y}
            perp.append(abs(a @ A @ b))
        yhat = y / np.linalg.norm(y)
        for _ in range(300):
            a, _b = centred_basis(N, rng)                     # only _|_ 1
            v = yhat - (yhat @ a) * a
            v /= np.linalg.norm(v)
            contain.append(abs(a @ A @ v))                    # plane contains y
        mp, mc = float(np.mean(perp)), float(np.mean(contain))

        row = {"seed": 42 + s,
               "V1_tweedie_rel": v1,
               "V2_cov_identity_rel": v2, "V2_cov_asym": v2_sym, "V2_cov_min_eig": v2_psd,
               "V3_rank1_ratio_sv2_over_sv1": rank1, "V3_align_top_sv_with_y": align,
               "V4_mustar_ambient_asym": amb,
               "V5_mean_skew_perp": mp, "V5_mean_skew_containing_y": mc,
               "V5_suppression": (mc / mp) if mp > 0 else float("inf")}
        out["rows"].append(row)
        print(f"\n  seed {42+s}")
        print(f"    V1  grad log p = -g                    rel err {v1:.3e}")
        print(f"    V2  Cov = w I + Hess log p             rel err {v2:.3e}   "
              f"asym {v2_sym:.2e}   min eig {v2_psd:+.3e}")
        print(f"    V3  J_h - Cov rank one                 sv2/sv1 {rank1:.3e}   "
              f"|<u1, y/|y|>| {align:.6f}")
        print(f"    V4  mu*=E[f|y] AMBIENT asym            {amb:.4f}")
        print(f"    V5  mean |a'(J-J')b|  perp to {{1,y}}    {mp:.3e}")
        print(f"        mean |a'(J-J')b|  containing y     {mc:.3e}   "
              f"suppression {mc/mp:.1f}x")

    r = out["rows"]
    agg = {k: float(np.mean([q[k] for q in r])) for k in r[0] if k != "seed"}
    out["aggregate"] = agg
    print("\n" + "=" * 88)
    print("AGGREGATE over 5 contexts")
    print("=" * 88)
    print(f"  V1 generalised Tweedie          max rel err {max(q['V1_tweedie_rel'] for q in r):.3e}")
    print(f"  V2 covariance identity          max rel err {max(q['V2_cov_identity_rel'] for q in r):.3e}")
    print(f"     symmetric to                 {max(q['V2_cov_asym'] for q in r):.3e}"
          f"   min eigenvalue {min(q['V2_cov_min_eig'] for q in r):+.3e}")
    print(f"  V3 contamination rank one       max sv2/sv1 {max(q['V3_rank1_ratio_sv2_over_sv1'] for q in r):.3e}")
    print(f"     top singular vector || y     min |cos| {min(q['V3_align_top_sv_with_y'] for q in r):.6f}")
    print(f"  V4 mu* AMBIENT asym             mean {agg['V4_mustar_ambient_asym']:.4f}"
          f"   range [{min(q['V4_mustar_ambient_asym'] for q in r):.4f}, "
          f"{max(q['V4_mustar_ambient_asym'] for q in r):.4f}]")
    print(f"  V5 suppression perp vs along y  mean {agg['V5_suppression']:.1f}x"
          f"   range [{min(q['V5_suppression'] for q in r):.1f}x, "
          f"{max(q['V5_suppression'] for q in r):.1f}x]")

    p = HERE.parent / "results" / "t3_1_noise_scale.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
