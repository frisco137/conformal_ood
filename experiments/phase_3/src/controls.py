"""T1.4 -- the Tier 1 control battery, including the two new controls.

Existing controls come from Phase 1 unchanged: exact GP (wrapped and unwrapped),
hierarchical GP, targeted imitator. Two are NEW and both are required.

  NoiseHyperpriorGP
      Prior mixes over sigma^2 (log-uniform on [0.25, 4]) at fixed lengthscale.
      This supplies the CLASS-LEVEL FLOOR FOR A1, which the paper does not
      currently have: Phase 1's hierarchical GP mixes LENGTHSCALE ONLY at fixed
      sigma^2, where Brown applies directly and asymmetry sits at ~1e-8. Mixing
      the noise scale is the case a reviewer will actually raise, and T3.1 proves
      the resulting map's Jacobian is symmetric PSD plus a rank-one term along y
      -- so it is a genuine Bayesian object with genuinely nonzero raw asymmetry.
      Tests P5 (projected negeig < 1e-10) and P6 (the honest A1 floor).

  LOOSmoother
      J = D^-1 (K - diag K), the self-excluding kernel smoother. A new N5
      catalogue row and a live mechanism candidate for A2 failure. It is
      row-scaled symmetric BY CONSTRUCTION --
          d_i J_ij = Ktilde_ij = Ktilde_ji = d_j J_ji
      -- so it must profile cleanly in the row system, where TabICL sits at
      residual 0.570. Tests P7.

Both are exact, closed-form, CPU-only maps. Both expose `.predict` (for the
pipeline) and `.jacobian` (for the analytic reference), matching the interface of
`experiments.phase_1.core.surrogates`.
"""
from __future__ import annotations

import numpy as np

__all__ = ["NoiseHyperpriorGP", "LOOSmoother", "SIGMA2_GRID"]

#: log-uniform grid over sigma^2, the plan's [0.25, 4]
SIGMA2_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 9))


class NoiseHyperpriorGP:
    """Exact posterior mean under f ~ N(0, K), sigma^2 ~ uniform on a log grid.

    mu*(y) = sum_k w_k(y) * W_k y,  w_k(y) proportional to N(y; 0, K + s_k I),
    W_k = K (K + s_k I)^-1.

    The Jacobian is available in closed form: differentiating the mixture weights
    gives, with m_k = W_k y and v_k = -C_k^-1 y,

        J = sum_k w_k W_k + sum_k w_k m_k (v_k - vbar)'

    which is the same structure as Phase 1's HierarchicalGP.jacobian, mixing over
    sigma^2 instead of lengthscale.
    """

    def __init__(self, X, sigma2_grid=SIGMA2_GRID, lengthscale=1.0):
        self.X = np.asarray(X, float)
        self.n = len(self.X)
        self.s2 = np.asarray(sigma2_grid, float)
        d2 = np.sum((self.X[:, None] - self.X[None, :]) ** 2, axis=-1)
        self.K = np.exp(-d2 / (2 * lengthscale ** 2))
        self.Ci, self.W, self.logdet = [], [], []
        for s in self.s2:
            C = self.K + s * np.eye(self.n)
            Ci = np.linalg.inv(C)
            self.Ci.append(Ci)
            self.W.append(self.K @ Ci)
            self.logdet.append(np.linalg.slogdet(C)[1])

    def _w(self, y):
        lp = np.array([-0.5 * (y @ Ci @ y + ld)
                       for Ci, ld in zip(self.Ci, self.logdet)])
        lp -= lp.max()
        w = np.exp(lp)
        return w / w.sum()

    def predict(self, y):
        y = np.asarray(y, float).ravel()
        w = self._w(y)
        return sum(w[k] * (self.W[k] @ y) for k in range(len(self.s2)))

    def jacobian(self, y):
        y = np.asarray(y, float).ravel()
        w = self._w(y)
        v = [-(Ci @ y) for Ci in self.Ci]
        vbar = sum(w[k] * v[k] for k in range(len(self.s2)))
        J = np.zeros((self.n, self.n))
        for k in range(len(self.s2)):
            m = self.W[k] @ y
            J += w[k] * self.W[k]
            J += w[k] * np.outer(m, v[k] - vbar)
        return J

    def predictive_variance(self, y):
        """s^2_i for a fresh observation at x_i, mixing over sigma^2.

        Total variance over the sigma^2 posterior: E[Var] + Var[E], plus the
        (mixed) observation noise E[sigma^2 | y].
        """
        y = np.asarray(y, float).ravel()
        w = self._w(y)
        mus = [self.W[k] @ y for k in range(len(self.s2))]
        mbar = sum(w[k] * mus[k] for k in range(len(self.s2)))
        var = np.zeros(self.n)
        for k in range(len(self.s2)):
            post = self.K - self.K @ self.Ci[k] @ self.K
            var += w[k] * (np.diag(post) + (mus[k] - mbar) ** 2)
        return var + float(np.sum(w * self.s2))


class LOOSmoother:
    """Self-excluding kernel smoother: J = D^-1 (K - diag K).

    Row-scaled symmetric by construction, so the N6/N5 ROW system must recover it
    with residual at machine precision. Excluding the diagonal is what makes it a
    live A2-failure candidate: a point's own label is removed from its own
    prediction, so raising y_i lowers m_i through the normalisation only.
    """

    def __init__(self, X, lengthscale=1.0):
        self.X = np.asarray(X, float)
        self.n = len(self.X)
        d2 = np.sum((self.X[:, None] - self.X[None, :]) ** 2, axis=-1)
        K = np.exp(-d2 / (2 * lengthscale ** 2))
        self.Ktilde = K - np.diag(np.diag(K))          # self-excluded
        self.d = self.Ktilde.sum(axis=1)
        self.d = np.where(np.abs(self.d) < 1e-12, 1e-12, self.d)
        self.W = self.Ktilde / self.d[:, None]

    def predict(self, y):
        return self.W @ np.asarray(y, float).ravel()

    def jacobian(self, y=None):
        return self.W


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from experiments.phase_1.core.context import generate_audit_context
    from experiments.phase_1.core.metrics import asym, get_Q, negeig, profile_jacobian

    print("controls self-test, audit context seed 42, n=100")
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=42)
    Q = get_Q(y, seed=0)

    g = NoiseHyperpriorGP(X)
    J = g.jacobian(y)
    Jr = Q.T @ J @ Q
    print(f"\n  NoiseHyperpriorGP  (sigma^2 log-uniform on "
          f"[{SIGMA2_GRID[0]:.2f}, {SIGMA2_GRID[-1]:.2f}], {len(SIGMA2_GRID)} pts)")
    print(f"    ambient   asym {asym(J):.6f}   negeig {negeig(J):.6e}")
    print(f"    projected asym {asym(Jr):.6f}   negeig {negeig(Jr):.6e}   <- P5, P6")

    s = LOOSmoother(X)
    Js = s.jacobian()
    Jsr = Q.T @ Js @ Q
    row = profile_jacobian(Js, "row")
    col = profile_jacobian(Js, "column")
    print(f"\n  LOOSmoother  J = D^-1 (K - diag K)")
    print(f"    ambient   asym {asym(Js):.6f}   negeig {negeig(Js):.6e}")
    print(f"    projected asym {asym(Jsr):.6f}   negeig {negeig(Jsr):.6f}   <- P7")
    print(f"    row  system  residual {row['residual_rel']:.3e}  post-profile asym {row['asym']:.3e}")
    print(f"    col  system  residual {col['residual_rel']:.3e}  post-profile asym {col['asym']:.3e}")
