"""Shared library for the Jacobian-derived predictive variance experiment.

Nothing canonical is reimplemented here. `asym`, `negeig`, `get_Q`,
`profile_jacobian`, `second_difference_norm` are imported from
`experiments.phase_1.core.metrics` wherever they are needed; `ExactGP`,
`HierarchicalGP`, `NadarayaWatson`, `Imitator` from
`experiments.phase_1.core.surrogates`.

What IS new here, and why:

  * The audit's surrogates are in-sample objects: `ExactGP(X).predict(y)` returns
    the map at the SAME X it was built on, and has no notion of a held-out
    query. This experiment needs out-of-sample prediction and the closed-form
    predictive variance at a query. `GPTrainTest` / `HierGPTrainTest` below add
    exactly that, with kernels written to be bit-comparable to the audit's --
    `check_matches_audit_surrogate()` asserts it, and chunk 1 runs that check.

  * The append-and-difference route for `J_**`, the noise-scale estimator, the
    two variance forms, split conformal, and the interval metrics.

THE ESTIMATOR FORM -- see CORRECTION_NOTE in this module's docstring below.

The brief specifies

    s2_jac(x_*) = sigma2_hat * (1 + J_**)

with `J_** = dm_*/dy_*` measured after appending `(x_*, y_*)` to the context.
That form does not estimate the held-out predictive variance. It is off by
12-17% on the exact GP, where the answer is closed form, and it is bounded above
by `2 sigma2` while the quantity it targets is unbounded. Both forms are
implemented and both are reported; chunk 1 measures the gap. Derivation in
`s2_from_Jstar`.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Kernel, written once, matching experiments/core/surrogates.py exactly:
#   dists = sum((X[:,None] - X[None,:])**2, -1);  K = exp(-dists / (2*ls**2))
# ---------------------------------------------------------------------------


def rbf(A, B, lengthscale=1.0):
    d2 = np.sum((np.asarray(A)[:, None] - np.asarray(B)[None, :]) ** 2, axis=-1)
    return np.exp(-d2 / (2.0 * lengthscale ** 2))


class GPTrainTest:
    """Exact GP with out-of-sample prediction and closed-form predictive variance.

    Same kernel and same `sigma` convention as `core.surrogates.ExactGP`
    (sigma is the noise STANDARD DEVIATION; sigma**2 enters the Gram inverse).
    """

    def __init__(self, X, y, sigma=0.5, lengthscale=1.0):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float).ravel()
        self.sigma = float(sigma)
        self.ls = float(lengthscale)
        self.K = rbf(self.X, self.X, self.ls)
        self.inv = np.linalg.inv(self.K + self.sigma ** 2 * np.eye(len(self.X)))

    def predict(self, X_test):
        Ks = rbf(np.asarray(X_test, float), self.X, self.ls)
        return Ks @ self.inv @ self.y

    def predictive_variance(self, X_test):
        """s^2(x_*) = Var(f_*|y) + sigma^2, for a FRESH observation at x_*.

        Conditioned on the original context only -- this is the quantity the
        experiment is trying to estimate.
        """
        X_test = np.asarray(X_test, float)
        Ks = rbf(X_test, self.X, self.ls)
        kss = np.diag(rbf(X_test, X_test, self.ls))
        return kss - np.einsum("ij,jk,ik->i", Ks, self.inv, Ks) + self.sigma ** 2


class HierGPTrainTest:
    """Hierarchical GP: mixture over lengthscales, weights from the marginal.

    Same lengthscale ladder and weighting as `core.surrogates.HierarchicalGP`
    (8 lengthscales, logspace(0.2, 5.0)). Adds out-of-sample prediction and the
    mixture predictive variance
        Var(f_*|y) = E_theta[Var_theta(f_*|y)] + Var_theta(E_theta[f_*|y])
    which is T4 evaluated at the query rather than at the context.
    """

    def __init__(self, X, y, sigma=0.5):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y, float).ravel()
        self.sigma = float(sigma)
        n = len(self.X)
        self.lengthscales = np.logspace(np.log10(0.2), np.log10(5.0), 8)
        self.Ks, self.invs = [], []
        for ls in self.lengthscales:
            K = rbf(self.X, self.X, ls)
            self.Ks.append(K)
            self.invs.append(np.linalg.inv(K + self.sigma ** 2 * np.eye(n)))
        self.w = self._weights()

    def _weights(self):
        n = len(self.X)
        logw = []
        for K, inv in zip(self.Ks, self.invs):
            _, logdet = np.linalg.slogdet(K + self.sigma ** 2 * np.eye(n))
            logw.append(-0.5 * (logdet + self.y @ inv @ self.y + n * np.log(2 * np.pi)))
        logw = np.asarray(logw)
        w = np.exp(logw - logw.max())
        return w / w.sum()

    def predict(self, X_test):
        X_test = np.asarray(X_test, float)
        out = np.zeros(len(X_test))
        for i, ls in enumerate(self.lengthscales):
            out += self.w[i] * (rbf(X_test, self.X, ls) @ self.invs[i] @ self.y)
        return out

    def predictive_variance(self, X_test):
        X_test = np.asarray(X_test, float)
        mus, EV = [], np.zeros(len(X_test))
        for i, ls in enumerate(self.lengthscales):
            Ks = rbf(X_test, self.X, ls)
            kss = np.diag(rbf(X_test, X_test, ls))
            EV += self.w[i] * (kss - np.einsum("ij,jk,ik->i", Ks, self.invs[i], Ks))
            mus.append(Ks @ self.invs[i] @ self.y)
        mus = np.asarray(mus)
        mbar = np.einsum("i,ij->j", self.w, mus)
        Vmu = np.einsum("i,ij->j", self.w, (mus - mbar) ** 2)
        return EV + Vmu + self.sigma ** 2


def check_matches_audit_surrogate(X, y, sigma=0.5, lengthscale=1.0):
    """The train/test GPs must be the SAME objects as the audit's in-sample ones.

    Returns max abs deviation of (a) the mean map and (b) the Jacobian /
    predictive variance, evaluated in-sample against core.surrogates.
    """
    from experiments.phase_1.core.surrogates import ExactGP, HierarchicalGP

    egp = ExactGP(X, sigma=sigma, lengthscale=lengthscale)
    mine = GPTrainTest(X, y, sigma=sigma, lengthscale=lengthscale)
    hgp = HierarchicalGP(X, sigma=sigma)
    hmine = HierGPTrainTest(X, y, sigma=sigma)
    return {
        "exactgp_mean": float(np.max(np.abs(egp.predict(y) - mine.predict(X)))),
        "exactgp_var": float(np.max(np.abs(egp.predictive_variance(y)
                                           - mine.predictive_variance(X)))),
        "hiergp_mean": float(np.max(np.abs(hgp.predict(y) - hmine.predict(X)))),
        "hiergp_var": float(np.max(np.abs(hgp.predictive_variance(y)
                                          - hmine.predictive_variance(X)))),
    }


# ---------------------------------------------------------------------------
# The append-and-difference route
# ---------------------------------------------------------------------------


def jacobian_star(predict, X_ctx, y_ctx, x_star, y_star, h):
    """J_** = dm_*/dy_*, with (x_*, y_*) appended to the context.

    `predict(X_train, y_train, X_test) -> np.ndarray` is the only model
    interface used. Two forward passes.
    """
    Xa = np.vstack([np.asarray(X_ctx, float), np.asarray(x_star, float).reshape(1, -1)])
    ya = np.append(np.asarray(y_ctx, float).ravel(), float(y_star))
    xq = Xa[-1:].copy()

    yp = ya.copy(); yp[-1] += h
    ym = ya.copy(); ym[-1] -= h
    mp = float(np.asarray(predict(Xa, yp, xq)).ravel()[0])
    mm = float(np.asarray(predict(Xa, ym, xq)).ravel()[0])
    return (mp - mm) / (2.0 * h)


def jacobian_star_batch(predict, X_ctx, y_ctx, X_star, y_star_rule, h):
    """`jacobian_star` over many queries. Returns (J_star, y_star_used).

    `y_star_rule` is either a float, or a callable(y_ctx) -> float, or an array
    of one hypothetical label per query.
    """
    X_star = np.asarray(X_star, float)
    if callable(y_star_rule):
        ys = np.full(len(X_star), float(y_star_rule(y_ctx)))
    elif np.isscalar(y_star_rule):
        ys = np.full(len(X_star), float(y_star_rule))
    else:
        ys = np.asarray(y_star_rule, float).ravel()
    J = np.array([jacobian_star(predict, X_ctx, y_ctx, X_star[i], ys[i], h)
                  for i in range(len(X_star))])
    return J, ys


def context_jacobian(predict, X_ctx, y_ctx, h):
    """Full ambient n x n context Jacobian by central differences.

    The diagonal is what `tr J` needs; the whole matrix costs the same 2n
    forward passes and is kept for chunk 4.
    """
    X_ctx = np.asarray(X_ctx, float)
    y = np.asarray(y_ctx, float).ravel()
    n = len(y)
    J = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = h
        mp = np.asarray(predict(X_ctx, y + e, X_ctx)).ravel()
        mm = np.asarray(predict(X_ctx, y - e, X_ctx)).ravel()
        J[:, i] = (mp - mm) / (2.0 * h)
    return J


def sigma2_hat(m_ctx, y_ctx, trJ):
    """sigma2_hat = ||m(y) - y||^2 / (n - tr J), the effective-dof estimator.

    Returns (sigma2_hat, n_minus_trJ, defined). `defined` is False when
    n - tr J <= 0, in which case sigma2_hat is nan and must be reported as
    undefined rather than worked around.
    """
    m_ctx = np.asarray(m_ctx, float).ravel()
    y_ctx = np.asarray(y_ctx, float).ravel()
    n = len(y_ctx)
    dof = float(n - trJ)
    if dof <= 0:
        return float("nan"), dof, False
    return float(np.sum((m_ctx - y_ctx) ** 2) / dof), dof, True


# ---------------------------------------------------------------------------
# The two variance forms
# ---------------------------------------------------------------------------

#: floor applied to any variance estimate before it is used, as a multiple of
#: sigma2_hat. Stated wherever a clip rate is reported.
CLIP_FLOOR_FRAC = 1e-3


def s2_from_Jstar(sigma2, J_star, form="inverted", floor_frac=CLIP_FLOOR_FRAC):
    """Predictive variance at a query from the appended-query Jacobian diagonal.

    DERIVATION, and why two forms exist.
    ------------------------------------
    T2 applied to the AUGMENTED context (x_*, y_*) appended, at the appended
    row, is exact for every prior:

        sigma^2 J_**  =  Var(f_* | y, y_*)                                 (1)

    Note the conditioning set: it includes the hypothetical label y_* we just
    invented. The quantity the experiment wants is the predictive variance at a
    fresh observation given the ORIGINAL context,

        s^2(x_*)  =  Var(f_* | y) + sigma^2                                 (2)

    and (1) is strictly smaller than Var(f_*|y), because it conditions on one
    extra observation. So `sigma2 * (1 + J_**)`, the brief's form, is not (2).

    Under a Gaussian posterior, writing v = Var(f_*|y), the update from
    observing y_* = f_* + eps with eps ~ N(0, sigma^2) is

        Var(f_* | y, y_*) = v - v^2/(v + sigma^2) = v sigma^2/(v + sigma^2)

    so (1) gives J_** = v/(v + sigma^2), which inverts to v = sigma^2
    J_**/(1 - J_**), and therefore

        s^2(x_*) = v + sigma^2 = sigma^2 / (1 - J_**)                       (3)

    (3) is exact for the exact GP -- machine precision in chunk 1 -- while the
    brief's form carries 12-17% error there and, since J_** in [0,1) for a
    Bayesian, is bounded above by 2 sigma^2 no matter how uncertain the query
    is. The inversion step assumes the posterior at the appended point is
    Gaussian; where it is not, the error is a third-cumulant term (T3), which
    is what the hierarchical GP in chunk 1.3 measures.

    form="specified" returns sigma2 * (1 + J_**), the brief's form, reported
    throughout beside the inverted one.
    form="inverted"  returns sigma2 / (1 - J_**), used as the primary.

    Returns (s2_clipped, clipped_mask, s2_raw).
    """
    sigma2 = float(sigma2)
    J_star = np.asarray(J_star, float)
    if form == "specified":
        raw = sigma2 * (1.0 + J_star)
    elif form == "inverted":
        denom = 1.0 - J_star
        raw = np.where(denom > 0, sigma2 / np.where(denom > 0, denom, 1.0), -np.inf)
    else:
        raise ValueError("form must be 'specified' or 'inverted'")
    floor = floor_frac * sigma2
    clipped_mask = ~(raw >= floor)          # catches <= floor, nan and -inf
    s2 = np.where(clipped_mask, floor, raw)
    return s2, clipped_mask, raw


# ---------------------------------------------------------------------------
# Split conformal
# ---------------------------------------------------------------------------


def conformal_halfwidth(residuals_cal, alpha):
    """Absolute-residual split-conformal half-width at level 1 - alpha.

    Uses the finite-sample-valid rank ceil((n+1)(1-alpha)); returns inf when
    that rank exceeds n, which is the honest answer for a calibration set too
    small for the level.
    """
    r = np.sort(np.abs(np.asarray(residuals_cal, float).ravel()))
    n = len(r)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(r[k - 1])


# ---------------------------------------------------------------------------
# Interval metrics
# ---------------------------------------------------------------------------


def z_for(level):
    return float(stats.norm.ppf(0.5 + level / 2.0))


def gaussian_interval(mean, s2, level):
    s = np.sqrt(np.maximum(np.asarray(s2, float), 0.0))
    z = z_for(level)
    return np.asarray(mean, float) - z * s, np.asarray(mean, float) + z * s


def coverage(y, lo, hi):
    y = np.asarray(y, float).ravel()
    return float(np.mean((y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))))


def interval_score(y, lo, hi, alpha):
    """Winkler / negatively-oriented interval score at nominal 1 - alpha.

    width + (2/alpha)(lo - y) if y < lo + (2/alpha)(y - hi) if y > hi.
    Lower is better; penalises width and miss jointly.
    """
    y = np.asarray(y, float).ravel()
    lo = np.asarray(lo, float).ravel()
    hi = np.asarray(hi, float).ravel()
    return (hi - lo
            + (2.0 / alpha) * np.maximum(0.0, lo - y)
            + (2.0 / alpha) * np.maximum(0.0, y - hi))


def gaussian_nll(y, mean, s2, floor=1e-12):
    y = np.asarray(y, float).ravel()
    mean = np.asarray(mean, float).ravel()
    s2 = np.maximum(np.asarray(s2, float).ravel(), floor)
    return 0.5 * (np.log(2 * np.pi * s2) + (y - mean) ** 2 / s2)


def spearman(a, b):
    """Rank correlation, nan when either input is constant (zero rank variance)."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def paired_bootstrap_ci(diff, n_boot=10000, level=0.95, seed=0):
    """Percentile bootstrap CI for the mean of a paired difference vector."""
    d = np.asarray(diff, float).ravel()
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": int(len(d))}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    return {"mean": float(d.mean()), "lo": float(np.quantile(means, a)),
            "hi": float(np.quantile(means, 1 - a)), "n": int(len(d))}


def wilcoxon(diff):
    d = np.asarray(diff, float).ravel()
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) < 6:
        return {"stat": float("nan"), "p": float("nan"), "n": int(len(d))}
    r = stats.wilcoxon(d)
    return {"stat": float(r.statistic), "p": float(r.pvalue), "n": int(len(d))}


# ---------------------------------------------------------------------------
# JSON helper -- every number in this experiment lands in a JSON file
# ---------------------------------------------------------------------------


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o
