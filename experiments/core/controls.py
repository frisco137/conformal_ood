"""Canonical control maps: the affine normaliser, and the two imitators.

THIS IS THE ONLY DEFINITION OF THESE OBJECTS. Import them from here.

Why this module exists
----------------------
``TargetedImitator`` is the load-bearing control of the whole instrument
argument. It is a map that is *known* to be Bayes-impossible by construction,
wrapped in the same normaliser the real models wear, and its job is to prove
that the projection ``Q^T . Q`` transmits a violation rather than laundering
it (E1.2(b), C-I3). If that control is wrong, no statement about any model is
licensed.

It shipped with a wrong closed form. ``inner`` applies

    -(c/2) * <v, u> * v          ->  Jacobian contribution  -(c/2) v v^T

while ``inner_jacobian`` returned ``-c * outer(v, v)``: twice the rank-one
term. The map itself was always correct; only the analytic reference it is
checked against was wrong, and only in the rank-one term.

The bug was found independently at least three times and worked around inline
in three separate scripts, each re-deriving ``-(c/2) * outer(v, v)`` by hand:

    experiments/tier0_instrument/chunk3_dither_controls.py
    experiments/phase_2/src/chunk4_aux.py
    experiments/phase_2/src/exp22_analyse.py

and was NOT worked around in ``chunk1_control_validation.run_1a``, which is the
one place that wrote a number to disk. See ``QUARANTINE.md`` for what that
number was and what supersedes it.

Consolidating here removes the possibility of a fourth divergence.
``test_controls.py`` finite-differences ``inner`` against ``inner_jacobian`` on
every control in this module, so the two cannot drift apart again without a
test failing.

Configuration constants are the audit's standing values and must not be
changed without re-running every control in FINAL_NUMBERS.md section 2.
"""
from __future__ import annotations

import numpy as np

from .surrogates import ExactGP

#: Noise SD of the exact-GP control. The audit's standing value.
GP_SIGMA = 0.5

#: TabPFN v2's measured median output jump, used as the quantisation step of
#: the artifact control. diagnostic_a_results.json.
DELTA = 6.87e-4


# --------------------------------------------------------------- the wrapper

def wrap(g):
    """Affine context-statistic normaliser, N1: ``m(y) = ybar + s_y * g(u)``.

    This is the confound N1 says contributes an O(1) asymmetric rank-one term
    to the ambient Jacobian, and that ``Q^T J Q = Q^T G Q`` removes exactly.
    """
    def predict(y_eval):
        mu, sd = np.mean(y_eval), np.std(y_eval) + 1e-8
        return mu + sd * np.asarray(g((y_eval - mu) / sd)).flatten()
    return predict


def quantize(f, delta):
    """Round the output of ``f`` to a grid of spacing ``delta``.

    Used to build the artifact floor: a map whose true asym, negeig and
    curvature are all exactly zero, so every nonzero value it returns is
    manufactured by output quantisation.
    """
    return lambda y_eval: np.round(np.asarray(f(y_eval)).flatten() / delta) * delta


# --------------------------------------------------------------- the imitators

class ImitatorWrapped:
    """Saturating imitator: ``egp + eps*sin(<a,u>/eps - phase - pi)*a``.

    Phase-tuned so ``cos(.) = -1`` exactly at ``u = y_std``, giving inner
    Jacobian ``G = W - a a^T``. ``||a||^2 ~ 100`` dwarfs ``||W||_2 <= 1``, so
    negeig saturates near 1.
    """

    def __init__(self, X, y, eps=0.01):
        self.egp = ExactGP(X, sigma=GP_SIGMA)
        n = len(y)
        a = np.random.RandomState(42).randn(n)
        self.a = a - np.mean(a)
        self.eps = eps
        u = (y - np.mean(y)) / (np.std(y) + 1e-8)
        self.phase = np.dot(self.a, u) / eps

    def inner(self, u):
        s = np.sin(np.dot(self.a, u) / self.eps - self.phase - np.pi)
        return self.egp.predict(u) + self.eps * s * self.a

    def inner_jacobian(self, u):
        # d/du [ eps * sin(<a,u>/eps + k) * a ] = cos(<a,u>/eps + k) * a a^T.
        # The eps cancels against the 1/eps from the chain rule; this is the
        # T1 construction's whole point -- an O(eps) perturbation in value with
        # an O(1) perturbation in derivative.
        c = np.cos(np.dot(self.a, u) / self.eps - self.phase - np.pi)
        return self.egp.jacobian(u) + c * np.outer(self.a, self.a)

    @property
    def predict(self):
        return wrap(self.inner)


class TargetedImitator:
    """Imitator tuned to a prescribed asym/negeig, with a closed-form Jacobian.

    ``inner(u) = W u + M_anti u - (c/2) <v,u> v`` with ``M_anti`` antisymmetric,
    so the closed form is

        G = W + M_anti - (c/2) v v^T

    The rank-one term carries the A2 violation and the antisymmetric term
    carries the A1 violation, both at a size fixed in advance.
    """

    def __init__(self, X, y, seed=42):
        self.egp = ExactGP(X, sigma=GP_SIGMA)
        J_gp = self.egp.jacobian(y)
        n = len(y)
        rng = np.random.RandomState(seed)
        M = rng.randn(n, n)
        M_anti = (M - M.T) / 2
        target = np.sqrt(0.09 / 0.91) * np.linalg.norm(J_gp, 'fro')
        self.M_anti = M_anti * (target / np.linalg.norm(M_anti, 'fro'))
        v = rng.randn(n)
        self.v = v / np.linalg.norm(v)
        lam_min = np.min(np.linalg.eigvalsh((J_gp + J_gp.T) / 2))
        self.c = 0.8 * np.linalg.norm(J_gp, 2) + max(0.0, lam_min)

    def inner(self, u):
        return (self.egp.predict(u) + self.M_anti @ u
                - (self.c / 2) * np.dot(self.v, u) * self.v)

    def inner_jacobian(self, u):
        # CORRECTED. This returned -c * outer(v, v) until 2026-08-26: twice the
        # rank-one term `inner` actually applies. The map was never wrong, only
        # this reference. See the module docstring.
        return (self.egp.jacobian(u) + self.M_anti
                - (self.c / 2) * np.outer(self.v, self.v))

    @property
    def predict(self):
        return wrap(self.inner)


# ------------------------------------------------------------------ self-check

def finite_difference_jacobian(f, u, h=1e-6):
    """Plain central-difference Jacobian of ``f`` at ``u``, for the self-check.

    Deliberately NOT core.metrics.central_jacobian: this module must be able to
    validate itself without depending on the instrument it is used to validate.
    """
    u = np.asarray(u, float).ravel()
    n = len(u)
    J = np.empty((n, n))
    for j in range(n):
        up = u.copy(); up[j] += h
        um = u.copy(); um[j] -= h
        J[:, j] = (np.asarray(f(up)).ravel() - np.asarray(f(um)).ravel()) / (2 * h)
    return J


def check_inner_jacobian(control, u, h=1e-6):
    """Relative Frobenius error of ``inner_jacobian`` against finite differences.

    Returns a float. This is the check that keeps the closed form and the map
    from diverging; ``test_controls.py`` asserts it stays small.
    """
    J_fd = finite_difference_jacobian(control.inner, u, h=h)
    J_cf = np.asarray(control.inner_jacobian(u), float)
    return float(np.linalg.norm(J_fd - J_cf, 'fro') / np.linalg.norm(J_cf, 'fro'))
