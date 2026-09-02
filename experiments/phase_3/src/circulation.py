"""T0.3 -- the circulation instrument. Symmetry measured by forward passes only.

THE IDENTITY
------------
On a loop of constant mean and constant centred radius -- a circle in the sphere
spanned by orthonormal a, b, both orthogonal to 1 --

    y(t) = ybar*1 + r*(cos t * a + sin t * b),      t in [0, 2pi)
    I    = contour_integral m(y) . dy               (forward passes only)

For an affine map m(y) = J y + c, carrying the constant term to zero around the
closed loop:

    dy/dt = r(-sin t * a + cos t * b)
    m(y).dy/dt = r^2 [ -cos t sin t (a'Ja) + cos^2 t (b'Ja)
                       - sin^2 t (a'Jb) + sin t cos t (b'Jb) ]
    integral over [0,2pi):  cos*sin -> 0,  cos^2 -> pi,  sin^2 -> pi

    I = pi r^2 (b'Ja - a'Jb) = -pi r^2 * a'(J - J') b

NOTE THE SIGN. The plan states I/(pi r^2) = a'(J-J')b; with the orientation above
the derivation gives the NEGATIVE of that. The sign is a convention fixed by loop
orientation, not a result. `circulation_asym` returns the value with the sign
pinned so that it matches a'(J-J')b, and `test_circulation.py` asserts it against
an analytic J so the convention cannot drift.

WHY THIS IS THE RIGHT INSTRUMENT FOR PHASE 3
--------------------------------------------
* The normaliser confound vanishes identically. m(y) = ybar*1 + s_y*g(u) has both
  ybar and s_y CONSTANT along such a loop, so N1's O(1) asymmetric rank-one term
  contributes nothing and no projection is needed.
* The unidentifiable sector is excluded by construction: the loop cannot move
  along 1, and (when a, b are also drawn orthogonal to the centred y) not along
  y either.
* Quantisation-immune. An output quantum enters ADDITIVELY under the integral and
  averages down as 1/sqrt(N), instead of being amplified by 1/t the way a central
  difference amplifies it.
* Frobenius recovery from random planes -- see `frobenius_from_planes`.

FOR A NONLINEAR MAP the identity is not exact: by Stokes, I/(pi r^2) is the
area-average of the antisymmetric 2-form over the enclosed disc. That is a
feature (it is a genuine average of the local asymmetry, not a point estimate)
but it means loop and finite-difference values are not required to agree to
machine precision on a nonlinear map -- only to within the field's curvature over
the disc. P1's 20% tolerance is set for that reason.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "centred_basis", "loop_points", "circulation_asym", "frobenius_from_planes",
    "FROBENIUS_CONST",
]

#: E[(a'Ab)^2] = ||A||_F^2 / (m(m-1)) for antisymmetric A and random orthonormal
#: a, b in an m-dimensional space. Derivation: a'Ab = (O'AO)_{12} for a random
#: orthogonal O; O'AO is antisymmetric with the same Frobenius norm, its m(m-1)
#: off-diagonal entries are exchangeable, and ||A||_F^2 is their sum of squares.
#:
#: The plan states 2||A||_F^2/(m(m-1)). That is a factor of 2 too large;
#: `test_circulation.py` checks the constant numerically rather than trusting
#: either statement. Deviation logged in phase_3_results.md section 3.
FROBENIUS_CONST = 1.0


def centred_basis(n, rng, exclude=None):
    """Random orthonormal pair (a, b), both orthogonal to 1 and to `exclude`.

    `exclude` is normally the centred label vector y - mean(y). Excluding it as
    well as 1 puts the loop entirely inside the subspace the projection Q keeps,
    which is what makes the loop blind to the rank-one noise-scale contamination
    of T3.1 as well as to the normaliser.
    """
    n = int(n)
    basis = [np.ones(n)]
    if exclude is not None:
        e = np.asarray(exclude, float).ravel()
        e = e - e.mean()
        if np.linalg.norm(e) > 1e-12:
            basis.append(e)
    B = np.column_stack(basis)
    # projector onto the complement of span(basis)
    Qb, _ = np.linalg.qr(B)
    P = np.eye(n) - Qb @ Qb.T

    out = []
    for _ in range(2):
        v = P @ rng.standard_normal(n)
        for w in out:
            v -= (v @ w) * w
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            raise ValueError("degenerate centred basis draw")
        out.append(v / nv)
    return out[0], out[1]


def loop_points(y_bar, r, a, b, N):
    """The N quadrature points of the loop, shape (N, n)."""
    th = 2.0 * np.pi * np.arange(N) / N
    n = len(a)
    return (y_bar * np.ones((N, n))
            + r * (np.cos(th)[:, None] * a[None, :]
                   + np.sin(th)[:, None] * b[None, :])), th


def circulation_asym(predict, y_bar, r, a, b, N=64, return_raw=False):
    """a'(J - J')b, estimated from N forward passes on a closed loop.

    `predict` maps a length-n label vector to a length-n prediction vector.

    Quadrature is the trapezoidal rule on a periodic integrand, which converges
    spectrally -- doubling N is far more effective here than it would be on an
    open interval. Returns the value with the sign pinned to a'(J-J')b.
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    pts, th = loop_points(y_bar, r, a, b, N)
    dydt = r * (-np.sin(th)[:, None] * a[None, :] + np.cos(th)[:, None] * b[None, :])

    integrand = np.empty(N)
    preds = np.empty((N, len(a)))
    for k in range(N):
        m = np.asarray(predict(pts[k]), float).ravel()
        preds[k] = m
        integrand[k] = m @ dydt[k]

    I = float(integrand.sum() * (2.0 * np.pi / N))    # trapezoid == mean * 2pi
    val = -I / (np.pi * r ** 2)                        # sign pinned; see docstring
    if return_raw:
        return val, {"I": I, "integrand": integrand, "preds": preds, "points": pts}
    return val


def frobenius_from_planes(predict, y_bar, r, n, n_planes=40, N=64, seed=0,
                          exclude=None, return_samples=False):
    """Recover ||J - J'||_F restricted to the centred subspace, from random planes.

    ||A||_F^2 = m(m-1) * E[(a'Ab)^2], with m the dimension of the subspace the
    planes are drawn from: n-1 when only 1 is excluded, n-2 when the centred y is
    excluded too.
    """
    rng = np.random.default_rng(seed)
    m_dim = n - 1 - (1 if exclude is not None else 0)
    vals = np.empty(n_planes)
    for i in range(n_planes):
        a, b = centred_basis(n, rng, exclude=exclude)
        vals[i] = circulation_asym(predict, y_bar, r, a, b, N=N)
    est = float(np.sqrt(max(m_dim * (m_dim - 1) * np.mean(vals ** 2)
                            / FROBENIUS_CONST, 0.0)))
    if return_samples:
        return est, vals, m_dim
    return est
