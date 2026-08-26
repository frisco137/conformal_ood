"""Regression tests for the control maps.

The point of this file is one assertion: every control's ``inner_jacobian`` is
the actual derivative of its ``inner``. ``TargetedImitator`` shipped a closed
form that was twice the true rank-one term for months, was independently
rediscovered and worked around inline in three separate scripts, and in the one
place it was not worked around it wrote a wrong number to disk
(``chunk1_results.json -> 1A.targeted_imitator.analytic``).

The imitator is the control that proves the projection transmits violations
rather than laundering them. A wrong closed form there undermines the whole
instrument argument, so it gets a test rather than a comment.

Run:  .venv/bin/python -m pytest experiments/core/test_controls.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.core.context import generate_audit_context
from experiments.core.controls import (
    DELTA, GP_SIGMA, ImitatorWrapped, TargetedImitator, check_inner_jacobian,
    quantize, wrap,
)
from experiments.core.metrics import get_Q, negeig
from experiments.core.surrogates import ExactGP

SEEDS = [42, 100, 200, 300, 400]


def _context(seed):
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
    u = (y - np.mean(y)) / (np.std(y) + 1e-8)
    return X, y, u


@pytest.mark.parametrize("seed", SEEDS)
def test_targeted_imitator_closed_form_is_the_derivative(seed):
    """The regression this file exists for."""
    X, y, u = _context(seed)
    err = check_inner_jacobian(TargetedImitator(X, y, seed=seed), u)
    assert err < 1e-7, f"inner_jacobian is not the derivative of inner: rel err {err:.3e}"


@pytest.mark.parametrize("seed", SEEDS)
def test_imitator_wrapped_closed_form_is_the_derivative(seed):
    X, y, u = _context(seed)
    # eps = 0.01 means the map oscillates on a 1/eps scale, so the finite
    # difference needs a correspondingly small step to be truncation-limited.
    err = check_inner_jacobian(ImitatorWrapped(X, y), u, h=1e-9)
    assert err < 1e-5, f"inner_jacobian is not the derivative of inner: rel err {err:.3e}"


@pytest.mark.parametrize("seed", SEEDS)
def test_the_specific_doubling_bug_would_be_caught(seed):
    """The old -c*vv^T form must fail the check the fixed form passes.

    Without this, a test that only asserts the correct form passes would still
    pass if someone made the check itself vacuous.
    """
    X, y, u = _context(seed)
    ti = TargetedImitator(X, y, seed=seed)

    class _Buggy:
        inner = ti.inner

        @staticmethod
        def inner_jacobian(uu):
            return ti.egp.jacobian(uu) + ti.M_anti - ti.c * np.outer(ti.v, ti.v)

    assert check_inner_jacobian(_Buggy(), u) > 1e-3


@pytest.mark.parametrize("seed", SEEDS)
def test_imitator_violates_a2_through_the_projection(seed):
    """E1.2(b) / C-I3: the projection must not launder the violation away.

    This is the control's actual job. Threshold is the plan's `10 x A_floor`
    with the measured reduced-basis floor of ~3.1e-12; 1e-3 is far stricter and
    still passed by an order of magnitude.
    """
    X, y, u = _context(seed)
    ti = TargetedImitator(X, y, seed=seed)
    Q = get_Q(y, seed=0)
    assert negeig(Q.T @ ti.inner_jacobian(u) @ Q) > 1e-3


def test_wrapper_is_shift_and_scale_equivariant():
    """N1: the normaliser makes any inner map degree-1 homogeneous and
    shift-equivariant, which is what licenses the projection."""
    X, y, _ = _context(42)
    m = wrap(ExactGP(X, sigma=GP_SIGMA).predict)
    base = m(y)
    shift = m(y + 3.0) - 3.0
    scale = m(2.0 * y) / 2.0
    assert np.max(np.abs(shift - base)) / np.linalg.norm(base) < 1e-12
    assert np.max(np.abs(scale - base)) / np.linalg.norm(base) < 1e-12


def test_quantize_lands_on_the_grid():
    X, y, _ = _context(42)
    q = quantize(wrap(ExactGP(X, sigma=GP_SIGMA).predict), DELTA)(y)
    assert np.max(np.abs(q / DELTA - np.round(q / DELTA))) < 1e-9
