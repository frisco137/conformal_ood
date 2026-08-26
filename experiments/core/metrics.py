import numpy as np

def asym(J):
    """
    asym(J) = ||J - J.T||_F / ||J||_F
    This precisely matches the C-N1 definition.
    """
    norm_denom = np.linalg.norm(J, ord='fro')
    if norm_denom < 1e-12:
        return 0.0
    return np.linalg.norm(J - J.T, ord='fro') / norm_denom

def negeig(J):
    """
    negeig(J) = -min(0, lambda_min(sym J)) / ||J||_2
    """
    sym_J = (J + J.T) / 2
    vals = np.linalg.eigvalsh(sym_J)
    min_val = np.min(vals)
    if min_val >= 0:
        return 0.0
    norm_J = np.linalg.norm(J, ord=2)
    if norm_J < 1e-12:
        return 0.0
    return -min_val / norm_J

def central_jacobian(model_func, y, h=1e-4):
    n = len(y)
    J = np.zeros((n, n), dtype=y.dtype)
    for j in range(n):
        y_plus = y.copy()
        y_plus[j] += h
        m_plus = model_func(y_plus)
        
        y_minus = y.copy()
        y_minus[j] -= h
        m_minus = model_func(y_minus)
        
        J[:, j] = (m_plus - m_minus) / (2 * h)
    return J

# An UNSEEDED get_Q used to be defined here. It was dead code -- the seeded
# definition below shadowed it at import time, so nothing ever called it -- but
# it was the first one a reader met going down the file, and an unseeded get_Q
# is exactly the defect that made 12 earlier copies of this function across the
# tree non-reproducible (FINAL_NUMBERS.md 1.1). Removed 2026-08-26. This changes
# no behaviour: the shadowing definition is unchanged and is the only get_Q that
# was ever reachable. Use get_Q(y, seed=...) below.

def extract_variance(quantiles, levels):
    """
    Extracts the predictive variance via numerical integration of the 
    second moment over the quantile grid, instead of a tail spread.
    Levels represent probabilities p in [0, 1].
    Mean = int_0^1 Q(p) dp
    Second moment = int_0^1 Q(p)^2 dp
    Variance = E[X^2] - (E[X])^2
    """
    mean_val = np.trapz(quantiles, x=levels, axis=-1)
    second_moment = np.trapz(quantiles**2, x=levels, axis=-1)
    # Ensure non-negative due to numerical precision
    var = second_moment - mean_val**2
    return np.maximum(var, 0.0)


# =============================================================================
# CANONICAL INSTRUMENT  (added Chunk 1, control-validation run)
#
# Definitions are fixed by theory_and_claims.md. Every quantity below has
# exactly one implementation and every script must call it from here.
#
#   asym(J)    = ||J - J^T||_F / ||J||_F              (above, line 3)
#   negeig(J)  = -min(0, lam_min(sym J)) / ||J||_2    (above, line 12)
#   negfrac(J) = #{lam_i(sym J) < 0} / dim
#   second_difference_norm(...)  -- the quantity previously called "curvature"
#   profile_jacobian(J, mode)    -- returns post-profile asym AND a real residual
#
# Superseded implementations, do not use:
#   ||(J - J^T)/2||_F / ||J||_F   -- half of canonical asym; present in every
#                                    tier2_audit/ script.
#   instrument_c_d.py:89          -- returns negfrac under the name negeig.
#   batch4_lastrun.py:69          -- returns sum(neg eigs)/sum(eigs) as negfrac.
#   item3.py:95                   -- residual is always 0.0 (lstsq returns an
#                                    empty residual array on a rank-deficient
#                                    design matrix).
# =============================================================================


class RetractedMetricWarning(UserWarning):
    """Raised as a warning when a retracted metric is computed."""


def negfrac(J, rel_tol=1e-10, acknowledge_retracted=False):
    """RETRACTED. Fraction of eigenvalues of sym(J) that are negative.

    ***  DO NOT REPORT THIS AS A MODEL RESULT.  ***

    It measures the rank deficiency of the CONTEXT, not a property of the model.
    generate_audit_context duplicates the first 10 rows of X exactly, so K is
    rank-deficient and W = K(K+s2 I)^-1 inherits a null space: the exact GP's
    reduced spectrum has nine eigenvalues at machine zero and then a hard gap to
    2.2e-2, and any noise tips them negative. The quantised control and TabICL v2
    both read 9/98 = 0.0918367 -- on every seed, at every amplitude, in every
    dither configuration. It does not respond to the probe at all.

    FINAL_NUMBERS.md section 8 lists it under "explicitly not measured";
    section 1.3 lists the files carrying retracted values.

    The function is KEPT, not quarantined, because FINAL_NUMBERS.md section 7.4
    uses it as evidence for its own retraction: it verifies N3 by showing negfrac
    pinned at 0.0918367 while negeig falls by a factor of 106 over the same
    matrices. Removing it would make that section unreproducible.

    Pass acknowledge_retracted=True to silence the warning. The return value is
    identical either way -- no recorded number changes.

    Use negeig instead for any positivity statement. It weights by magnitude and
    is the canonical A2 quantity.
    """
    if not acknowledge_retracted:
        import warnings
        warnings.warn(
            "negfrac is RETRACTED: it reads the audit context's rank deficiency "
            "(9/98 on the control and on TabICL alike), not the model. Use negeig "
            "for A2. See experiments/_quarantine/QUARANTINE.md. Pass "
            "acknowledge_retracted=True if you are reproducing FINAL_NUMBERS 7.4.",
            RetractedMetricWarning, stacklevel=2,
        )
    sym_J = (J + J.T) / 2
    eigs = np.linalg.eigvalsh(sym_J)
    scale = np.linalg.norm(J, ord=2)
    if scale < 1e-12:
        return 0.0
    return float(np.sum(eigs < -rel_tol * scale) / len(eigs))


def second_difference_norm(m_plus, m_base, m_minus, t):
    """||m(y+tq) - 2m(y) + m(y-tq)||_2 / (t^2 ||m(y)||_2).

    NOT a signed directional second derivative, and NOT the M0 tangential
    great-circle curvature with geodesic correction. It is the norm of a
    normalised second difference, and at small t it is dominated by output
    quantisation as eps/t^2. Always report it with the amplitude t and, for a
    quantised map, alongside a t-sweep.
    """
    denom = t**2 * np.linalg.norm(m_base)
    if denom < 1e-300:
        return 0.0
    return float(np.linalg.norm(m_plus - 2.0 * m_base + m_minus) / denom)


def get_Q(y, seed=0):
    """Orthonormal basis of span{1, u}^perp, u = y - mean(y). Shape (n, n-2).

    Seeded. The previous 12 copies of this function across the tree used an
    unseeded np.random.randn, so no measurement was reproducible.

    asym, negeig, negfrac and the spectrum are invariant to the choice of basis
    within the subspace (Q' = Q R gives Q'^T J Q' = R^T (Q^T J Q) R for
    orthogonal R). profile_jacobian is NOT invariant, since it reads individual
    entries -- which is the reason this must be seeded.
    """
    n = len(y)
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    A[:, 0] = np.ones(n)
    A[:, 1] = y - np.mean(y)
    Q_full, _ = np.linalg.qr(A)
    return Q_full[:, 2:]


def reduced_jacobian(predict, y, Q, t, dither=False, N=10,
                     dither_halfwidth=None, seed=0):
    """Central-difference Jacobian probed along the columns of Q, projected.

    Returns (J_reduced, curvatures, m_base) with J_reduced = Q^T J Q of shape
    (n-2, n-2) and curvatures a list of second_difference_norm per direction.

    dither_halfwidth is the HALF-width of the uniform jitter: noise is drawn
    from U[-dither_halfwidth, +dither_halfwidth]. It is a required argument
    when dither=True and has no default, because the tree previously contained
    two conventions -- U[-delta, delta] in batch1/batch2/batch3/item2/item3 and
    U[-delta/2, delta/2] in the control scripts -- which meant TabPFN's headline
    run and the control built to validate it were dithered at widths differing
    by a factor of two.
    """
    if dither and dither_halfwidth is None:
        raise ValueError("dither_halfwidth is required when dither=True")

    n, m_dim = Q.shape

    def _eval(y_eval, j):
        if not dither:
            return np.asarray(predict(y_eval)).flatten()
        rng = np.random.RandomState(seed + j)
        noise = rng.uniform(-dither_halfwidth, dither_halfwidth, (N, n))
        acc = np.zeros(n)
        for k in range(N):
            acc += np.asarray(predict(y_eval + noise[k])).flatten()
        return acc / N

    m_base = _eval(y, 0)

    J_cols, curvatures = [], []
    for j in range(m_dim):
        q_j = Q[:, j]
        m_plus = _eval(y + t * q_j, j)
        m_minus = _eval(y - t * q_j, j)
        J_cols.append((m_plus - m_minus) / (2.0 * t))
        curvatures.append(second_difference_norm(m_plus, m_base, m_minus, t))

    J = Q.T @ np.column_stack(J_cols)
    return J, curvatures, m_base


def profile_jacobian(J, mode, weighted=False):
    """Fit the best diagonal rescaling that would make J symmetric, in log space.

    mode='column'  heteroscedastic Bayes (N6):  J_ij s_j = J_ji s_i,
                   tested on J @ diag(s).
    mode='row'     Nadaraya-Watson (N5):        d_i J_ij = d_j J_ji,
                   tested on diag(d) @ J.

    Returns dict with the post-profile canonical asym, the relative residual of
    the log-space least-squares fit, the recovered profile, and bookkeeping.

    The design matrix has one +1 and one -1 per row, so its columns sum to zero
    and it is rank-deficient by construction (rank <= n-1). numpy returns an
    EMPTY residuals array in that case, which is why item3.py:95 reported
    0.00e+00 for every fit. The residual here is computed explicitly.
    """
    if mode not in ("column", "row"):
        raise ValueError("mode must be 'column' or 'row'")

    n = J.shape[0]
    rows, rhs, wts, pairs_used = [], [], [], 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = J[i, j], J[j, i]
            if abs(a) <= 1e-12 or abs(b) <= 1e-12:
                continue
            r = np.zeros(n)
            r[i], r[j] = 1.0, -1.0
            rows.append(r)
            if mode == "column":
                rhs.append(np.log(abs(a)) - np.log(abs(b)))
            else:
                rhs.append(np.log(abs(b)) - np.log(abs(a)))
            wts.append(abs(a) * abs(b))
            pairs_used += 1

    total_pairs = n * (n - 1) // 2
    if pairs_used == 0:
        return {"asym": float(asym(J)), "residual_rel": float("nan"),
                "residual_abs": float("nan"), "pairs_used": 0,
                "pairs_total": total_pairs, "profile": None}

    A = np.array(rows)
    b = np.array(rhs)
    w = np.array(wts)

    # Unweighted, every pair counts the same, is dominated by near-zero entries:
    # a pair with |J_ij| at the noise floor contributes an O(1) log-ratio. The
    # weighted form scales each equation by sqrt(|J_ij J_ji|) so the fit is
    # driven by the entries that carry the matrix.
    if weighted:
        sw = np.sqrt(w)
        Af, bf = A * sw[:, None], b * sw
    else:
        Af, bf = A, b

    log_s, _, rank, _ = np.linalg.lstsq(Af, bf, rcond=None)
    log_s -= np.mean(log_s)                     # gauge fix: geometric mean 1
    s = np.exp(log_s)

    residual_abs = float(np.linalg.norm(Af @ log_s - bf))
    norm_b = float(np.linalg.norm(bf))
    residual_rel = residual_abs / norm_b if norm_b > 0 else float("nan")

    J_scaled = J * s[np.newaxis, :] if mode == "column" else J * s[:, np.newaxis]

    return {"asym": float(asym(J_scaled)),
            "residual_rel": residual_rel,
            "residual_abs": residual_abs,
            "pairs_used": pairs_used,
            "pairs_total": total_pairs,
            "design_rank": int(rank),
            "profile": s}
