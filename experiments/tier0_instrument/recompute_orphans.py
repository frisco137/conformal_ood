"""Recompute the orphaned FINAL_NUMBERS entries from the tracked arrays.

Ten result files under tier0_instrument/ carried FINAL_NUMBERS entries with no
producing script -- they were written in-session and the code was never saved
(FINAL_NUMBERS.md section 9.3, last row). This script is that missing code.

    section  quantity                          recorded in
    2.5      seven stress tests                exp_stress_tests.json
    2.8      row-sum residual r_1              clarif_item4_6.json
    2.9      ambient floor                     clarif_item4_6.json
    3.4/4.2  artifact floor brackets           exp4_floor_brackets.json
    3.6      E1.4 paired, TabICL RoPE          e14_paired.json      <- NOT RECOMPUTABLE
    4.3      leading negative eigenvector      exp_negeigvec.json
    4.3      #(J_ii < 0)                       exp1_a3_final.json / clarif_item4_6.json
    5.2/5.3  A3 regressions                    exp1_a3_final.json
    5.8      no-intercept A3 fits              clarif_item5.json
    7.3      N1(ii) projection exactness       exp_n1ii.json

WHY THIS RUNS WITHOUT A GPU
---------------------------
Every model-dependent quantity above is a pure function of a Jacobian that is
already on disk and tracked:

    exp12_ambient_jacobians.npz   ambient 100x100 J, y, and s2, for
                                  tabicl_v2 / tabpfn_v2_{dither,nodither} / tabswift
    exp3_reduced_jacobians.npz    reduced 98x98 Q^T J Q for tabpfn / tabswift
    exp4_tabicl_reduced.npz       reduced 98x98 for tabicl

The control quantities (2.5, 2.9, 3.4/4.2, 7.3) are analytic NumPy and never
needed a GPU at all.

THE ONE EXCEPTION, stated rather than approximated
--------------------------------------------------
Section 3.6, the E1.4 paired RoPE analysis, requires fresh TabICL v2 forward
passes with `model_.row_interactor.tf_row.rope = None`. No stored array can
stand in for a Jacobian of a MODIFIED model. It is reported as NOT RECOMPUTABLE
and left alone. Re-running it needs ~2000 GPU fits (5 seeds x 2 conditions x
100 columns x 2 evals), roughly an hour.

CLASSIFICATION
--------------
Every recomputed value is compared against the record and labelled:

    IDENTICAL   exact float equality
    AGREES      relative difference <= 1e-12   (same arithmetic, different order)
    CLOSE       relative difference <= 1e-6
    DIFFERS     anything else -- investigate, do not average

Quantities whose original CONSTRUCTION is not recoverable from the record (some
of the section 2.5 stress tests specify a random matrix whose seed was never
written down) are labelled CONSTRUCTION-UNDETERMINED and compared on order of
magnitude only, which is all those tests ever asserted.

Writes recompute_orphans.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from experiments.core.context import generate_audit_context          # noqa: E402
from experiments.core.controls import GP_SIGMA, quantize, wrap       # noqa: E402
from experiments.core.metrics import (                               # noqa: E402
    asym, central_jacobian, get_Q, negeig, reduced_jacobian,
)
from experiments.core.surrogates import ExactGP                      # noqa: E402

SEEDS = [42, 100, 200, 300, 400]
Q_SEED = 0
#: below this, a quantity whose true value is exactly zero is at the float64
#: floor and only its magnitude is meaningful. All such quantities in this file
#: sit at 1e-12 or smaller.
MACHINE_ZERO = 1e-9
N, D, CTX_SIGMA = 100, 5, 1.0

AMB = np.load(HERE / "exp12_ambient_jacobians.npz")
RED = np.load(HERE / "exp3_reduced_jacobians.npz")
REDICL = np.load(HERE / "exp4_tabicl_reduced.npz")


def rec(name):
    return json.load(open(HERE / f"{name}.json"))


def ctx(s):
    return generate_audit_context(n=N, d=D, sigma=CTX_SIGMA, seed=s)


# ------------------------------------------------------------- classification

def classify(new, old, undetermined=False):
    """Label one recomputed scalar against its record."""
    if old is None:
        return "NO RECORD", float("nan")
    new, old = float(new), float(old)
    if new == old:
        return "IDENTICAL", 0.0
    # Several of these quantities are instrument-error floors whose TRUE value
    # is exactly zero (a permutation-equivariance residual, a projection-
    # exactness residual). Two float64 evaluations of zero differ by whatever
    # their rounding paths differ by, and their RELATIVE difference is
    # meaningless -- dividing by a recorded 0.0 gives 1e285. Judge those on
    # absolute magnitude, which is the only claim they ever made.
    if abs(new) <= MACHINE_ZERO and abs(old) <= MACHINE_ZERO:
        return "BOTH AT MACHINE PRECISION", abs(new - old)
    denom = max(abs(old), 1e-300)
    rel = abs(new - old) / denom
    if undetermined:
        # order-of-magnitude only: both must sit in the same decade
        same_decade = (abs(new) < 1e-9 and abs(old) < 1e-9) or rel < 10.0
        return ("CONSTRUCTION-UNDETERMINED (consistent)" if same_decade
                else "CONSTRUCTION-UNDETERMINED (INCONSISTENT)"), rel
    if rel <= 1e-12:
        return "AGREES", rel
    if rel <= 1e-6:
        return "CLOSE", rel
    return "DIFFERS", rel


RESULTS = []


def check(section, quantity, new, old, undetermined=False):
    verdict, rel = classify(new, old, undetermined)
    RESULTS.append({"section": section, "quantity": quantity,
                    "recomputed": (float(new) if new is not None else None),
                    "recorded": (float(old) if old is not None else None),
                    "rel_diff": rel, "verdict": verdict})
    return verdict


# ============================================================ 2.8  r_1, 4.3 diag
def s_2_8_and_4_3_diag():
    r = rec("clarif_item4_6")
    for tag, key in [("tabicl_v2", "tabicl_v2"), ("tabswift", "tabswift"),
                     ("tabpfn_v2_dither", "tabpfn_v2_dither")]:
        ones = np.ones(N)
        for i, s in enumerate(SEEDS):
            J = AMB[f"{tag}__J__{s}"]
            r1 = float(np.linalg.norm(J @ ones - ones) / np.linalg.norm(ones))
            check("2.8", f"r_1 {tag} seed{s}", r1, r[f"{key}_r1"][i])
            nneg = int(np.sum(np.diag(J) < 0))
            check("4.3", f"#(J_ii<0) {tag} seed{s}", nneg, r[f"{key}_diag_neg"][i])


# ============================================================ 2.9 ambient floor
def s_2_9():
    r = rec("clarif_item4_6")
    for name, wrapped in [("exactgp_unwrapped_ambient", False),
                          ("exactgp_wrapped_ambient", True)]:
        for i, s in enumerate(SEEDS):
            X, y, _ = ctx(s)
            egp = ExactGP(X, sigma=GP_SIGMA)
            f = wrap(egp.predict) if wrapped else (lambda yy: egp.predict(yy))
            J = central_jacobian(f, y, h=1e-3)
            check("2.9", f"{name} asym seed{s}", asym(J), r[name]["asym"][i])
            check("2.9", f"{name} negeig seed{s}", negeig(J), r[name]["negeig"][i])
            check("2.9", f"{name} ||J||_F seed{s}",
                  float(np.linalg.norm(J, "fro")), r[name]["norm_F"][i])


# ============================================================ 7.3  N1(ii)
def s_7_3():
    r = rec("exp_n1ii")["n1ii_wrapped_vs_unwrapped_rel"]
    for i, s in enumerate(SEEDS):
        X, y, _ = ctx(s)
        egp = ExactGP(X, sigma=GP_SIGMA)
        Q = get_Q(y, seed=Q_SEED)
        # The original used the REDUCED probe, not ambient-then-project. Both
        # confirm N1(ii) exactness, but they round differently: ambient gives
        # 2.01e-12 on seed 42 where reduced gives 2.6799e-12, and the record
        # says 2.6799e-12. Identified by trying both.
        a, _, _ = reduced_jacobian(wrap(egp.predict), y, Q, 1e-3)
        b, _, _ = reduced_jacobian(lambda yy: egp.predict(yy), y, Q, 1e-3)
        check("7.3", f"N1(ii) rel seed{s}",
              float(np.linalg.norm(a - b, "fro") / np.linalg.norm(b, "fro")), r[i])


# ============================================================ 4.3  negeigvector
def s_4_3_vec():
    r = rec("exp_negeigvec")
    for tag in ["tabpfn_dither_t1e-01", "tabswift_t1e-01"]:
        for i, s in enumerate(SEEDS):
            J = RED[f"{tag}__J__{s}"]
            S = (J + J.T) / 2.0
            w, V = np.linalg.eigh(S)
            v = V[:, 0]
            X, y, _ = ctx(s)
            u = get_Q(y, seed=Q_SEED) @ v
            u = u / np.linalg.norm(u)
            wt = u ** 2
            wt = wt / wt.sum()
            top5 = list(np.argsort(wt)[::-1][:5])
            pr = float((wt.sum() ** 2) / np.sum(wt ** 2))
            rr = r[tag][i]
            check("4.3", f"{tag} lam_min seed{s}", float(w[0]), rr["lam_min"])
            check("4.3", f"{tag} participation seed{s}", pr, rr["participation_ratio"])
            check("4.3", f"{tag} top5_massfrac seed{s}",
                  float(wt[top5].sum()), rr["top5_massfrac"])
            same = [int(a) for a in top5] == [int(a) for a in rr["top5_points"]]
            RESULTS.append({"section": "4.3",
                            "quantity": f"{tag} top5_points seed{s}",
                            "recomputed": str([int(a) for a in top5]),
                            "recorded": str([int(a) for a in rr["top5_points"]]),
                            "rel_diff": 0.0 if same else 1.0,
                            "verdict": "IDENTICAL" if same else "DIFFERS"})


# ============================================================ 5.2/5.3  A3 OLS
def ols(x, y):
    Xd = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    pred = Xd @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(coef[1]), float(coef[0]), (1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def s_5_2_5_3():
    r = rec("exp1_a3_final")
    for tag in ["tabicl_v2", "tabpfn_v2_dither", "tabpfn_v2_nodither"]:
        for i, s in enumerate(SEEDS):
            J = AMB[f"{tag}__J__{s}"]
            key = f"{tag}__s2__{s}"
            if key not in AMB:
                continue
            s2 = AMB[key]
            y = AMB[f"{tag}__y__{s}"]
            Jd = np.diag(J)
            slope, intercept, r2 = ols(1.0 + Jd, s2)
            s_y = float(np.std(y))
            rr = r[tag][i]
            check("5.2/5.3", f"{tag} slope seed{s}", slope, rr["slope"])
            check("5.2/5.3", f"{tag} intercept seed{s}", intercept, rr["intercept"])
            check("5.2/5.3", f"{tag} R2 seed{s}", r2, rr["r2"])
            check("5.2/5.3", f"{tag} sigma2_hat seed{s}", slope / s_y ** 2, rr["sigma2_hat"])
            check("5.2/5.3", f"{tag} cv_s seed{s}",
                  float(np.std(s2) / np.mean(s2)), rr["cv_s"])
            check("5.2/5.3", f"{tag} cv_J seed{s}",
                  float(np.std(Jd) / abs(np.mean(Jd))), rr["cv_J"])
            check("5.2/5.3", f"{tag} s_y seed{s}", s_y, rr["s_y"])


# ============================================================ 5.8  no-intercept
def s_5_8():
    r = rec("clarif_item5")
    for tag in ["tabicl_v2", "tabpfn_v2_dither"]:
        for i, s in enumerate(SEEDS):
            J = AMB[f"{tag}__J__{s}"]
            s2 = AMB[f"{tag}__s2__{s}"]
            x = 1.0 + np.diag(J)
            b = float(np.dot(x, s2) / np.dot(x, x))          # no-intercept LS
            res = s2 - b * x
            ss_res = float(np.sum(res ** 2))
            r2_unc = 1.0 - ss_res / float(np.sum(s2 ** 2))
            r2_cen = 1.0 - ss_res / float(np.sum((s2 - np.mean(s2)) ** 2))
            row = r[tag][i]
            check("5.8", f"{tag} b seed{s}", b, row[4])
            check("5.8", f"{tag} R2_uncentred seed{s}", r2_unc, row[5])
            check("5.8", f"{tag} R2_centred seed{s}", r2_cen, row[6])


# ============================================================ 3.4/4.2 brackets
def s_3_4_4_2():
    r = rec("exp4_floor_brackets")
    quanta = rec("exp4_results")
    # cross-check the deltas against the measured quanta, which DO have a script
    for label, qkey, cfg_t in [("TabICL v2", "quantum_tabicl", 1e-3),
                               ("TabSwift", "quantum_tabswift", 1e-1)]:
        q = quanta[qkey]
        for bracket, qfield in [("lower (p10 jump)", "p10_nonzero"),
                                ("upper (median jump)", "median_nonzero_jump")]:
            delta = r[label][bracket]["delta"]
            check("3.4", f"{label} {bracket} delta vs measured quantum",
                  q[qfield], delta)
            rows_a, rows_n = [], []
            for s in SEEDS:
                X, y, _ = ctx(s)
                Q = get_Q(y, seed=Q_SEED)
                p = quantize(wrap(ExactGP(X, sigma=GP_SIGMA).predict), delta)
                J, _, _ = reduced_jacobian(p, y, Q, cfg_t)
                rows_a.append(asym(J))
                rows_n.append(negeig(J))
            for i, s in enumerate(SEEDS):
                check("3.4/4.2", f"{label} {bracket} asym seed{s}",
                      rows_a[i], r[label][bracket]["asym"][i])
                check("3.4/4.2", f"{label} {bracket} negeig seed{s}",
                      rows_n[i], r[label][bracket]["negeig"][i])


# ============================================================ 2.5 stress tests
def s_2_5():
    """Seven instrument stress tests.

    The original constructions were not written down. Each is re-implemented
    from its description in FINAL_NUMBERS 2.5. Tests 1, 2 and 6 depend on a
    random draw whose seed is unrecoverable, so they are compared on order of
    magnitude -- which is all they ever asserted (every one is a
    'machine-precision' claim, not a value claim).
    """
    r = rec("exp_stress_tests")
    X, y, _ = ctx(42)
    Q = get_Q(y, seed=Q_SEED)
    egp = ExactGP(X, sigma=GP_SIGMA)

    # 1 -- estimator recovers a known asymmetric J
    rng = np.random.RandomState(0)
    A = rng.randn(N, N)
    J = central_jacobian(lambda yy: A @ yy, y, h=1e-3)
    check("2.5", "1 estimator recovers known asymmetric J",
          float(np.linalg.norm(J - A, "fro") / np.linalg.norm(A, "fro")),
          r["1_estimator_recovers"], undetermined=True)

    # 2 -- transpose / orientation: m(y) = y_1 * ones must give J[:,1] = 1, else 0
    Jt = central_jacobian(lambda yy: np.full(N, yy[1]), y, h=1e-3)
    ref = np.zeros((N, N)); ref[:, 1] = 1.0
    check("2.5", "2 transpose orientation", float(np.max(np.abs(Jt - ref))),
          r["2_transpose"], undetermined=True)

    # 3 -- basis invariance of asym / negeig across Q seeds {0,1,2}
    W = egp.jacobian(y)
    av = [asym(get_Q(y, seed=q).T @ W @ get_Q(y, seed=q)) for q in (0, 1, 2)]
    nv = [negeig(get_Q(y, seed=q).T @ W @ get_Q(y, seed=q)) for q in (0, 1, 2)]
    check("2.5", "3 basis invariance asym spread",
          float(np.ptp(av)), r["3_basis_inv_asym"], undetermined=True)
    check("2.5", "3 basis invariance negeig spread",
          float(np.ptp(nv)), r["3_basis_inv_negeig"], undetermined=True)

    # 4 -- null perturbation: a constant map has J = 0
    check("2.5", "4 null perturbation ||J||_F",
          float(np.linalg.norm(central_jacobian(lambda yy: np.ones(N), y, h=1e-3), "fro")),
          r["4_null"])

    # 5 -- determinism of the analytic pipeline
    J1 = central_jacobian(wrap(egp.predict), y, h=1e-3)
    J2 = central_jacobian(wrap(egp.predict), y, h=1e-3)
    check("2.5", "5 determinism analytic", float(np.max(np.abs(J1 - J2))), r["5_determinism_analytic"])

    # 6 -- reduced probe vs ambient-then-project
    Jr, _, _ = reduced_jacobian(wrap(egp.predict), y, Q, 1e-3)
    Ja = Q.T @ central_jacobian(wrap(egp.predict), y, h=1e-3) @ Q
    check("2.5", "6 reduced vs ambient-then-project",
          float(np.linalg.norm(Jr - Ja, "fro") / np.linalg.norm(Ja, "fro")),
          r["6_reduced_vs_ambient"], undetermined=True)

    # 7 -- permutation consistency on the analytic map
    perm = np.random.RandomState(1).permutation(N)
    egp_p = ExactGP(X[perm], sigma=GP_SIGMA)
    Jp = egp_p.jacobian(y[perm])
    check("2.5", "7 permutation consistency",
          float(np.linalg.norm(Jp - W[np.ix_(perm, perm)], "fro") / np.linalg.norm(W, "fro")),
          r["7_permutation"])


# ============================================================ 3.6 -- cannot
def s_3_6():
    RESULTS.append({
        "section": "3.6", "quantity": "E1.4 paired, TabICL RoPE zeroed",
        "recomputed": None, "recorded": None, "rel_diff": float("nan"),
        "verdict": "NOT RECOMPUTABLE",
        "note": ("Requires fresh TabICL v2 forward passes with "
                 "model_.row_interactor.tf_row.rope = None. No stored array is a "
                 "Jacobian of the MODIFIED model, so nothing on disk can stand in. "
                 "e14_paired.json is left as the record. Re-running needs ~2000 "
                 "GPU fits, roughly an hour.")})


def main():
    for fn in (s_2_5, s_2_8_and_4_3_diag, s_2_9, s_3_4_4_2, s_3_6,
               s_4_3_vec, s_5_2_5_3, s_5_8, s_7_3):
        print(f"running {fn.__name__} ...", flush=True)
        fn()

    tally = {}
    for row in RESULTS:
        tally[row["verdict"]] = tally.get(row["verdict"], 0) + 1

    print("\n" + "=" * 78)
    print("VERDICT TALLY")
    print("=" * 78)
    for k in sorted(tally):
        print(f"  {k:<45} {tally[k]:4d}")

    bad = [r for r in RESULTS if r["verdict"] in
           ("DIFFERS", "CLOSE", "CONSTRUCTION-UNDETERMINED (INCONSISTENT)",
            "BOTH AT MACHINE PRECISION")]
    if bad:
        print("\n" + "=" * 78)
        print("NOT BIT-IDENTICAL -- every one listed")
        print("=" * 78)
        for r in bad:
            print(f"  [{r['verdict']:<40}] {r['section']:<9} {r['quantity']}")
            print(f"      recomputed {r['recomputed']!r}")
            print(f"      recorded   {r['recorded']!r}   rel {r['rel_diff']:.3e}")

    print("\n" + "=" * 78)
    print("PER SECTION")
    print("=" * 78)
    secs = {}
    for r in RESULTS:
        secs.setdefault(r["section"], {}).setdefault(r["verdict"], 0)
        secs[r["section"]][r["verdict"]] += 1
    for s in sorted(secs):
        print(f"  {s:<10} " + "  ".join(f"{k}={v}" for k, v in sorted(secs[s].items())))

    p = HERE / "recompute_orphans.json"
    json.dump({"tally": tally, "rows": RESULTS}, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
