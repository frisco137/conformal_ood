"""EXPERIMENT 2.3 analysis -- detection AUC and precision@k (2.3.4), baselines
(2.3.5, 2.3.6, 2.3.8), and the uncorrupted baseline (2.3.10).

SCORES. Jacobian-derived, from exp23_corrupt.py / exp23_controls.py:
    colnorm, absJii, rownorm, asym_rc, abs_asym_rc, w
`asym_rc = ||J[i,:]|| - ||J[:,i]||` is signed; `abs_asym_rc` is its magnitude,
which is the form the A1 claim is about (a symmetric J gives zero for both).
Both are reported because AUC is orientation-sensitive and the signed score
would otherwise hide a reversed-direction signal.

BASELINES:
    resid       |m_i - y_i| on the corrupted context                 (2.3.5)
    cook        Cook's distance, OLS on (X, y_corrupted)             (2.3.6)
    gp_infl     diag K(K + sigma^2 I)^-1                             (2.3.8)
    gp_loo_res  |y_i - m_i| / (1 - H_ii), GP standardised LOO residual

`gp_infl` contains no `y` and therefore cannot respond to corruption; its AUC is
chance by construction. It is reported as specified, with that stated. The
GP-based score that does use `y` is `gp_loo_res`, added so the "Bayesian
reference method" of 2.3.8 has a form that can actually detect anything.

ORIENTATION. AUC is computed with "higher score = more likely corrupted". An AUC
below 0.5 means the reverse ordering detects, and is reported as measured rather
than flipped.

Writes results/exp23_results.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC.parent.parent.parent))
warnings.filterwarnings("ignore")

from experiments.phase_2.src.paths import RESULTS, ARRAYS               # noqa: E402
from experiments.phase_2.src.estimators import jsonable, rbf            # noqa: E402
from experiments.phase_2.src.exp23_corrupt import (                     # noqa: E402
    SEEDS, SCHEMES, RATES, DIDS, scores_from_J,
)

MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]
CONTROLS = ["exactgp", "hiergp"]
JAC_SCORES = ["colnorm", "absJii", "rownorm", "asym_rc", "abs_asym_rc", "w"]
BASE_SCORES = ["resid", "cook", "gp_infl", "gp_loo_res"]
ALL_SCORES = JAC_SCORES + BASE_SCORES
GP_SIGMA = 0.5


def auc(score, labels):
    """Mann-Whitney AUC, higher score = positive class. nan if one class empty."""
    s = np.asarray(score, float); y = np.asarray(labels, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = stats.rankdata(s)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def precision_at_k(score, labels, k):
    s = np.asarray(score, float); y = np.asarray(labels, bool)
    top = np.argsort(s)[::-1][:k]
    return float(y[top].sum() / k) if k > 0 else float("nan")


def cooks_distance(X, y):
    A = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    p = A.shape[1]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = np.asarray(y, float) - A @ beta
    H = A @ np.linalg.pinv(A.T @ A) @ A.T
    h = np.clip(np.diag(H), 0.0, 1.0 - 1e-12)
    s2 = float(r @ r / max(len(y) - p, 1))
    return (r ** 2 / (p * max(s2, 1e-300))) * (h / (1.0 - h) ** 2)


def gp_scores(X, y):
    K = rbf(X, X, 1.0)
    H = K @ np.linalg.inv(K + GP_SIGMA ** 2 * np.eye(len(X)))
    m = H @ np.asarray(y, float)
    h = np.clip(np.diag(H), 0.0, 1.0 - 1e-12)
    return np.diag(H).copy(), np.abs(np.asarray(y, float) - m) / (1.0 - h)


def build_scores(J, y, m_ctx, X):
    sc, lam = scores_from_J(J, y)
    sc["abs_asym_rc"] = np.abs(sc["asym_rc"])
    sc["resid"] = np.abs(np.asarray(m_ctx, float) - np.asarray(y, float))
    sc["cook"] = cooks_distance(X, y)
    gi, gl = gp_scores(X, y)
    sc["gp_infl"] = gi
    sc["gp_loo_res"] = gl
    return sc, lam


def collect(tagsrc, jsonp, npzp, data):
    if not jsonp.exists() or not npzp.exists():
        return None
    meta = json.load(open(jsonp)); Z = np.load(npzp)
    out = []
    for r in meta["rows"]:
        tag = f"{tagsrc(r)}"
        if f"{tag}__J" not in Z.files:
            continue
        X = data[f"{r['did']}__{r['seed']}__X_ctx"]
        J = Z[f"{tag}__J"].astype(np.float64)
        yc = Z[f"{tag}__y_corr"]; m_ctx = Z[f"{tag}__m_ctx"]
        idx = Z[f"{tag}__idx"]
        lab = np.zeros(len(yc), bool); lab[idx] = True
        sc, lam = build_scores(J, yc, m_ctx, X)
        row = {k: r[k] for k in ["model", "did", "name", "seed", "scheme", "rate",
                                 "n_ctx", "n_corrupt", "n_changed"]}
        row["lam_min"] = lam
        row["auc"] = {k: auc(sc[k], lab) for k in ALL_SCORES}
        row["prec_at_k"] = {k: precision_at_k(sc[k], lab, int(r["n_corrupt"]))
                            for k in ALL_SCORES}
        row["prec_at_k_chance"] = float(r["n_corrupt"] / r["n_ctx"])
        out.append(row)
    return out


def uncorrupted_baseline(data, manifest):
    """2.3.10 -- scores on the CLEAN context, ranked against the index set that
    WOULD have been corrupted. Any AUC away from 0.5 is spurious concentration."""
    from experiments.phase_2.src.exp23_corrupt import corrupt_indices
    out = []
    for mid in MODELS:
        p = ARRAYS / f"chunk2_arrays_{mid}.npz"
        if not p.exists():
            continue
        Z = np.load(p)
        for did in DIDS:
            for s in SEEDS:
                tag = f"{mid}__{did}__{s}"
                if f"{tag}__J_ctx" not in Z.files:
                    continue
                X = data[f"{did}__{s}__X_ctx"]
                y = data[f"{did}__{s}__y_ctx"].astype(float)
                J = Z[f"{tag}__J_ctx"].astype(np.float64)
                sc, lam = build_scores(J, y, Z[f"{tag}__m_ctx"], X)
                for rate in RATES:
                    idx, k = corrupt_indices(did, s, rate, len(y))
                    lab = np.zeros(len(y), bool); lab[idx] = True
                    out.append({"model": mid, "did": did, "seed": s, "rate": rate,
                                "auc": {q: auc(sc[q], lab) for q in ALL_SCORES},
                                "prec_at_k": {q: precision_at_k(sc[q], lab, k)
                                              for q in ALL_SCORES},
                                "prec_at_k_chance": float(k / len(y))})
    return out


def loo_baseline(data):
    """2.3.7 -- actual leave-one-out on the corrupted context, 10% rate only.

    Two forms: `loo_test_shift` = ||m_test^(-i) - m_test||, the influence form of
    2.2.5; `loo_self` = |y_i - m_i^(-i)|, the point's own out-of-fold residual.
    """
    out = []
    for mid in MODELS:
        jp, npz = RESULTS / f"exp23_loo_{mid}.json", ARRAYS / f"exp23_loo_{mid}.npz"
        if not jp.exists() or not npz.exists():
            continue
        meta = json.load(open(jp)); Z = np.load(npz)
        for r in meta["rows"]:
            tag = f"{mid}__{r['did']}__{r['seed']}__{r['scheme']}__{int(r['rate']*100)}"
            if f"{tag}__loo_self" not in Z.files:
                continue
            lab = np.zeros(r["n_ctx"], bool); lab[Z[f"{tag}__idx"]] = True
            sc = {"loo_test_shift": Z[f"{tag}__loo_test_shift"],
                  "loo_self": Z[f"{tag}__loo_self"]}
            out.append({**{k: r[k] for k in ["model", "did", "seed", "scheme", "rate",
                                             "n_ctx", "n_corrupt"]},
                        "auc": {k: auc(v, lab) for k, v in sc.items()},
                        "prec_at_k": {k: precision_at_k(v, lab, int(r["n_corrupt"]))
                                      for k, v in sc.items()}})
    return out


def nanmean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def main():
    data = np.load(ARRAYS / "chunk2_data.npz")
    manifest = json.load(open(RESULTS / "chunk2_datasets.json"))
    res = {"_config": {"scores": ALL_SCORES, "jac_scores": JAC_SCORES,
                       "base_scores": BASE_SCORES, "dids": DIDS, "seeds": SEEDS,
                       "schemes": SCHEMES, "rates": RATES,
                       "orientation": "higher score = predicted corrupted"}}

    rows = []
    for mid in MODELS:
        r = collect(lambda x: f"{x['model']}__{x['did']}__{x['seed']}__{x['scheme']}__{int(x['rate']*100)}",
                    RESULTS / f"exp23_corrupt_{mid}.json",
                    ARRAYS / f"exp23_corrupt_{mid}.npz", data)
        if r is None:
            print(f"  {mid}: exp23_corrupt output absent -- NOT MEASURED")
            continue
        rows += r
        print(f"  {mid}: {len(r)} corrupted-context combinations scored")
    rc = collect(lambda x: f"{x['model']}__{x['did']}__{x['seed']}__{x['scheme']}__{int(x['rate']*100)}",
                 RESULTS / "exp23_controls.json", ARRAYS / "exp23_controls.npz", data)
    if rc:
        rows += rc
        print(f"  controls: {len(rc)} combinations scored")
    res["per_combination"] = rows
    res["2.3.10_uncorrupted"] = uncorrupted_baseline(data, manifest)
    res["2.3.7_actual_loo"] = loo_baseline(data)

    # ------------------------------------------------------------- aggregates
    agg = {}
    for mid in MODELS + CONTROLS:
        sub = [r for r in rows if r["model"] == mid]
        if not sub:
            continue
        agg[mid] = {"n": len(sub), "overall": {}, "by_scheme": {}, "by_rate": {}}
        for k in ALL_SCORES:
            agg[mid]["overall"][k] = {"auc": nanmean([r["auc"][k] for r in sub]),
                                      "prec": nanmean([r["prec_at_k"][k] for r in sub])}
        for sch in SCHEMES:
            s2 = [r for r in sub if r["scheme"] == sch]
            agg[mid]["by_scheme"][sch] = {k: nanmean([r["auc"][k] for r in s2])
                                          for k in ALL_SCORES}
        for rt in RATES:
            s2 = [r for r in sub if r["rate"] == rt]
            agg[mid]["by_rate"][f"{rt:.2f}"] = {k: nanmean([r["auc"][k] for r in s2])
                                                for k in ALL_SCORES}
        agg[mid]["chance_prec"] = nanmean([r["prec_at_k_chance"] for r in sub])
    res["aggregate"] = agg

    unc = {}
    for mid in MODELS:
        sub = [r for r in res["2.3.10_uncorrupted"] if r["model"] == mid]
        if sub:
            unc[mid] = {k: nanmean([r["auc"][k] for r in sub]) for k in ALL_SCORES}
    res["2.3.10_uncorrupted_agg"] = unc

    with open(RESULTS / "exp23_results.json", "w") as f:
        json.dump(jsonable(res), f, indent=2)

    print("\n" + "=" * 100)
    print("2.3.4  detection AUC, mean over combinations (chance = 0.5)")
    print("=" * 100)
    print(f"  {'model':<11}" + "".join(f"{k:>13}" for k in ALL_SCORES))
    for mid, a in agg.items():
        print(f"  {mid:<11}" + "".join(f"{a['overall'][k]['auc']:13.4f}" for k in ALL_SCORES))
    print("\n  uncorrupted baseline (2.3.10), AUC against the would-be corrupted set")
    for mid, a in unc.items():
        print(f"  {mid:<11}" + "".join(f"{a[k]:13.4f}" for k in ALL_SCORES))
    lb = res["2.3.7_actual_loo"]
    if lb:
        print("\n  2.3.7 actual LOO on the corrupted context (10% rate only)")
        print(f"  {'model':<11}{'n':>5}{'loo_test_shift AUC':>21}{'loo_self AUC':>15}"
              f"{'loo_test_shift P@k':>21}{'loo_self P@k':>15}")
        for mid in MODELS:
            s = [r for r in lb if r["model"] == mid]
            if not s:
                continue
            print(f"  {mid:<11}{len(s):>5}"
                  f"{nanmean([r['auc']['loo_test_shift'] for r in s]):21.4f}"
                  f"{nanmean([r['auc']['loo_self'] for r in s]):15.4f}"
                  f"{nanmean([r['prec_at_k']['loo_test_shift'] for r in s]):21.4f}"
                  f"{nanmean([r['prec_at_k']['loo_self'] for r in s]):15.4f}")
    else:
        print("\n  2.3.7 actual LOO -- NOT MEASURED (exp23_loo outputs absent)")
    print(f"\nSaved: {RESULTS/'exp23_results.json'}")


if __name__ == "__main__":
    main()
