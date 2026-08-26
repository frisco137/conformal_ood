"""EXPERIMENT 2.2 -- does the A2 violation localise to identifiable context points?

Every item except 2.2.5 and 2.2.7 is a function of Jacobians already on disk.
Those two come from exp22_loo.py and are merged here when present.

THE SCORE.  v = unit eigenvector of lam_min(sym(Q^T J Q)); u = Q v with ||u||=1;
w_i = u_i^2 normalised to sum to 1.

FIVE SCORES COMPARED ON EVERY TARGET (2.2.8):
    w            the eigendirection loading
    |J_ii|       diagonal sensitivity
    colnorm      ||J[:,i]||, how much label i moves all predictions
    resid        |m_i - y_i| on the context
    leverage     hat-matrix diagonal of an OLS fit on X_ctx with intercept

FOUR TARGETS:
    2.2.4  own residual |m_i - y_i|              (context points)
    2.2.5  LOO influence ||m_test^(-i) - m_test|| (context points, from exp22_loo)
    2.2.6  local held-out error                   (TEST points; a score is mapped
           to a test point as the mean over its k=5 nearest context points)
    2.2.7  matched-pair perturbation              (paired comparison, from exp22_loo)

`resid` appears as both a target (2.2.4) and a baseline score (2.2.8); its
self-correlation is 1 by construction and is printed as such, not hidden.

Writes results/exp22_results.json.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC.parent.parent.parent))                  # repo root
sys.path.insert(0, str(SRC.parent.parent / "tier0_instrument"))
warnings.filterwarnings("ignore")

from experiments.core.metrics import get_Q                         # noqa: E402
from experiments.core.context import generate_audit_context        # noqa: E402
from experiments.core.surrogates import ExactGP                    # noqa: E402
from experiments.phase_2.src.paths import RESULTS, ARRAYS, TIER0   # noqa: E402
from experiments.phase_2.src.estimators import jsonable, spearman  # noqa: E402
from chunk1_control_validation import TargetedImitator             # noqa: E402

SEEDS = [42, 100, 200, 300, 400]
MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]
Q_SEED = 0
KNN = 5                       # 2.2.6, fixed in advance
TOPK_STABILITY = 10           # 2.2.3
SCORES = ["w", "absJii", "colnorm", "resid", "leverage"]
KFULL = None


# ------------------------------------------------------------------ primitives
def loading(J, y):
    Q = get_Q(y, seed=Q_SEED)
    Jr = Q.T @ J @ Q
    S = (Jr + Jr.T) / 2.0
    ev, V = np.linalg.eigh(S)
    u = Q @ V[:, 0]
    u = u / np.linalg.norm(u)
    w = u ** 2
    return w / w.sum(), float(ev[0]), u


def participation(w):
    """(sum w)^2 / sum w^2 on the squared loadings; w already sums to 1."""
    return float(1.0 / np.sum(w ** 2))


def leverage(X):
    A = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    H = A @ np.linalg.pinv(A.T @ A) @ A.T
    return np.clip(np.diag(H), 0.0, 1.0)


def all_scores(J, y, m_ctx, X):
    w, lam, _ = loading(J, y)
    return {"w": w,
            "absJii": np.abs(np.diag(J)),
            "colnorm": np.linalg.norm(J, axis=0),
            "rownorm": np.linalg.norm(J, axis=1),
            "resid": np.abs(np.asarray(m_ctx, float) - np.asarray(y, float)),
            "leverage": leverage(X)}, w, lam


def gp_loo_influence(X, y, X_test, Ks, s2, m_test):
    """||m_test^(-i) - m_test|| for the GP control, kernel hyperparameters held
    fixed at the full-context fit so the comparison isolates dropping the row."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        keep = np.ones(n, bool); keep[i] = False
        Ki = KFULL[np.ix_(keep, keep)]
        a = np.linalg.solve(Ki + s2 * np.eye(n - 1), y[keep])
        out[i] = float(np.linalg.norm(Ks[:, keep] @ a - m_test))
    return out


def gp_influence(X, y, X_test=None):
    """W = K(K + sigma^2 I)^-1 from a marginal-likelihood GP fit. Symmetric PSD.

    Scale-free: W is the same whether y is standardised or not, so the
    normalize_y convention inside the fit does not enter.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    k = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e3))
         + WhiteKernel(1.0, (1e-6, 1e3)))
    gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2,
                                  random_state=0).fit(np.asarray(X, float),
                                                      np.asarray(y, float))
    K = gp.kernel_.k1(np.asarray(X, float))
    s2 = float(gp.kernel_.k2.noise_level)
    W = K @ np.linalg.inv(K + s2 * np.eye(len(X)))
    global KFULL
    KFULL = K
    Ks = (gp.kernel_.k1(np.asarray(X_test, float), np.asarray(X, float))
          if X_test is not None else None)
    return W, str(gp.kernel_), s2, Ks


def knn_map(score, X_ctx, X_test, k=KNN):
    """Map a per-context-point score to test points: mean over the k nearest."""
    d = np.sqrt(((np.asarray(X_test)[:, None] - np.asarray(X_ctx)[None, :]) ** 2).sum(-1))
    idx = np.argsort(d, axis=1)[:, :k]
    return np.asarray(score)[idx].mean(axis=1)


# ------------------------------------------------------------------ 2.2.1 repro
def item_2_2_1_reproduction():
    """Recompute the audit's eigenvector localisation from its stored reduced
    Jacobians and compare against FINAL_NUMBERS.md 4.3 / exp_negeigvec.json."""
    npz = TIER0 / "exp3_reduced_jacobians.npz"
    ref_p = TIER0 / "exp_negeigvec.json"
    if not npz.exists() or not ref_p.exists():
        return {"status": "NOT MEASURED", "reason": f"missing {npz.name} or {ref_p.name}"}
    Z = np.load(npz)
    ref = json.load(open(ref_p))
    out = {}
    for tag in ref:
        rows = []
        for r in ref[tag]:
            s = r["seed"]
            key = f"{tag}__J__{s}"
            if key not in Z.files:
                rows.append({"seed": s, "status": "NOT MEASURED",
                             "reason": f"{key} absent from {npz.name}"})
                continue
            Jr = Z[key].astype(np.float64)            # already Q^T J Q, 98x98
            _, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
            Q = get_Q(y, seed=Q_SEED)
            S = (Jr + Jr.T) / 2.0
            ev, V = np.linalg.eigh(S)
            u = Q @ V[:, 0]; u = u / np.linalg.norm(u)
            w = u ** 2; w = w / w.sum()
            top5 = np.argsort(w)[::-1][:5]
            rows.append({
                "seed": s,
                "lam_min_recomputed": float(ev[0]), "lam_min_reported": r["lam_min"],
                "participation_recomputed": participation(w),
                "participation_reported": r["participation_ratio"],
                "top5_recomputed": top5.tolist(), "top5_reported": r["top5_points"],
                "top5_massfrac_recomputed": float(w[top5].sum()),
                "top5_massfrac_reported": r["top5_massfrac"],
                "top5_set_matches": sorted(top5.tolist()) == sorted(r["top5_points"]),
            })
        out[tag] = rows
    return out


# ------------------------------------------------------------------ controls
def item_2_2_11_imitator():
    """Imitator: the negative direction is constructed as -(c/2) v v^T.

    The analytic support in the reduced space is the projection of v onto
    span{1, y-ybar}^perp, so the recoverable target is P v with P = Q Q^T.
    Reported: |cos| between the measured u and P v, and Spearman(w, (Pv)^2).
    Uses the CORRECTED inner Jacobian -- TargetedImitator.inner_jacobian ships
    -c*vv^T where the map applies -(c/2)*vv^T (FINAL_NUMBERS.md 2.3).
    """
    rows = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        ti = TargetedImitator(X, y, seed=s)
        u_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
        G = (ti.egp.jacobian(u_std) + ti.M_anti
             - (ti.c / 2.0) * np.outer(ti.v, ti.v))
        w, lam, u = loading(G, y)
        Q = get_Q(y, seed=Q_SEED)
        Pv = Q @ (Q.T @ ti.v)
        Pv_n = Pv / np.linalg.norm(Pv)
        w_analytic = Pv_n ** 2 / np.sum(Pv_n ** 2)
        rows.append({"seed": s, "lam_min": lam,
                     "abs_cos_u_vs_Pv": float(abs(np.dot(u, Pv_n))),
                     "spearman_w_vs_analytic": spearman(w, w_analytic),
                     "participation": participation(w),
                     "participation_analytic": participation(w_analytic)})
    return rows


def item_2_2_11b_strong_imitator(mults=(1.0, 3.0, 10.0)):
    """Power check with the constructed direction made dominant.

    The shipped TargetedImitator sets c = 0.8*||J_gp||_2 + max(0, lam_min), so
    the rank-one term -(c/2) v v^T has magnitude ~0.4 while Q^T W Q has its own
    spectrum spread over [0, ~1]. The minimum eigenvector of the sum is then a
    MIXTURE of P v and W's least-sensitive directions, and only partial recovery
    is expected -- that is a property of the construction's magnitude, not of
    the score. Here the rank-one term is scaled to c = mult * lam_max(Q^T W Q)
    so the constructed direction dominates, and recovery is measured as a
    function of mult. Reported alongside is the competition ratio
    (c/2) / lam_max(Q^T W Q), which is ~0.4 for the shipped imitator.
    """
    out = {}
    for mult in mults:
        rows = []
        for s in SEEDS:
            X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
            ti = TargetedImitator(X, y, seed=s)
            u_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
            W = ti.egp.jacobian(u_std)
            Q = get_Q(y, seed=Q_SEED)
            lam_max = float(np.linalg.eigvalsh((Q.T @ W @ Q + (Q.T @ W @ Q).T) / 2)[-1])
            c = mult * lam_max
            G = W + ti.M_anti - (c / 2.0) * np.outer(ti.v, ti.v)
            w, lam, u = loading(G, y)
            Pv = Q @ (Q.T @ ti.v); Pv_n = Pv / np.linalg.norm(Pv)
            w_an = Pv_n ** 2 / np.sum(Pv_n ** 2)
            rows.append({"seed": s, "mult": mult, "c": c, "lam_max_QtWQ": lam_max,
                         "competition_ratio": float((c / 2.0) / lam_max),
                         "lam_min": lam,
                         "abs_cos_u_vs_Pv": float(abs(np.dot(u, Pv_n))),
                         "spearman_w_vs_analytic": spearman(w, w_an)})
        out[f"mult_{mult:g}"] = rows
    return out


def shipped_imitator_competition():
    """(c/2) / lam_max(Q^T W Q) for the imitator as shipped, per seed."""
    rows = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        ti = TargetedImitator(X, y, seed=s)
        u_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
        Q = get_Q(y, seed=Q_SEED)
        Wr = Q.T @ ti.egp.jacobian(u_std) @ Q
        lam_max = float(np.linalg.eigvalsh((Wr + Wr.T) / 2)[-1])
        rows.append({"seed": s, "c": float(ti.c), "lam_max_QtWQ": lam_max,
                     "competition_ratio": float((ti.c / 2.0) / lam_max)})
    return rows


def item_2_2_10_gp_audit_context():
    """Exact GP on the AUDIT context, where 10 rows of X are exact duplicates and
    lam_min sits at machine zero, so the direction is arbitrary."""
    rows = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        W = ExactGP(X, sigma=0.5).jacobian(y)
        w, lam, _ = loading(W, y)
        m = W @ y
        sc = {"w": w, "absJii": np.abs(np.diag(W)),
              "colnorm": np.linalg.norm(W, axis=0),
              "resid": np.abs(m - y), "leverage": leverage(X)}
        rows.append({"seed": s, "lam_min": lam, "participation": participation(w),
                     "spearman_w_vs_resid": spearman(w, sc["resid"]),
                     "spearman_w_vs_absJii": spearman(w, sc["absJii"]),
                     "spearman_w_vs_colnorm": spearman(w, sc["colnorm"]),
                     "spearman_w_vs_leverage": spearman(w, sc["leverage"])})
    return rows


# ------------------------------------------------------------------ main sweep
def rebuild_raw_frame(manifest):
    """Common feature frame for 2.2.3.

    chunk2_data.npz stores X standardised with each split's own context-row
    statistics, so two splits of the same dataset sit in slightly different
    frames. For a cross-split geometric comparison the points are put back into
    one frame: the split's own standardiser is undone using statistics
    recovered from the stored arrays, and everything is re-standardised by the
    pooled context rows of the five splits of that dataset.
    """
    data = np.load(ARRAYS / "chunk2_data.npz")
    frames = {}
    for rec in manifest["accepted"]:
        did = rec["did"]
        pooled = np.vstack([data[f"{did}__{s}__X_ctx"] for s in SEEDS])
        mu, sd = pooled.mean(0), pooled.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        frames[did] = {s: (data[f"{did}__{s}__X_ctx"] - mu) / sd for s in SEEDS}
    return frames


def nn_distance(A, B):
    """Mean over rows of A of the distance to the nearest row of B."""
    d = np.sqrt(((np.asarray(A)[:, None] - np.asarray(B)[None, :]) ** 2).sum(-1))
    return float(d.min(axis=1).mean())


def item_2_2_3(frames, load_by):
    """Are the high-loading points in the same region of feature space across
    splits? Cross-split nearest-neighbour distance between top-k loading sets,
    against a random-subset baseline of the same size (200 draws, seeded)."""
    out = {}
    for mid, per_did in load_by.items():
        out[mid] = {}
        for did, per_seed in per_did.items():
            rng = np.random.RandomState(0)
            obs, base = [], []
            for a in SEEDS:
                for b in SEEDS:
                    if a >= b:
                        continue
                    ta = np.argsort(per_seed[a])[::-1][:TOPK_STABILITY]
                    tb = np.argsort(per_seed[b])[::-1][:TOPK_STABILITY]
                    Xa, Xb = frames[did][a], frames[did][b]
                    obs.append(nn_distance(Xa[ta], Xb[tb]))
                    base.append(float(np.mean([
                        nn_distance(Xa[rng.choice(len(Xa), TOPK_STABILITY, replace=False)],
                                    Xb[rng.choice(len(Xb), TOPK_STABILITY, replace=False)])
                        for _ in range(20)])))
            obs, base = np.array(obs), np.array(base)
            out[mid][did] = {"pairs": len(obs),
                             "observed_nn_dist": obs.tolist(),
                             "random_nn_dist": base.tolist(),
                             "mean_observed": float(obs.mean()),
                             "mean_random": float(base.mean()),
                             "ratio_obs_over_random": float(obs.mean() / base.mean())}
    return out


def main():
    manifest = json.load(open(RESULTS / "chunk2_datasets.json"))
    data = np.load(ARRAYS / "chunk2_data.npz")
    res = {"_config": {"seeds": SEEDS, "Q_seed": Q_SEED, "knn": KNN,
                       "topk_stability": TOPK_STABILITY, "scores": SCORES,
                       "note": ("the brief's `phase2.md` / `PHASE2_RESULTS.md` are this "
                                "directory's UNCERTAINTY_EXPERIMENT.md / RESULTS.md; the "
                                "participation ratios of 3-33/98 it cites are "
                                "FINAL_NUMBERS.md 4.3, not a phase-2 number")}}

    print("=" * 78); print("2.2.1 reproduction of the audit's eigenvector localisation")
    print("=" * 78)
    res["2.2.1_reproduction"] = item_2_2_1_reproduction()
    for tag, rows in res["2.2.1_reproduction"].items():
        for r in rows:
            if "status" in r:
                print(f"  {tag:24s} seed {r['seed']}: {r['status']}"); continue
            print(f"  {tag:24s} seed {r['seed']:<5} lam_min {r['lam_min_recomputed']:+.5f} "
                  f"(reported {r['lam_min_reported']:+.5f})  participation "
                  f"{r['participation_recomputed']:.2f} (reported "
                  f"{r['participation_reported']:.2f})  top5 set matches "
                  f"{r['top5_set_matches']}")

    print("\n" + "=" * 78); print("2.2.11 imitator control")
    print("=" * 78)
    res["2.2.11_imitator"] = item_2_2_11_imitator()
    for r in res["2.2.11_imitator"]:
        print(f"  seed {r['seed']:<5} lam_min {r['lam_min']:+.5f}  |cos(u, Pv)| "
              f"{r['abs_cos_u_vs_Pv']:.6f}  spearman(w, analytic) "
              f"{r['spearman_w_vs_analytic']:.6f}  participation {r['participation']:.2f} "
              f"(analytic {r['participation_analytic']:.2f})")

    print("\n" + "=" * 78); print("2.2.11b power check: constructed direction made dominant")
    print("=" * 78)
    res["2.2.11b_strong_imitator"] = item_2_2_11b_strong_imitator()
    res["2.2.11c_shipped_competition"] = shipped_imitator_competition()
    cr = np.mean([r["competition_ratio"] for r in res["2.2.11c_shipped_competition"]])
    print(f"  shipped imitator competition ratio (c/2)/lam_max(Q^T W Q): mean {cr:.4f}")
    for k, rows in res["2.2.11b_strong_imitator"].items():
        print(f"  {k:<10} competition {np.mean([r['competition_ratio'] for r in rows]):6.3f}  "
              f"|cos(u,Pv)| mean {np.mean([r['abs_cos_u_vs_Pv'] for r in rows]):.6f}  "
              f"spearman(w, analytic) mean "
              f"{np.mean([r['spearman_w_vs_analytic'] for r in rows]):.6f}")

    print("\n" + "=" * 78); print("2.2.10a exact GP on the AUDIT context (lam_min at machine zero)")
    print("=" * 78)
    res["2.2.10a_gp_audit_context"] = item_2_2_10_gp_audit_context()
    for r in res["2.2.10a_gp_audit_context"]:
        print(f"  seed {r['seed']:<5} lam_min {r['lam_min']:+.3e}  spearman w vs "
              f"resid {r['spearman_w_vs_resid']:+.4f}  |Jii| {r['spearman_w_vs_absJii']:+.4f}  "
              f"colnorm {r['spearman_w_vs_colnorm']:+.4f}  leverage {r['spearman_w_vs_leverage']:+.4f}")

    # ---------------------------------------------------------------- per split
    print("\n" + "=" * 78); print("per-split scores and targets")
    print("=" * 78)
    per_split, load_by = [], {}
    loo_json = {}
    for mid in MODELS + ["gp_oracle"]:
        src = RESULTS / f"exp22_loo_{mid}.json"
        if src.exists():
            loo_json[mid] = {(r["did"], r["seed"]): r for r in json.load(open(src))["rows"]}
    loo_npz = {}
    for mid in MODELS:
        p = ARRAYS / f"exp22_loo_{mid}.npz"
        if p.exists():
            loo_npz[mid] = np.load(p)

    ZG = np.load(ARRAYS / "chunk2_arrays_tabpfn_v2.npz")
    for mid in MODELS + ["gp_oracle"]:
        Z = None
        if mid in MODELS:
            p = ARRAYS / f"chunk2_arrays_{mid}.npz"
            if not p.exists():
                print(f"  {mid}: chunk 2 arrays absent -- NOT MEASURED"); continue
            Z = np.load(p)
        load_by[mid] = {}
        for rec in manifest["accepted"]:
            did = rec["did"]
            load_by[mid][did] = {}
            for s in SEEDS:
                X = data[f"{did}__{s}__X_ctx"]; y = data[f"{did}__{s}__y_ctx"].astype(float)
                Xt = data[f"{did}__{s}__X_test"]; yt = data[f"{did}__{s}__y_test"].astype(float)
                if mid in MODELS:
                    tag = f"{mid}__{did}__{s}"
                    J = Z[f"{tag}__J_ctx"].astype(np.float64)
                    m_ctx = Z[f"{tag}__m_ctx"]; m_test = Z[f"{tag}__m_test"]
                    extra = {}
                else:
                    J, kern, s2, Ks = gp_influence(X, y, X_test=Xt)
                    m_ctx = J @ y
                    # the oracle GP's test predictions do not depend on the model,
                    # so they are the ones chunk 2 already stored under gp_mu
                    m_test = ZG[f"tabpfn_v2__{did}__{s}__gp_mu"]
                    extra = {"gp_kernel": kern, "gp_noise": s2}
                sc, w, lam = all_scores(J, y, m_ctx, X)
                load_by[mid][did][s] = w
                row = {"model": mid, "did": did, "name": rec["name"], "seed": s,
                       "lam_min": lam, "participation": participation(w),
                       "top10": np.argsort(w)[::-1][:10].tolist(),
                       "top10_w": np.sort(w)[::-1][:10].tolist(),
                       "mass_top5": float(np.sort(w)[::-1][:5].sum()),
                       "mass_top10": float(np.sort(w)[::-1][:10].sum()),
                       "mass_top20": float(np.sort(w)[::-1][:20].sum()),
                       **extra}
                # ---- targets
                tgt = {"resid": sc["resid"]}
                if mid in loo_npz and f"{mid}__{did}__{s}__loo_influence" in loo_npz[mid].files:
                    tgt["loo"] = loo_npz[mid][f"{mid}__{did}__{s}__loo_influence"]
                elif mid == "gp_oracle":
                    # analytic control: refit the GP without each row, kernel fixed
                    tgt["loo"] = gp_loo_influence(X, y, Xt, Ks, s2, m_test)
                row["spearman_score_vs_target"] = {
                    t: {k: spearman(sc[k], tv) for k in SCORES} for t, tv in tgt.items()}
                # ---- 2.2.6 local held-out error, over TEST points
                if m_test is not None:
                    sq = (yt - np.asarray(m_test, float)) ** 2
                    row["spearman_knn_vs_testerr"] = {
                        k: spearman(knn_map(sc[k], X, Xt), sq) for k in SCORES}
                # ---- 2.2.9 pairwise among scores
                row["pairwise_score_spearman"] = {
                    f"{a}|{b}": spearman(sc[a], sc[b])
                    for i, a in enumerate(SCORES) for b in SCORES[i + 1:]}
                if mid in loo_json and (did, s) in loo_json[mid]:
                    L = loo_json[mid][(did, s)]
                    row["perturb"] = {k: L[k] for k in
                                      ["i_high", "i_low_matched", "w_high", "w_low",
                                       "Jii_high", "Jii_low", "Jii_match_rel_gap",
                                       "sens_high", "sens_low", "sens_ratio_high_over_low",
                                       "perturb_delta"]}
                per_split.append(row)
        print(f"  {mid}: {len([r for r in per_split if r['model']==mid])} splits")

    res["per_split"] = per_split
    print("\n" + "=" * 78); print("2.2.3 stability of the high-loading set in feature space")
    print("=" * 78)
    res["2.2.3_stability"] = item_2_2_3(rebuild_raw_frame(manifest), load_by)
    for mid, d in res["2.2.3_stability"].items():
        rr = np.array([v["ratio_obs_over_random"] for v in d.values()])
        print(f"  {mid:<11} ratio observed/random NN distance over 12 datasets: "
              f"mean {rr.mean():.4f}  min {rr.min():.4f}  max {rr.max():.4f}")

    with open(RESULTS / "exp22_results.json", "w") as f:
        json.dump(jsonable(res), f, indent=2)
    print(f"\nSaved: {RESULTS / 'exp22_results.json'}")


if __name__ == "__main__":
    main()
