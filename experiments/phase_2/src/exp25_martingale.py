"""EXPERIMENT 2.5 (A) -- self-consistency, with the localised prediction from A2.

MARTINGALE. If y' ~ q(.|D_n, x') is drawn from the model's OWN predictive at x'
and appended, then for any target x_*
    E_{y'}[ mu_{n+1}(x_*) ] = mu_n(x_*)
and, by the law of total variance,
    Var_n(f_*) = E_{y'}[ Var_{n+1}(f_*) ] + Var_{y'}( mu_{n+1}(x_*) ).

WHICH JACOBIAN THE LOCALISED PREDICTION IS ABOUT. The brief writes the localised
claim with the context diagonal J_ii, via mu_i(y + delta e_i) - mu_i(y) ~ J_ii
delta. That is the PERTURB operation -- replace an existing label -- whereas the
martingale is the APPEND operation, n -> n+1. The object whose sign governs the
appended increment at the probe is the appended-query diagonal
    J_** = d mu_{n+1}(x') / d y',
for which T2 on the augmented context gives sigma^2 J_** = Var(f_* | y, y') >= 0.
A negative J_** is therefore exactly "appending a higher label at x' lowers the
prediction at x'". J_** is what is stratified on here; it is already stored for
every held-out point by Experiment 2.1 (`<tag>__J_star`).

STRATA AVAILABLE (counted from the stored J_star over 3000 held-out probes each):
    tabicl_v2    0 negative  (  0.00% ) -- negative stratum EMPTY, 2.5.4 not
                 evaluable for this model and reported as NOT MEASURED
    tabpfn_v2   57 negative  (  1.90% ) across 11 of 60 splits
    tabswift   157 negative  (  5.23% ) across 15 of 60 splits

SAMPLING (2.5.1), stated exactly:
    tabicl_v2  inverse CDF on output_type="quantiles" over 999 levels
    tabpfn_v2  inverse CDF on output_type="quantiles" over 999 levels, which the
               package computes from the FullSupportBarDistribution including its
               half-normal tail bins
    tabswift   HAS NO PREDICTIVE DISTRIBUTION. Sampled as Gaussian with the
               Jacobian-derived variance of 2.1, and every TabSwift number is
               labelled `gaussian_jacobian_surrogate`. Nothing is synthesised
               silently.

Writes results/exp25_<model>.json and arrays/exp25_<model>.npz.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np


class _MockAnalytics:
    def __getattr__(self, name):
        return lambda *a, **k: None


sys.modules["analytics"] = _MockAnalytics()
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC.parent.parent.parent))
warnings.filterwarnings("ignore")

import torch                                                        # noqa: E402
from models import load                                             # noqa: E402
from experiments.core.metrics import extract_variance               # noqa: E402
from experiments.phase_2.src.paths import RESULTS, ARRAYS           # noqa: E402
from experiments.phase_2.src.estimators import jsonable             # noqa: E402

SEEDS = [42, 100, 200, 300, 400]
MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]
K = 64                      # 2.5.7 also reports the first 32 of the same draws
K_SUB = 32
LEVELS = np.linspace(1e-3, 1 - 1e-3, 999)
N_GENERAL = 2               # general probes per split, for 2.5.2 / 2.5.3
SAMPLE_SEED = 20260819


def predict_dist(est, mid, Xq):
    """(mean, variance) at Xq under the model's own head."""
    Xq = np.asarray(Xq, float)
    if mid == "tabicl_v2":
        Q = np.asarray(est.predict(Xq, output_type="quantiles", alphas=list(LEVELS)))
        if Q.ndim == 2 and Q.shape[0] == len(LEVELS):
            Q = Q.T
        return np.trapz(Q, x=LEVELS, axis=-1), np.asarray(extract_variance(Q, LEVELS)).ravel()
    if mid == "tabpfn_v2":
        Q = np.asarray(est.predict(Xq, output_type="quantiles", quantiles=list(LEVELS)))
        if Q.ndim == 2 and Q.shape[0] == len(LEVELS):
            Q = Q.T
        return np.trapz(Q, x=LEVELS, axis=-1), np.asarray(extract_variance(Q, LEVELS)).ravel()
    m = np.asarray(est.predict(Xq)).ravel()
    return m, np.full(len(m), np.nan)          # tabswift: no distribution


def quantiles_at(est, mid, Xq):
    Xq = np.asarray(Xq, float)
    if mid == "tabicl_v2":
        Q = np.asarray(est.predict(Xq, output_type="quantiles", alphas=list(LEVELS)))
    else:
        Q = np.asarray(est.predict(Xq, output_type="quantiles", quantiles=list(LEVELS)))
    if Q.ndim == 2 and Q.shape[0] == len(LEVELS):
        Q = Q.T
    return Q


def draw(est, mid, x_prime, rng, s2_fallback):
    """K draws from q(.|D_n, x'). Returns (samples, how)."""
    if mid == "tabswift":
        m = float(np.asarray(est.predict(np.asarray(x_prime, float).reshape(1, -1))).ravel()[0])
        s = float(np.sqrt(max(s2_fallback, 1e-300)))
        return m + s * rng.standard_normal(K), "gaussian_jacobian_surrogate"
    Q = quantiles_at(est, mid, np.asarray(x_prime, float).reshape(1, -1))[0]
    u = rng.uniform(LEVELS[0], LEVELS[-1], K)
    return np.interp(u, LEVELS, Q), "inverse_cdf_999_levels"


def main():
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or MODELS
    data = np.load(ARRAYS / "chunk2_data.npz")
    manifest = json.load(open(RESULTS / "chunk2_datasets.json"))

    for mid in which:
        Z = np.load(ARRAYS / f"chunk2_arrays_{mid}.npz")
        meta = json.load(open(RESULTS / f"chunk2_estimator_{mid}.json"))
        rowmap = {(r["did"], r["seed"]): r for r in meta["rows"]}
        print("\n" + "=" * 78)
        print(f"MODEL {mid}   K={K} draws per probe")
        print("=" * 78, flush=True)
        model = load(mid, task="regression", device="cuda")
        est = model.estimator
        out, arrays = [], {}

        for rec in manifest["accepted"]:
            did = rec["did"]
            for s in SEEDS:
                t0 = time.time()
                tag = f"{mid}__{did}__{s}"
                X = data[f"{did}__{s}__X_ctx"]; y = data[f"{did}__{s}__y_ctx"].astype(float)
                Xt = data[f"{did}__{s}__X_test"]
                Jst = Z[f"{tag}__J_star"].astype(float)
                s2j = Z[f"{tag}__s2_jac"].astype(float)
                rng = np.random.default_rng(SAMPLE_SEED + 1000 * s + did)

                # ---- probe selection
                probes = []
                neg = np.where(Jst < 0)[0]
                pos = np.where(Jst > 0)[0]
                for i in neg:                                  # 2.5.4 negative stratum
                    probes.append((int(i), "negative"))
                    if len(pos):
                        j = int(pos[np.argmin(np.abs(np.abs(Jst[pos]) - abs(Jst[i])))])
                        probes.append((j, "matched_positive"))
                gen = rng.choice(len(Jst), size=min(N_GENERAL, len(Jst)), replace=False)
                probes += [(int(i), "general") for i in gen]

                est.fit(X, y)
                for pi, kind in probes:
                    xp = Xt[pi]
                    other = int((pi + 1) % len(Xt))
                    targets = np.vstack([Xt[pi], Xt[other]])
                    est.fit(X, y)
                    mu_n, var_n = predict_dist(est, mid, targets)
                    ys, how = draw(est, mid, xp, rng, s2j[pi] if np.isfinite(s2j[pi]) else 1.0)

                    Xa = np.vstack([X, xp])
                    mus = np.empty((K, 2)); vrs = np.empty((K, 2))
                    for k in range(K):
                        est.fit(Xa, np.append(y, ys[k]))
                        m1, v1 = predict_dist(est, mid, targets)
                        mus[k] = m1; vrs[k] = v1
                    est.fit(X, y)

                    e = {"did": did, "name": rec["name"], "seed": s, "probe": pi,
                         "kind": kind, "J_star": float(Jst[pi]),
                         "sampling": how, "K": K,
                         "y_sample_mean": float(ys.mean()), "y_sample_sd": float(ys.std()),
                         "mu_n": mu_n.tolist(), "var_n": var_n.tolist()}
                    for tname, ti in [("self", 0), ("other", 1)]:
                        for kk, KK in [("K64", K), ("K32", K_SUB)]:
                            mm = mus[:KK, ti]
                            sn = float(np.sqrt(var_n[ti])) if np.isfinite(var_n[ti]) else float("nan")
                            if not np.isfinite(sn) or sn <= 0:
                                sn = float(np.std(mm)) if np.std(mm) > 0 else 1.0
                            d = float(mm.mean() - mu_n[ti])
                            e[f"drift_{tname}_{kk}"] = d / sn
                            e[f"drift_se_{tname}_{kk}"] = float(mm.std(ddof=1) / np.sqrt(KK)) / sn
                            e[f"drift_raw_{tname}_{kk}"] = d
                            e[f"s_n_{tname}"] = sn
                        vv = vrs[:, ti]
                        if np.all(np.isfinite(vv)) and np.isfinite(var_n[ti]):
                            lhs = float(var_n[ti])
                            rhs = float(vv.mean() + mus[:, ti].var(ddof=0))
                            e[f"tv_lhs_{tname}"] = lhs
                            e[f"tv_rhs_{tname}"] = rhs
                            e[f"tv_rel_gap_{tname}"] = (lhs - rhs) / lhs if lhs > 0 else float("nan")
                            e[f"E_var_next_{tname}"] = float(vv.mean())
                            e[f"var_of_mean_{tname}"] = float(mus[:, ti].var(ddof=0))
                        else:
                            e[f"tv_rel_gap_{tname}"] = None
                    out.append(e)
                    arrays[f"{tag}__p{pi}__{kind}__mus"] = mus
                    arrays[f"{tag}__p{pi}__{kind}__ys"] = ys

                nneg = sum(1 for _, k in probes if k == "negative")
                print(f"  did={did:<6} seed {s:<5} {time.time()-t0:6.1f}s  probes={len(probes):3d} "
                      f"(neg {nneg})  Jstar range [{Jst.min():+.3f}, {Jst.max():+.3f}]", flush=True)
                np.savez_compressed(ARRAYS / f"exp25_{mid}.npz", **arrays)
                with open(RESULTS / f"exp25_{mid}.json", "w") as f:
                    json.dump(jsonable({"_config": {
                        "K": K, "K_sub": K_SUB, "levels": len(LEVELS),
                        "n_general": N_GENERAL, "sample_seed": SAMPLE_SEED,
                        "stratified_on": "J_star (appended-query diagonal) from Experiment 2.1",
                        "sampling": how}, "rows": out}), f, indent=2)
        del model
        torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
