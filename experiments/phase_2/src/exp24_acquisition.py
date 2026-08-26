"""EXPERIMENT 2.4 (D) -- acquisition-argmax sensitivity.

No BO loop and NO NEW MODEL CALLS: every quantity is a function of the means and
variances already stored by Experiment 2.1 in arrays/chunk2_arrays_<model>.npz.

FRAMING (2.4.1), stated because it is a proxy rather than BO itself. Each of the
12 datasets of 2.1 is read as a maximisation surrogate: `y` is the objective, the
100 context rows are evaluations already made, the 50 held-out rows are the
candidate pool, and the incumbent is `y+ = max(y_ctx)`. Nothing is acquired and
nothing is refitted; the question is only which pool point each variance channel
would select.

FOUR VARIANCE CHANNELS, ONE MEAN. All four are fed the model's own `m_test`, so a
difference in the argmax is caused by the variance and nothing else:
    native      the model's head (TabSwift has none -- reported absent)
    jacobian    sigma_hat^2 / (1 - J_**), the corrected form of RESULTS.md 0
    conformal   (qhat(0.90)/z(0.90))^2, constant across the pool
    gp_var      a fitted GP's predictive variance
The full oracle GP -- its own mean AND its own variance -- is reported separately
as `gp_full`, because it is a different object from a variance channel.

ACQUISITIONS. z = (mu - y+)/s.
    EI  = s [ z Phi(z) + phi(z) ]
    UCB = mu + sqrt(beta) s,  beta = 2 log(|pool| t^2 / (6 delta)), delta = 0.1,
          t = 1 (one-shot). |pool| = 50 gives beta = 8.8459, sqrt(beta) = 2.9742.

A STRUCTURAL NOTE ON CONFORMAL, recorded because it makes that channel
degenerate here rather than merely different. Split conformal has constant width,
so s is the same at every pool point. UCB is then mu + const, and EI is
s[z Phi(z) + phi(z)] with z monotone in mu at fixed s, so both are monotone
transforms of mu. The conformal channel therefore always selects argmax(mu) --
pure exploitation -- under either acquisition. This is reported, not worked
around.

Writes results/exp24_results.json.
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

from experiments.phase_2.src.paths import RESULTS, ARRAYS          # noqa: E402
from experiments.phase_2.src.estimators import jsonable, spearman  # noqa: E402

SEEDS = [42, 100, 200, 300, 400]
MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]
CHANNELS = ["native", "jacobian", "conformal", "gp_var"]
CONF_LEVEL = 0.90
DELTA = 0.1
T_ITER = 1
SCALES = [0.5, 1.0, 2.0]
Z_STRATA = 3
INCUMBENT_Q = [0.50, 0.75, 0.90, 1.00]


def z_for(level):
    return float(stats.norm.ppf(0.5 + level / 2.0))


#: EI is evaluated in LOG SPACE. The naive form underflows: h(z) = z Phi(z) +
#: phi(z) evaluates to exactly 0.0 in float64 for z < -39, and the pool-maximum z
#: is below -38 on 3 of TabPFN's 60 splits (did=560, where it reaches -173), so
#: every pool point returned EI = 0 and argmax picked index 0 arbitrarily. Ranks
#: and argmax are unchanged by the log, and log EI stays finite to z ~ -1e8.
LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
Z_ASYMPTOTIC = -6.0


def log_h(z):
    """log(z Phi(z) + phi(z)), stable for very negative z.

    Direct evaluation above Z_ASYMPTOTIC; below it the asymptotic expansion
    h(z) = phi(z) (1/z^2 - 3/z^4 + 15/z^6 - 105/z^8 + ...), which is the standard
    series for the Mills-ratio remainder. Agreement across the switch is checked
    in `_check_log_h`.
    """
    z = np.asarray(z, float)
    out = np.empty_like(z)
    hi = z > Z_ASYMPTOTIC
    if hi.any():
        zh = z[hi]
        out[hi] = np.log(zh * stats.norm.cdf(zh) + stats.norm.pdf(zh))
    lo = ~hi
    if lo.any():
        zl = z[lo]
        w = 1.0 / zl ** 2
        series = w * (1.0 - 3.0 * w + 15.0 * w ** 2 - 105.0 * w ** 3)
        out[lo] = -0.5 * zl ** 2 - LOG_SQRT_2PI + np.log(series)
    return out


def _check_log_h():
    """Max abs deviation between the two branches on an overlap where both are
    accurate. Reported in the JSON so the switch is not taken on trust."""
    zz = np.linspace(-12.0, -6.0, 601)
    direct = np.log(zz * stats.norm.cdf(zz) + stats.norm.pdf(zz))
    w = 1.0 / zz ** 2
    asym = (-0.5 * zz ** 2 - LOG_SQRT_2PI
            + np.log(w * (1.0 - 3.0 * w + 15.0 * w ** 2 - 105.0 * w ** 3)))
    return float(np.max(np.abs(direct - asym)))


def ei(mu, s, ybest):
    """Returns (log EI, z). log EI is rank-equivalent to EI."""
    mu = np.asarray(mu, float); s = np.asarray(s, float)
    z = np.where(s > 0, (mu - ybest) / np.where(s > 0, s, 1.0), 0.0)
    out = np.where(s > 0,
                   np.log(np.where(s > 0, s, 1.0)) + log_h(z),
                   np.log(np.maximum(mu - ybest, 1e-300)))
    return out, z


def ucb(mu, s, beta_sqrt):
    return np.asarray(mu, float) + beta_sqrt * np.asarray(s, float)


def channels_for(mid, row, Z):
    """(mean, {channel: sigma}) plus the separate gp_full object."""
    tag = f"{mid}__{row['did']}__{row['seed']}"
    mu = Z[f"{tag}__m_test"].astype(float)
    y = Z[f"{tag}__y_test"].astype(float)
    ch = {}
    if row["native_available"] and f"{tag}__s2_native" in Z.files:
        ch["native"] = np.sqrt(np.maximum(Z[f"{tag}__s2_native"].astype(float), 0.0))
    s2j = Z[f"{tag}__s2_jac"].astype(float)
    if row["sigma2_hat_defined"] and np.all(np.isfinite(s2j)):
        ch["jacobian"] = np.sqrt(np.maximum(s2j, 0.0))
    q = row["conformal_qhat"][f"{CONF_LEVEL:.2f}"]
    if q is not None and np.isfinite(q):
        ch["conformal"] = np.full(len(mu), float(q) / z_for(CONF_LEVEL))
    ch["gp_var"] = np.sqrt(np.maximum(Z[f"{tag}__s2_gp"].astype(float), 0.0))
    gp_full = (Z[f"{tag}__gp_mu"].astype(float), ch["gp_var"])
    return mu, y, ch, gp_full


def main():
    manifest = json.load(open(RESULTS / "chunk2_datasets.json"))
    res = {"_config": {"channels": CHANNELS, "conf_level": CONF_LEVEL,
                       "delta": DELTA, "t_iter": T_ITER, "scales": SCALES,
                       "z_strata": Z_STRATA, "seeds": SEEDS, "incumbent_quantiles": INCUMBENT_Q,
                       "framing": ("maximisation surrogate: y is the objective, the 100 "
                                   "context rows are evaluations made, the 50 held-out "
                                   "rows are the candidate pool, y+ = max(y_ctx)"),
                       "same_mean": "all four channels are fed the model's own m_test",
                       "ei_in_log_space": True,
                       "log_h_branch_max_dev": _check_log_h()}}
    rows = []
    for mid in MODELS:
        p = ARRAYS / f"chunk2_arrays_{mid}.npz"
        if not p.exists():
            print(f"  {mid}: arrays absent -- NOT MEASURED"); continue
        Z = np.load(p)
        meta = json.load(open(RESULTS / f"chunk2_estimator_{mid}.json"))
        for row in meta["rows"]:
            tag = f"{mid}__{row['did']}__{row['seed']}"
            mu, y, ch, gp_full = channels_for(mid, row, Z)
            ybest = float(np.max(Z[f"{tag}__y_ctx"].astype(float)))
            npool = len(mu)
            beta = 2.0 * np.log(npool * T_ITER ** 2 / (6.0 * DELTA))
            bs = float(np.sqrt(beta))
            ymax = float(np.max(y)); ymin = float(np.min(y))
            rng = ymax - ymin if ymax > ymin else 1.0

            rec = {"model": mid, "did": row["did"], "name": row["name"],
                   "seed": row["seed"], "n_pool": npool, "y_best_ctx": ybest,
                   "pool_max": ymax, "pool_range": rng, "beta": beta,
                   "channels_present": sorted(ch), "acq": {}}

            surf, picks, zs = {}, {}, {}
            for a in ["EI", "UCB"]:
                surf[a], picks[a] = {}, {}
                for c, s in ch.items():
                    if a == "EI":
                        v, z = ei(mu, s, ybest); zs[c] = z
                    else:
                        v = ucb(mu, s, bs)
                    surf[a][c] = v
                    picks[a][c] = int(np.argmax(v))
                # gp_full uses its own mean
                if a == "EI":
                    v, _ = ei(gp_full[0], gp_full[1], ybest)
                else:
                    v = ucb(gp_full[0], gp_full[1], bs)
                surf[a]["gp_full"] = v; picks[a]["gp_full"] = int(np.argmax(v))
                # greedy-on-the-mean reference
                picks[a]["greedy_mean"] = int(np.argmax(mu))

            for a in ["EI", "UCB"]:
                ent = {"picks": picks[a],
                       "pick_y": {c: float(y[i]) for c, i in picks[a].items()},
                       "regret_gap": {c: float(ymax - y[i]) for c, i in picks[a].items()},
                       "regret_gap_norm": {c: float((ymax - y[i]) / rng)
                                           for c, i in picks[a].items()},
                       "agreement": {}, "surface_spearman": {}}
                names = sorted(surf[a])
                for i, c1 in enumerate(names):
                    for c2 in names[i + 1:]:
                        ent["agreement"][f"{c1}|{c2}"] = bool(picks[a][c1] == picks[a][c2])
                        ent["surface_spearman"][f"{c1}|{c2}"] = spearman(surf[a][c1], surf[a][c2])
                rec["acq"][a] = ent

            # ---- 2.4.6 frontier prediction, d EI / d s = phi(z), maximal at z = 0
            #
            # TWO CORRECTIONS, both forced by measurement.
            #
            # (a) The gap is taken in EI space, not log-EI space. log EI carries a
            #     -z^2/2 term, so a gap measured on it grows as z^2 by construction
            #     and would manufacture the opposite of the prediction. EI itself is
            #     recovered as exp(log EI); where that underflows both channels are
            #     ~0 and the gap is correctly ~0.
            #
            # (b) The incumbent is swept. With the standard y+ = max(y_ctx), only
            #     1.3% of TabPFN pool points have |z| < 1 and 0.37% have z > 0
            #     (median z = -6.56, min -884): the incumbent is the maximum of 100
            #     observed values and the pool is a random held-out sample, so EI
            #     sits in its deep tail and z = 0 is never reached. The prediction is
            #     untestable there. y+ = quantile(y_ctx, q) for lower q brings the
            #     frontier into range (q = 0.50 gives 43% of points with |z| < 1).
            #     All quantiles are reported; q = 1.00 is the standard BO incumbent.
            if "native" in ch and "jacobian" in ch:
                rec["frontier"] = {}
                yctx = Z[f"{tag}__y_ctx"].astype(float)
                for q in INCUMBENT_Q:
                    yb = float(np.quantile(yctx, q))
                    la, za = ei(mu, ch["native"], yb)
                    lb, _ = ei(mu, ch["jacobian"], yb)
                    gap = np.abs(np.exp(la) - np.exp(lb))
                    phiz = stats.norm.pdf(za)
                    e = {"y_best": yb,
                         "abs_z_median": float(np.median(np.abs(za))),
                         "frac_absz_lt1": float(np.mean(np.abs(za) < 1.0)),
                         "frac_z_gt0": float(np.mean(za > 0)),
                         "spearman_gap_vs_phiz": spearman(gap, phiz),
                         "spearman_gap_vs_absz": spearman(gap, np.abs(za)),
                         "gap_all_zero": bool(np.all(gap == 0.0))}
                    order = np.argsort(np.abs(za))
                    e["strata"] = []
                    for si, idx in enumerate(np.array_split(order, Z_STRATA)):
                        st = {"stratum": si, "n": int(len(idx)),
                              "abs_z_mean": float(np.mean(np.abs(za[idx])))}
                        p1 = idx[int(np.argmax(la[idx]))]
                        p2 = idx[int(np.argmax(lb[idx]))]
                        st["agree_EI"] = bool(p1 == p2)
                        u1 = idx[int(np.argmax(ucb(mu, ch["native"], bs)[idx]))]
                        u2 = idx[int(np.argmax(ucb(mu, ch["jacobian"], bs)[idx]))]
                        st["agree_UCB"] = bool(u1 == u2)
                        e["strata"].append(st)
                    rec["frontier"][f"{q:.2f}"] = e

            # ---- 2.4.7 scale sensitivity
            rec["scale"] = {}
            for c, s in ch.items():
                rec["scale"][c] = {}
                for sc in SCALES:
                    e = {}
                    for a in ["EI", "UCB"]:
                        v = (ei(mu, sc * s, ybest)[0] if a == "EI"
                             else ucb(mu, sc * s, bs))
                        pk = int(np.argmax(v))
                        e[f"pick_{a}"] = pk
                        e[f"same_as_unscaled_{a}"] = bool(pk == picks[a][c])
                        if "native" in picks[a]:
                            e[f"same_as_native_{a}"] = bool(pk == picks[a]["native"])
                    rec["scale"][c][f"{sc:g}"] = e
            rows.append(rec)
        print(f"  {mid}: {len([r for r in rows if r['model']==mid])} splits")

    res["per_split"] = rows

    # ------------------------------------------------------------- aggregates
    agg = {}
    for mid in MODELS:
        sub = [r for r in rows if r["model"] == mid]
        if not sub:
            continue
        a_out = {}
        for a in ["EI", "UCB"]:
            pairs = {}
            for r in sub:
                for k, v in r["acq"][a]["agreement"].items():
                    pairs.setdefault(k, []).append(v)
            reg = {}
            for r in sub:
                for k, v in r["acq"][a]["regret_gap_norm"].items():
                    reg.setdefault(k, []).append(v)
            sspear = {}
            for r in sub:
                for k, v in r["acq"][a]["surface_spearman"].items():
                    if v is not None and np.isfinite(v):
                        sspear.setdefault(k, []).append(v)
            a_out[a] = {
                "agreement_rate": {k: float(np.mean(v)) for k, v in pairs.items()},
                "agreement_n": {k: len(v) for k, v in pairs.items()},
                "regret_gap_norm_mean": {k: float(np.mean(v)) for k, v in reg.items()},
                "regret_gap_norm_median": {k: float(np.median(v)) for k, v in reg.items()},
                "surface_spearman_mean": {k: float(np.mean(v)) for k, v in sspear.items()},
            }
        fr = [r["frontier"] for r in sub if "frontier" in r]
        if fr:
            a_out["frontier"] = {"n": len(fr)}
            for q in INCUMBENT_Q:
                k = f"{q:.2f}"
                xs = [x[k] for x in fr if k in x]
                good = lambda f: [x[f] for x in xs if x[f] is not None and np.isfinite(x[f])]
                a_out["frontier"][k] = {
                    "abs_z_median": float(np.mean([x["abs_z_median"] for x in xs])),
                    "frac_absz_lt1": float(np.mean([x["frac_absz_lt1"] for x in xs])),
                    "frac_z_gt0": float(np.mean([x["frac_z_gt0"] for x in xs])),
                    "n_gap_all_zero": int(sum(x["gap_all_zero"] for x in xs)),
                    "spearman_gap_vs_phiz_mean": (float(np.mean(good("spearman_gap_vs_phiz")))
                                                  if good("spearman_gap_vs_phiz") else None),
                    "spearman_gap_vs_absz_mean": (float(np.mean(good("spearman_gap_vs_absz")))
                                                  if good("spearman_gap_vs_absz") else None),
                    "strata_agreement": {
                        f"s{si}": {a: float(np.mean([x["strata"][si][f"agree_{a}"] for x in xs]))
                                   for a in ["EI", "UCB"]} for si in range(Z_STRATA)},
                    "strata_abs_z": {f"s{si}": float(np.mean([x["strata"][si]["abs_z_mean"]
                                                             for x in xs]))
                                     for si in range(Z_STRATA)}}
        sc_out = {}
        for c in CHANNELS:
            ss = [r["scale"][c] for r in sub if c in r["scale"]]
            if not ss:
                continue
            sc_out[c] = {f"{s:g}": {
                f"same_as_unscaled_{a}": float(np.mean([x[f"{s:g}"][f"same_as_unscaled_{a}"]
                                                        for x in ss]))
                for a in ["EI", "UCB"]} for s in SCALES}
            for s in SCALES:
                for a in ["EI", "UCB"]:
                    key = f"same_as_native_{a}"
                    vals = [x[f"{s:g}"][key] for x in ss if key in x[f"{s:g}"]]
                    if vals:
                        sc_out[c][f"{s:g}"][key] = float(np.mean(vals))
        a_out["scale"] = sc_out
        a_out["n_splits"] = len(sub)
        agg[mid] = a_out
    res["aggregate"] = agg

    with open(RESULTS / "exp24_results.json", "w") as f:
        json.dump(jsonable(res), f, indent=2)

    print("\n" + "=" * 96)
    print("2.4.3/2.4.4  decision agreement and normalised regret gap")
    print("=" * 96)
    for mid, a in agg.items():
        print(f"\n{mid}  ({a['n_splits']} splits)")
        for acq in ["EI", "UCB"]:
            e = a[acq]
            print(f"  {acq}  agreement:  " + "  ".join(
                f"{k}={v:.3f}" for k, v in sorted(e["agreement_rate"].items())
                if "gp_full" not in k))
            print(f"  {acq}  regret gap: " + "  ".join(
                f"{k}={v:.4f}" for k, v in sorted(e["regret_gap_norm_mean"].items())))
    print(f"\nSaved: {RESULTS/'exp24_results.json'}")


if __name__ == "__main__":
    main()
