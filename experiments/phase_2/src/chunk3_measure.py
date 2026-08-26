"""CHUNK 3 -- coverage, interval score, sharpness, NLL, rank correlation.

Consumes chunk2_arrays_<model>.npz and chunk2_estimator_<model>.json. No model
calls: every quantity here is a function of stored predictions and variances.

Estimators scored: native (absent for TabSwift), jacobian, conformal, gp_oracle.
Estimators 1-3 share the model's mean channel; 4 carries its own.

Conventions that need stating because a choice was made:

  * Gaussian intervals m +/- z_level * s for native / jacobian / gp_oracle;
    conformal uses its own calibration quantile and is CONSTANT WIDTH.
  * NLL is a density statement and conformal does not supply a density. Its
    implied Gaussian sd is taken as qhat(0.90)/z(0.90) and every conformal NLL
    is labelled as that conversion, not as a native quantity.
  * 3.6's rank correlation is undefined for conformal: constant variance has
    zero rank variance. Reported as NOT MEASURED with that reason rather than
    as a zero.
  * 3.3 asks for sharpness conditional on achieved coverage. Raw mean width is
    reported beside achieved coverage, AND a recalibrated width: each estimator's
    scale is multiplied by the empirical quantile of |y-m|/s at the nominal
    level, which forces achieved coverage to equal nominal exactly, so widths
    are compared at matched coverage and a narrow-but-wrong estimator gets no
    credit.

Aggregation is per dataset first, then across datasets, so a dataset with more
test rows does not dominate. Pooled-over-points figures are reported separately
where they differ.

Writes chunk3_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC.parent.parent.parent))                  # repo root

from experiments.phase_2.src.paths import RESULTS, ARRAYS          # noqa: E402
from experiments.phase_2.src.estimators import (                   # noqa: E402
    coverage, interval_score, gaussian_nll, spearman, z_for,
    paired_bootstrap_ci, wilcoxon, jsonable,
)

MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]
NOMINAL = [0.50, 0.80, 0.90, 0.95]
EST = ["native", "jacobian", "conformal", "gp_oracle"]
NLL_LEVEL = 0.90          # the level whose conformal half-width becomes a sd


def load_model(mid):
    jp = RESULTS / f"chunk2_estimator_{mid}.json"
    np_ = ARRAYS / f"chunk2_arrays_{mid}.npz"
    if not jp.exists() or not np_.exists():
        return None, None
    return json.load(open(jp)), np.load(np_)


def estimator_inputs(mid, row, z):
    """(mean, s2, constant_width_halfwidths or None) per estimator for one split."""
    tag = f"{mid}__{row['did']}__{row['seed']}"
    y = z[f"{tag}__y_test"]
    m = z[f"{tag}__m_test"]
    out = {}
    if row["native_available"] and f"{tag}__s2_native" in z.files:
        out["native"] = {"m": m, "s2": z[f"{tag}__s2_native"], "const": None}
    # The Jacobian estimator is undefined on a split where n - tr J <= 0, and
    # therefore where sigma2_hat is nan. Those splits are excluded from every
    # jacobian aggregate and counted; they are not filled in, clipped to a
    # default, or silently propagated as nan into a mean.
    s2j = z[f"{tag}__s2_jac"]
    if row["sigma2_hat_defined"] and np.all(np.isfinite(s2j)):
        out["jacobian"] = {"m": m, "s2": s2j, "const": None}
    out["conformal"] = {"m": m, "s2": None,
                        "const": {f"{lv:.2f}": row["conformal_qhat"][f"{lv:.2f}"]
                                  for lv in NOMINAL}}
    out["gp_oracle"] = {"m": z[f"{tag}__gp_mu"], "s2": z[f"{tag}__s2_gp"], "const": None}
    return y, out


def score_split(y, d, level):
    """Coverage / interval score / width for one estimator at one level."""
    alpha = 1.0 - level
    m = np.asarray(d["m"], float)
    if d["const"] is not None:
        q = d["const"][f"{level:.2f}"]
        q = float(q) if q is not None else float("inf")
        lo, hi = m - q, m + q
        s = np.full(len(m), q / z_for(level))
    else:
        s = np.sqrt(np.maximum(np.asarray(d["s2"], float), 0.0))
        lo, hi = m - z_for(level) * s, m + z_for(level) * s
    cov = coverage(y, lo, hi)
    isc = interval_score(y, lo, hi, alpha)
    # recalibrated width: scale s so coverage is exactly `level`
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.abs(y - m) / np.where(s > 0, s, np.nan)
    ratio = ratio[np.isfinite(ratio)]
    c = float(np.quantile(ratio, level)) if ratio.size else float("nan")
    return {"coverage": cov, "signed_dev": cov - level,
            "interval_score": float(np.mean(isc)),
            "width": float(np.mean(hi - lo)),
            "recal_width": float(2 * c * np.mean(s)) if np.isfinite(c) else float("nan"),
            "recal_scale": c}


def main():
    per_split, missing = [], []
    for mid in MODELS:
        meta, z = load_model(mid)
        if meta is None:
            missing.append(mid)
            print(f"  {mid}: chunk 2 output absent -- NOT MEASURED")
            continue
        for row in meta["rows"]:
            y, ests = estimator_inputs(mid, row, z)
            tag = f"{mid}__{row['did']}__{row['seed']}"
            m = z[f"{tag}__m_test"]
            sq = (y - m) ** 2
            rec = {"model": mid, "did": row["did"], "name": row["name"],
                   "seed": row["seed"], "n_test": int(len(y)),
                   "sigma2_hat": row["sigma2_hat"], "trJ": row["trJ"],
                   "n_neg_Jii": row["n_neg_Jii"],
                   "clip_rate_inverted": row["clip_rate_inverted"],
                   "sigma2_hat_defined": bool(row["sigma2_hat_defined"]),
                   "jacobian_defined": "jacobian" in ests,
                   "n_minus_trJ": row["n_minus_trJ"],
                   "Jstar_ge_1_rate": row.get("Jstar_ge_1_rate"),
                   "test_rmse": row["test_rmse"], "est": {}}
            for name, d in ests.items():
                e = {"levels": {f"{lv:.2f}": score_split(y, d, lv) for lv in NOMINAL}}
                mm = np.asarray(d["m"], float)
                sq_e = (y - mm) ** 2
                if d["const"] is not None:
                    q = d["const"][f"{NLL_LEVEL:.2f}"]
                    s2 = np.full(len(y), (float(q) / z_for(NLL_LEVEL)) ** 2)
                    e["nll_note"] = "converted from qhat(0.90)/z(0.90); not a native density"
                    e["spearman_var_vs_sqres"] = None
                    e["spearman_note"] = "constant width: zero rank variance, undefined"
                else:
                    s2 = np.asarray(d["s2"], float)
                    e["spearman_var_vs_sqres"] = spearman(s2, sq_e)
                e["nll"] = float(np.mean(gaussian_nll(y, mm, s2)))
                e["mean_s2"] = float(np.mean(s2))
                rec["est"][name] = e
            rec["mean_sq_residual"] = float(np.mean(sq))
            per_split.append(rec)
        print(f"  {mid}: {len([r for r in per_split if r['model']==mid])} splits scored")

    # ---------------------------------------------------------------- per dataset
    per_dataset = {}
    for mid in MODELS:
        rows = [r for r in per_split if r["model"] == mid]
        if not rows:
            continue
        dids = sorted({r["did"] for r in rows})
        per_dataset[mid] = {}
        for did in dids:
            rs = [r for r in rows if r["did"] == did]
            entry = {"name": rs[0]["name"], "n_splits": len(rs),
                     "test_rmse": float(np.mean([r["test_rmse"] for r in rs])),
                     "clip_rate_inverted": float(np.mean([r["clip_rate_inverted"] for r in rs])),
                     "n_neg_Jii": float(np.mean([r["n_neg_Jii"] for r in rs])),
                     "n_splits_jacobian_defined": int(sum(r["jacobian_defined"] for r in rs)),
                     "n_minus_trJ_mean": float(np.mean([r["n_minus_trJ"] for r in rs])),
                     "Jstar_ge_1_rate": float(np.mean([r["Jstar_ge_1_rate"] or 0.0 for r in rs])),
                     "est": {}}
            for name in EST:
                sub = [r["est"][name] for r in rs if name in r["est"]]
                if not sub:
                    continue
                e = {"levels": {}}
                for lv in NOMINAL:
                    k = f"{lv:.2f}"
                    e["levels"][k] = {q: float(np.mean([s["levels"][k][q] for s in sub]))
                                      for q in ["coverage", "signed_dev", "interval_score",
                                                "width", "recal_width"]}
                    e["levels"][k]["coverage_per_split"] = [s["levels"][k]["coverage"]
                                                            for s in sub]
                e["nll"] = float(np.mean([s["nll"] for s in sub]))
                sp = [s["spearman_var_vs_sqres"] for s in sub
                      if s["spearman_var_vs_sqres"] is not None
                      and np.isfinite(s["spearman_var_vs_sqres"])]
                e["spearman_var_vs_sqres"] = float(np.mean(sp)) if sp else None
                e["spearman_per_split"] = [s["spearman_var_vs_sqres"] for s in sub]
                entry["est"][name] = e
            per_dataset[mid][did] = entry

    # ---------------------------------------------------------------- aggregate
    aggregate = {}
    for mid, dd in per_dataset.items():
        agg = {}
        for name in EST:
            ds = [v["est"][name] for v in dd.values() if name in v["est"]]
            if not ds:
                continue
            a = {"n_datasets": len(ds), "levels": {}}
            for lv in NOMINAL:
                k = f"{lv:.2f}"
                a["levels"][k] = {q: float(np.mean([d["levels"][k][q] for d in ds]))
                                  for q in ["coverage", "signed_dev", "interval_score",
                                            "width", "recal_width"]}
            a["nll"] = float(np.mean([d["nll"] for d in ds]))
            sp = [d["spearman_var_vs_sqres"] for d in ds
                  if d["spearman_var_vs_sqres"] is not None]
            a["spearman_var_vs_sqres"] = float(np.mean(sp)) if sp else None
            agg[name] = a
        rows_m = [r for r in per_split if r["model"] == mid]
        agg["_coverage"] = {
            "n_splits_total": len(rows_m),
            "n_splits_jacobian_defined": int(sum(r["jacobian_defined"] for r in rows_m)),
            "n_datasets_total": len(dd),
            "n_datasets_with_any_jacobian_split":
                int(sum(1 for v in dd.values() if "jacobian" in v["est"])),
            "mean_clip_rate_inverted": float(np.mean([r["clip_rate_inverted"] for r in rows_m])),
            "mean_Jstar_ge_1_rate": float(np.mean([r["Jstar_ge_1_rate"] or 0.0 for r in rows_m])),
            "mean_n_minus_trJ": float(np.mean([r["n_minus_trJ"] for r in rows_m])),
        }
        aggregate[mid] = agg

    # ---------------------------------------------------------- win/loss + paired
    comparisons = {}
    for mid, dd in per_dataset.items():
        pairs = [("jacobian", "native"), ("jacobian", "conformal"),
                 ("native", "conformal"), ("jacobian", "gp_oracle"),
                 ("native", "gp_oracle")]
        cmp_out = {}
        for a, b in pairs:
            dids = [k for k, v in dd.items() if a in v["est"] and b in v["est"]]
            if not dids:
                continue
            entry = {"n_datasets": len(dids), "per_level": {}}
            for lv in NOMINAL:
                k = f"{lv:.2f}"
                da = np.array([dd[i]["est"][a]["levels"][k]["interval_score"] for i in dids])
                db = np.array([dd[i]["est"][b]["levels"][k]["interval_score"] for i in dids])
                diff = da - db                       # negative => a better
                entry["per_level"][k] = {
                    "a_wins": int(np.sum(diff < 0)), "b_wins": int(np.sum(diff > 0)),
                    "ties": int(np.sum(diff == 0)),
                    "mean_diff": float(np.mean(diff)),
                    "bootstrap_ci": paired_bootstrap_ci(diff, seed=0),
                    "wilcoxon": wilcoxon(diff),
                    "per_dataset_diff": {str(i): float(x) for i, x in zip(dids, diff)}}
            # split-level paired test at the 90% level, 60 pairs rather than 12
            sa = [r["est"][a]["levels"]["0.90"]["interval_score"]
                  for r in per_split if r["model"] == mid and a in r["est"]]
            sb = [r["est"][b]["levels"]["0.90"]["interval_score"]
                  for r in per_split if r["model"] == mid and b in r["est"]]
            if len(sa) == len(sb) and sa:
                d90 = np.array(sa) - np.array(sb)
                entry["split_level_0.90"] = {"n": len(d90),
                                             "bootstrap_ci": paired_bootstrap_ci(d90, seed=1),
                                             "wilcoxon": wilcoxon(d90)}
            nlla = np.array([dd[i]["est"][a]["nll"] for i in dids])
            nllb = np.array([dd[i]["est"][b]["nll"] for i in dids])
            entry["nll"] = {"a_wins": int(np.sum(nlla < nllb)),
                            "b_wins": int(np.sum(nlla > nllb)),
                            "mean_diff": float(np.mean(nlla - nllb)),
                            "bootstrap_ci": paired_bootstrap_ci(nlla - nllb, seed=2)}
            cmp_out[f"{a}_vs_{b}"] = entry
        comparisons[mid] = cmp_out

    # ------------------------------------------------------------------ gate 2
    # Conformal's achieved coverage is the harness check: it is calibrated by
    # construction, so a deviation means the harness is wrong. Its per-split
    # coverage is not deterministic -- with n_cal calibration points and rank
    # k = ceil((n_cal+1)(1-alpha)), coverage given the calibration draw is
    # Beta(k, n_cal+1-k), with SD sqrt(l(1-l)/(n_cal+2)) ~ 0.030 at l=0.90 and
    # n_cal=100. The aggregate over all splits is what the +/-3% applies to; the
    # per-split SD is reported beside it so the two are not confused.
    gate2 = {}
    for mid, agg in aggregate.items():
        rows_m = [r for r in per_split if r["model"] == mid]
        g = {}
        for name, tol in [("conformal", 0.03), ("gp_oracle", 0.05)]:
            if name not in agg:
                continue
            covs = [r["est"][name]["levels"]["0.90"]["coverage"] for r in rows_m
                    if name in r["est"]]
            g[name] = {"aggregate_coverage_at_0.90": agg[name]["levels"]["0.90"]["coverage"],
                       "signed_dev": agg[name]["levels"]["0.90"]["signed_dev"],
                       "tolerance": tol,
                       "met": bool(abs(agg[name]["levels"]["0.90"]["signed_dev"]) <= tol),
                       "per_split_sd": float(np.std(covs, ddof=1)) if len(covs) > 1 else None,
                       "per_split_n": len(covs)}
        finite = {}
        for name in EST:
            if name not in agg:
                continue
            vals = [r["est"][name]["levels"][f"{lv:.2f}"]["width"]
                    for r in rows_m if name in r["est"] for lv in NOMINAL]
            finite[name] = {"n_intervals_scored": len(vals),
                            "all_finite": bool(np.all(np.isfinite(vals)))}
        g["finite_intervals"] = finite
        g["n_splits_jacobian_defined"] = agg["_coverage"]["n_splits_jacobian_defined"]
        g["n_splits_total"] = agg["_coverage"]["n_splits_total"]
        gate2[mid] = g

    out = {"gate2": gate2,
           "_config": {"models": MODELS, "estimators": EST, "nominal": NOMINAL,
                       "nll_level_for_conformal": NLL_LEVEL,
                       "aggregation": "per dataset first, then across datasets",
                       "missing_models": missing},
           "per_split": per_split, "per_dataset": per_dataset,
           "aggregate": aggregate, "comparisons": comparisons}
    p = RESULTS / "chunk3_results.json"
    with open(p, "w") as f:
        json.dump(jsonable(out), f, indent=2)

    # ------------------------------------------------------------------- print
    for mid, agg in aggregate.items():
        print("\n" + "=" * 78)
        print(f"{mid}  -- aggregate over datasets")
        print("=" * 78)
        print(f"  {'estimator':<11}{'cov50':>8}{'cov80':>8}{'cov90':>8}{'cov95':>8}"
              f"{'IS@90':>11}{'width@90':>11}{'recalW@90':>11}{'NLL':>9}{'rho':>8}")
        c = agg["_coverage"]
        print(f"  splits: {c['n_splits_total']} total, "
              f"{c['n_splits_jacobian_defined']} with the Jacobian estimator defined; "
              f"mean clip rate {c['mean_clip_rate_inverted']:.4f}, "
              f"mean rate(J_** >= 1) {c['mean_Jstar_ge_1_rate']:.4f}, "
              f"mean n-trJ {c['mean_n_minus_trJ']:.3f}")
        for name in EST:
            if name not in agg:
                print(f"  {name:<11}  NOT MEASURED (estimator absent or undefined on every split)")
                continue
            a = agg[name]
            rho = a["spearman_var_vs_sqres"]
            print(f"  {name:<11}"
                  + "".join(f"{a['levels'][f'{lv:.2f}']['coverage']:8.3f}" for lv in NOMINAL)
                  + f"{a['levels']['0.90']['interval_score']:11.4g}"
                  + f"{a['levels']['0.90']['width']:11.4g}"
                  + f"{a['levels']['0.90']['recal_width']:11.4g}"
                  + f"{a['nll']:9.3f}"
                  + (f"{rho:8.3f}" if rho is not None else f"{'n/a':>8}"))
        for k, v in comparisons.get(mid, {}).items():
            e = v["per_level"]["0.90"]
            ci = e["bootstrap_ci"]
            print(f"    IS@90 {k:<24} wins {e['a_wins']}-{e['b_wins']} of {v['n_datasets']}"
                  f"   mean diff {e['mean_diff']:+.4g}  CI [{ci['lo']:+.4g}, {ci['hi']:+.4g}]")
    print("\n" + "=" * 78)
    print("GATE 2  (conformal is the harness check: calibrated by construction)")
    print("=" * 78)
    for mid, g in gate2.items():
        for name in ["conformal", "gp_oracle"]:
            if name not in g:
                continue
            v = g[name]
            sd = v["per_split_sd"]
            print(f"  {mid:<11} {name:<10} coverage@90 {v['aggregate_coverage_at_0.90']:.4f}  "
                  f"signed dev {v['signed_dev']:+.4f}  tol +/-{v['tolerance']}  met={v['met']}"
                  + (f"   per-split sd {sd:.4f} over {v['per_split_n']}" if sd else ""))
        fin = g["finite_intervals"]
        bad = [k for k, x in fin.items() if not x["all_finite"]]
        print(f"  {mid:<11} finite intervals: "
              + ("all estimators finite" if not bad else f"NON-FINITE in {bad}")
              + f"   jacobian defined on {g['n_splits_jacobian_defined']}/{g['n_splits_total']} splits")
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
