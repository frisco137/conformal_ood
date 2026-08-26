"""EXPERIMENT 2.5 analysis -- drift (2.5.2), variance shrinkage (2.5.3), the
localised prediction (2.5.4), probe-vs-target stratification (2.5.5), and K
sensitivity (2.5.7).

Consumes results/exp25_<model>.json and results/exp25_controls.json. No model calls.

`drift` is (E[mu_{n+1}(x_*)] - mu_n(x_*)) / s_n(x_*), estimated from K = 64 draws
from the model's own predictive at x'. `drift_se` is the Monte Carlo standard
error of that estimate in the same units, so |drift|/SE is the quantity to read:
a martingale gives |drift|/SE ~ 1, not |drift| ~ 0.

Writes results/exp25_results.json.
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

from experiments.phase_2.src.paths import RESULTS                    # noqa: E402
from experiments.phase_2.src.estimators import jsonable, spearman    # noqa: E402

MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]


def nm(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def summarise(rows, tn):
    d = np.array([r[f"drift_{tn}_K64"] for r in rows], float)
    se = np.array([r[f"drift_se_{tn}_K64"] for r in rows], float)
    ok = np.isfinite(d) & np.isfinite(se) & (se > 0)
    out = {"n": int(len(rows)), "n_usable": int(ok.sum()),
           "drift_mean": nm(d), "drift_median": float(np.median(d[np.isfinite(d)]))
           if np.isfinite(d).any() else float("nan"),
           "abs_drift_mean": nm(np.abs(d)),
           "mc_se_mean": nm(se),
           "drift_over_se_median": float(np.median(np.abs(d[ok]) / se[ok])) if ok.any() else float("nan"),
           "frac_over_se_gt2": float(np.mean(np.abs(d[ok]) / se[ok] > 2)) if ok.any() else float("nan"),
           "frac_over_se_gt3": float(np.mean(np.abs(d[ok]) / se[ok] > 3)) if ok.any() else float("nan")}
    g = np.array([r.get(f"tv_rel_gap_{tn}") if r.get(f"tv_rel_gap_{tn}") is not None else np.nan
                  for r in rows], float)
    out["tv_rel_gap_mean"] = nm(g)
    out["tv_rel_gap_median"] = float(np.median(g[np.isfinite(g)])) if np.isfinite(g).any() else None
    out["tv_n_measured"] = int(np.isfinite(g).sum())
    # K sensitivity (2.5.7): same draws, first 32
    d32 = np.array([r[f"drift_{tn}_K32"] for r in rows], float)
    out["drift_mean_K32"] = nm(d32)
    fin = np.isfinite(d) & np.isfinite(d32)
    out["spearman_K64_vs_K32"] = spearman(d[fin], d32[fin]) if fin.sum() > 2 else None
    return out


def main():
    res = {"_config": {"models": MODELS,
                       "note": ("drift is normalised by s_n(x_*); |drift|/MC-SE is the "
                                "quantity a martingale constrains, not |drift| itself")}}
    per_model = {}
    for mid in MODELS:
        p = RESULTS / f"exp25_{mid}.json"
        if not p.exists():
            print(f"  {mid}: NOT MEASURED (exp25 output absent)")
            per_model[mid] = None
            continue
        rows = json.load(open(p))["rows"]
        ent = {"n_rows": len(rows),
               "sampling": rows[0]["sampling"],
               "by_kind": {}, "overall": {}}
        for tn in ["self", "other"]:
            ent["overall"][tn] = summarise(rows, tn)
        for kind in ["negative", "matched_positive", "general"]:
            sub = [r for r in rows if r["kind"] == kind]
            ent["by_kind"][kind] = ({tn: summarise(sub, tn) for tn in ["self", "other"]}
                                    if sub else None)
            if sub:
                ent["by_kind"][kind]["n"] = len(sub)
                ent["by_kind"][kind]["J_star_mean"] = nm([r["J_star"] for r in sub])

        # ---- 2.5.4 paired negative vs matched positive, within split
        neg = [r for r in rows if r["kind"] == "negative"]
        pos = [r for r in rows if r["kind"] == "matched_positive"]
        pair = {}
        if neg and pos:
            key = lambda r: (r["did"], r["seed"])
            from collections import defaultdict
            pby = defaultdict(list)
            for r in pos:
                pby[key(r)].append(r)
            dn, dp, jn, jp = [], [], [], []
            for i, r in enumerate(neg):
                cand = pby.get(key(r), [])
                if not cand:
                    continue
                m = cand[min(i, len(cand) - 1)] if len(cand) > 1 else cand[0]
                dn.append(r); dp.append(m); jn.append(r["J_star"]); jp.append(m["J_star"])
            for tn in ["self", "other"]:
                a = np.array([r[f"drift_{tn}_K64"] for r in dn], float)
                b = np.array([r[f"drift_{tn}_K64"] for r in dp], float)
                fin = np.isfinite(a) & np.isfinite(b)
                a, b = a[fin], b[fin]
                e = {"n_pairs": int(len(a)),
                     "neg_drift_mean": nm(a), "pos_drift_mean": nm(b),
                     "neg_abs_drift_mean": nm(np.abs(a)), "pos_abs_drift_mean": nm(np.abs(b)),
                     "neg_drift_negative_frac": float(np.mean(a < 0)) if len(a) else None,
                     "pos_drift_negative_frac": float(np.mean(b < 0)) if len(b) else None,
                     "J_star_neg_mean": nm(jn), "J_star_pos_mean": nm(jp),
                     "J_star_abs_gap_median": (float(np.median(np.abs(np.abs(np.array(jn))
                                                                      - np.array(jp))))
                                               if jn else None)}
                if len(a) >= 6:
                    e["wilcoxon_signed_p"] = float(stats.wilcoxon(a - b).pvalue)
                    e["wilcoxon_abs_p"] = float(stats.wilcoxon(np.abs(a) - np.abs(b)).pvalue)
                    e["mean_paired_diff_signed"] = float(np.mean(a - b))
                    e["mean_paired_diff_abs"] = float(np.mean(np.abs(a) - np.abs(b)))
                else:
                    e["wilcoxon_signed_p"] = None
                    e["wilcoxon_abs_p"] = None
                pair[tn] = e
        ent["2.5.4_paired"] = pair or None

        # ---- correlation of drift with J_star across all probes
        for tn in ["self", "other"]:
            js = np.array([r["J_star"] for r in rows], float)
            dd = np.array([r[f"drift_{tn}_K64"] for r in rows], float)
            fin = np.isfinite(js) & np.isfinite(dd)
            ent[f"spearman_drift_vs_Jstar_{tn}"] = (spearman(js[fin], dd[fin])
                                                    if fin.sum() > 2 else None)
        per_model[mid] = ent
    res["per_model"] = per_model

    cp = RESULTS / "exp25_controls.json"
    res["controls_present"] = cp.exists()

    with open(RESULTS / "exp25_results.json", "w") as f:
        json.dump(jsonable(res), f, indent=2)

    print("=" * 100)
    print("2.5.2 / 2.5.3 / 2.5.7  drift and variance shrinkage, all probes")
    print("=" * 100)
    print(f"  {'model':<11}{'tgt':<7}{'n':>5}{'drift mean':>13}{'|drift| mean':>14}"
          f"{'MC SE mean':>13}{'|d|/SE med':>12}{'frac>2':>9}{'frac>3':>9}{'tv gap mean':>13}{'tv n':>6}")
    for mid, e in per_model.items():
        if e is None:
            print(f"  {mid:<11} NOT MEASURED"); continue
        for tn in ["self", "other"]:
            o = e["overall"][tn]
            print(f"  {mid:<11}{tn:<7}{o['n']:>5}{o['drift_mean']:>13.4f}{o['abs_drift_mean']:>14.4f}"
                  f"{o['mc_se_mean']:>13.4f}{o['drift_over_se_median']:>12.3f}"
                  f"{o['frac_over_se_gt2']:>9.3f}{o['frac_over_se_gt3']:>9.3f}"
                  f"{o['tv_rel_gap_mean']:>13.4f}{o['tv_n_measured']:>6}")
    print("\n" + "=" * 100)
    print("2.5.4  negative vs matched-positive J_star, paired within split")
    print("=" * 100)
    for mid, e in per_model.items():
        if e is None or not e.get("2.5.4_paired"):
            print(f"  {mid:<11} NOT MEASURED -- no negative-J_star probes available")
            continue
        for tn in ["self", "other"]:
            q = e["2.5.4_paired"][tn]
            print(f"  {mid:<11}{tn:<7} n={q['n_pairs']:<4} neg drift {q['neg_drift_mean']:+.4f} "
                  f"(|.| {q['neg_abs_drift_mean']:.4f})  pos drift {q['pos_drift_mean']:+.4f} "
                  f"(|.| {q['pos_abs_drift_mean']:.4f})  frac neg-drift<0 "
                  f"{q['neg_drift_negative_frac'] if q['neg_drift_negative_frac'] is not None else float('nan'):.3f}"
                  f"  signed p {q['wilcoxon_signed_p'] if q['wilcoxon_signed_p'] else float('nan'):.4g}"
                  f"  |.| p {q['wilcoxon_abs_p'] if q['wilcoxon_abs_p'] else float('nan'):.4g}")
    print("\n  Spearman(drift, J_star) over all probes:")
    for mid, e in per_model.items():
        if e is None:
            continue
        print(f"    {mid:<11} self {e['spearman_drift_vs_Jstar_self']}  "
              f"other {e['spearman_drift_vs_Jstar_other']}")
    print(f"\nSaved: {RESULTS/'exp25_results.json'}")


if __name__ == "__main__":
    main()
