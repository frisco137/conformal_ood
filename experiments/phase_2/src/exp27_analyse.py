"""EXPERIMENTS 2.6 and 2.7 analysis -- BO regret, and the sign of the update at
the acquired point.

2.7 metrics: simple regret against iteration, per seed, paired across seeds within
an objective; win/loss across objectives; the iteration at which each arm first
reaches a fixed fraction of the optimum.

2.6 rides on the same trajectories. At each iteration, before appending, `J_**` at
the acquired point is recorded (Jacobian arm only, where it is computed anyway);
after appending, `move = mu_{n+1}(x_t) - mu_n(x_t)` and `surprise = y_t - mu_n(x_t)`.
For any posterior mean the two must share a sign, and `shrinkage = move/surprise`
must lie in [0, 1].

Writes results/exp27_results.json.
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

ARMS = ["tabpfn_native", "tabpfn_conformal", "tabpfn_jacobian", "gp", "random"]
FRACS = [0.5, 0.9]


def load_rows():
    rows = []
    for p in sorted(RESULTS.glob("exp27_bo_*.json")):
        rows += json.load(open(p))["rows"]
    return rows


def first_hit(reg, target):
    for i, r in enumerate(reg):
        if r <= target:
            return i
    return None


def main():
    rows = load_rows()
    if not rows:
        print("  no exp27 output -- NOT MEASURED"); return
    objs = sorted({r["objective"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    res = {"_config": {"objectives": objs, "seeds": seeds, "arms": ARMS,
                       "n_trajectories": len(rows)}}

    # ------------------------------------------------------------------ 2.7.4/5
    per_obj = {}
    for o in objs:
        sub = [r for r in rows if r["objective"] == o]
        rng = max(abs(r["simple_regret"][0]) for r in sub) or 1.0
        e = {"n_seeds": len({r["seed"] for r in sub}), "arms": {}}
        for arm in ARMS:
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            fr = np.array([r["final_regret"] for r in a], float)
            hits = {}
            for f in FRACS:
                tgt = (1 - f) * np.mean([r["simple_regret"][0] for r in a])
                h = [first_hit(r["simple_regret"], tgt) for r in a]
                hits[f"{f:.1f}"] = {"reached": int(sum(x is not None for x in h)),
                                    "of": len(h),
                                    "median_iter": (float(np.median([x for x in h
                                                                     if x is not None]))
                                                    if any(x is not None for x in h) else None)}
            e["arms"][arm] = {"final_regret_per_seed": fr.tolist(),
                              "final_regret_mean": float(fr.mean()),
                              "final_regret_median": float(np.median(fr)),
                              "first_hit": hits,
                              "regret_curve_mean": np.mean(
                                  [r["simple_regret"] for r in a], axis=0).tolist()}
        per_obj[o] = e
    res["per_objective"] = per_obj

    # paired comparisons, seed-matched within objective
    cmp_out = {}
    pairs = [("tabpfn_jacobian", "tabpfn_native"), ("tabpfn_jacobian", "gp"),
             ("tabpfn_native", "gp"), ("tabpfn_conformal", "gp"),
             ("tabpfn_native", "random"), ("gp", "random"),
             ("tabpfn_native", "tabpfn_conformal")]
    for a, b in pairs:
        diffs, wins, losses = [], 0, 0
        per = {}
        for o in objs:
            da = {r["seed"]: r["final_regret"] for r in rows
                  if r["objective"] == o and r["arm"] == a}
            db = {r["seed"]: r["final_regret"] for r in rows
                  if r["objective"] == o and r["arm"] == b}
            ks = sorted(set(da) & set(db))
            if not ks:
                continue
            d = np.array([da[k] - db[k] for k in ks], float)
            # scale-free per objective: normalise by the mean initial regret
            init = np.mean([r["simple_regret"][0] for r in rows
                            if r["objective"] == o and r["arm"] in (a, b)])
            dn = d / (init if init else 1.0)
            per[o] = {"n_seeds": len(ks), "mean_diff": float(d.mean()),
                      "mean_diff_norm": float(dn.mean()),
                      "a_better_seeds": int((d < 0).sum())}
            diffs += dn.tolist()
            if d.mean() < 0:
                wins += 1
            elif d.mean() > 0:
                losses += 1
        diffs = np.array(diffs, float)
        ent = {"per_objective": per, "objective_wins_a": wins, "objective_wins_b": losses,
               "n_paired": int(len(diffs)),
               "mean_norm_diff": float(diffs.mean()) if len(diffs) else None}
        if len(diffs) >= 6:
            ent["wilcoxon_p"] = float(stats.wilcoxon(diffs).pvalue)
            bs = np.random.RandomState(0).randint(0, len(diffs), (10000, len(diffs)))
            m = diffs[bs].mean(1)
            ent["bootstrap_ci"] = [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]
        cmp_out[f"{a}_vs_{b}"] = ent
    res["comparisons"] = cmp_out

    # ------------------------------------------------------------------ 2.6
    six = {}
    for arm in ["tabpfn_native", "tabpfn_conformal", "tabpfn_jacobian", "gp"]:
        recs = [(r, s) for r in rows if r["arm"] == arm for s in r["six"]]
        if not recs:
            continue
        viol = np.array([s["sign_violation"] for _, s in recs], bool)
        shr = np.array([s["shrinkage"] if s["shrinkage"] is not None else np.nan
                        for _, s in recs], float)
        js = np.array([s["J_star"] if s["J_star"] is not None else np.nan
                       for _, s in recs], float)
        e = {"n": int(len(recs)), "violation_rate": float(viol.mean()),
             "shrinkage_median": float(np.nanmedian(shr)),
             "shrinkage_frac_outside_01": float(np.nanmean((shr < 0) | (shr > 1))),
             "shrinkage_frac_negative": float(np.nanmean(shr < 0))}
        if np.isfinite(js).any():
            neg = np.isfinite(js) & (js < 0)
            pos = np.isfinite(js) & (js >= 0)
            e["J_star_available"] = int(np.isfinite(js).sum())
            e["n_Jstar_negative"] = int(neg.sum())
            e["violation_rate_Jstar_neg"] = float(viol[neg].mean()) if neg.any() else None
            e["violation_rate_Jstar_pos"] = float(viol[pos].mean()) if pos.any() else None
            if neg.any() and pos.any():
                tab = [[int((viol & neg).sum()), int((~viol & neg).sum())],
                       [int((viol & pos).sum()), int((~viol & pos).sum())]]
                e["fisher_p"] = float(stats.fisher_exact(tab)[1])
                e["contingency"] = tab
        # 2.6.4 per-trajectory violation count vs final regret
        pt = {}
        for r in rows:
            if r["arm"] != arm:
                continue
            pt.setdefault(r["objective"], []).append(
                (sum(s["sign_violation"] for s in r["six"]), r["final_regret"]))
        cors = {}
        for o, v in pt.items():
            if len(v) >= 3:
                a_ = np.array([x[0] for x in v], float); b_ = np.array([x[1] for x in v], float)
                cors[o] = spearman(a_, b_)
        e["violation_vs_final_regret_spearman"] = cors
        e["note_correlation"] = ("correlation over the trajectories available; "
                                 "no causal relationship claimed")
        six[arm] = e
    res["2.6"] = six

    # ------------------------------------------------------------------ 2.7.7/8
    audit = {}
    ar = [r for r in rows if r["arm"] == "tabpfn_jacobian"]
    if ar:
        allw = [w for r in ar for w in r["audit"]]
        audit = {"n_iterations": len(allw),
                 "n_sigma2_undefined": int(sum(not w["sigma2_hat_defined"] for w in allw)),
                 "frac_sigma2_undefined": float(np.mean([not w["sigma2_hat_defined"]
                                                         for w in allw])),
                 "trJ_over_n_mean": float(np.mean([w["trJ"] / w["n"] for w in allw])),
                 "trJ_over_n_max": float(np.max([w["trJ"] / w["n"] for w in allw])),
                 "n_minus_trJ_min": float(np.min([w["n_minus_trJ"] for w in allw])),
                 "neg_Jii_total": int(sum(w["n_neg_Jii"] for w in allw)),
                 "asym_mean": float(np.mean([w["asym"] for w in allw])),
                 "negeig_mean": float(np.mean([w["negeig"] for w in allw])),
                 "by_iteration": {}}
        for t in sorted({w["t"] for w in allw}):
            ws = [w for w in allw if w["t"] == t]
            audit["by_iteration"][str(t)] = {
                "n": float(np.mean([w["n"] for w in ws])),
                "trJ": float(np.mean([w["trJ"] for w in ws])),
                "n_minus_trJ": float(np.mean([w["n_minus_trJ"] for w in ws])),
                "frac_undefined": float(np.mean([not w["sigma2_hat_defined"] for w in ws])),
                "neg_Jii": float(np.mean([w["n_neg_Jii"] for w in ws])),
                "asym": float(np.mean([w["asym"] for w in ws])),
                "negeig": float(np.mean([w["negeig"] for w in ws]))}
    res["2.7.7_2.7.8_audit"] = audit

    with open(RESULTS / "exp27_results.json", "w") as f:
        json.dump(jsonable(res), f, indent=2)

    print("=" * 96)
    print("2.7.4/2.7.5  final simple regret, mean over seeds")
    print("=" * 96)
    print(f"  {'objective':<14}" + "".join(f"{a:>19}" for a in ARMS))
    for o in objs:
        e = per_obj[o]["arms"]
        print(f"  {o:<14}" + "".join(
            f"{e[a]['final_regret_mean']:19.5g}" if a in e else f"{'--':>19}" for a in ARMS))
    print("\n  paired, seed-matched, normalised by initial regret (negative favours the first arm)")
    for k, v in cmp_out.items():
        if v.get("mean_norm_diff") is None:
            continue
        ci = v.get("bootstrap_ci")
        print(f"    {k:<40} obj wins {v['objective_wins_a']}-{v['objective_wins_b']}  "
              f"mean {v['mean_norm_diff']:+.4f}"
              + (f"  CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={v['wilcoxon_p']:.4g}" if ci else ""))
    print("\n" + "=" * 96)
    print("2.6  sign of the update at the acquired point")
    print("=" * 96)
    for arm, e in six.items():
        print(f"  {arm:<18} n={e['n']:<5} violation rate {e['violation_rate']:.4f}   "
              f"shrinkage median {e['shrinkage_median']:+.4f}   outside [0,1] "
              f"{e['shrinkage_frac_outside_01']:.4f}   negative {e['shrinkage_frac_negative']:.4f}")
        if "n_Jstar_negative" in e:
            print(f"  {'':<18} J_** negative on {e['n_Jstar_negative']}/{e['J_star_available']}"
                  f"   violation rate | J_**<0 : "
                  f"{e['violation_rate_Jstar_neg'] if e['violation_rate_Jstar_neg'] is not None else float('nan'):.4f}"
                  f"   | J_**>=0 : "
                  f"{e['violation_rate_Jstar_pos'] if e['violation_rate_Jstar_pos'] is not None else float('nan'):.4f}"
                  + (f"   Fisher p={e['fisher_p']:.4g}" if "fisher_p" in e else ""))
    if audit:
        print("\n" + "=" * 96)
        print("2.7.7/2.7.8  audit quantities along the Jacobian-arm trajectories")
        print("=" * 96)
        print(f"  iterations {audit['n_iterations']}   sigma2_hat undefined on "
              f"{audit['n_sigma2_undefined']} ({audit['frac_sigma2_undefined']:.4f})")
        print(f"  tr J / n  mean {audit['trJ_over_n_mean']:.4f}  max {audit['trJ_over_n_max']:.4f}"
              f"   min (n - tr J) {audit['n_minus_trJ_min']:+.4f}")
        print(f"  negative J_ii total {audit['neg_Jii_total']}   asym mean "
              f"{audit['asym_mean']:.4f}   negeig mean {audit['negeig_mean']:.4f}")
        print(f"  {'t':>3}{'n':>7}{'trJ':>9}{'n-trJ':>9}{'undef':>8}{'negJii':>8}{'asym':>8}{'negeig':>8}")
        for t in sorted(audit["by_iteration"], key=int):
            b = audit["by_iteration"][t]
            print(f"  {t:>3}{b['n']:>7.1f}{b['trJ']:>9.3f}{b['n_minus_trJ']:>9.3f}"
                  f"{b['frac_undefined']:>8.3f}{b['neg_Jii']:>8.2f}{b['asym']:>8.3f}{b['negeig']:>8.3f}")
    print(f"\nSaved: {RESULTS/'exp27_results.json'}")


if __name__ == "__main__":
    main()
