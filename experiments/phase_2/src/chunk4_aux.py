"""CHUNK 4 -- SURE consistency (4.1) and directional monotonicity (4.2).

4.1 reads the stored context Jacobians and needs no model calls.
4.2 needs fresh forward passes, but only 11 per split, so it runs on every
    (model, dataset, split) rather than a subset.

4.1 -- what is being compared, stated because the two sides are not the same object
-----------------------------------------------------------------------------------
    R_SURE = ||m(y) - y||^2 + 2 sigma^2 tr J - n sigma^2

estimates the IN-SAMPLE risk ||m(y) - f||^2 on the context rows. Held-out
squared error estimates the OUT-OF-SAMPLE risk plus noise:
E(y_test - m_test)^2 = E(f_test - m_test)^2 + sigma^2. The comparable pair is
therefore

    per-point SURE            R_SURE / n_ctx
    per-point held-out risk   mean((y_test - m_test)^2) - sigma^2_cal

and the gap between them is reported as `sure_gap`. sigma^2 is the
conformal-calibration residual variance mean(r_cal^2), per the brief. It is a
different quantity from `sigma2_hat`; both are reported.

The two risks are not expected to be equal -- out-of-sample risk generally
exceeds in-sample risk -- so the gap is reported as a measured quantity, not as
an error in either.

4.2 -- directional monotonicity
-------------------------------
v = unit eigenvector of lambda_min(sym(Q^T J Q)); u = Q v, a direction in
context-label space with ||u|| = 1. For any posterior mean, A2 gives
u^T J u = v^T (Q^T J Q) v >= 0. Measured as a central difference

    D(t) = u . (m(y + t u) - m(y - t u)) / (2 t)   ->   u^T J u

at five amplitudes t, each a stated fraction of ||y - mean(y)|| so the probe is
comparable across datasets whose targets differ by three orders of magnitude.

Controls, through the identical procedure: exact GP (u^T J u >= 0 necessarily,
J = K(K+sigma^2 I)^-1 is PSD) and the targeted imitator from
tier0_instrument/chunk1_control_validation.py (constructed non-PSD; must go
negative). The controls are imported, not reimplemented.

Writes chunk4_results.json.
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
sys.path.insert(0, str(SRC.parent.parent.parent))              # repo root
warnings.filterwarnings("ignore")

from experiments.phase_2.src.paths import RESULTS, ARRAYS, TIER0    # noqa: E402

sys.path.insert(0, str(TIER0))     # wrap / TargetedImitator controls

from experiments.core.metrics import get_Q, negeig, asym          # noqa: E402
from experiments.core.context import generate_audit_context       # noqa: E402
from experiments.core.surrogates import ExactGP                   # noqa: E402
from experiments.phase_2.src.estimators import spearman, jsonable # noqa: E402
from chunk1_control_validation import TargetedImitator, wrap, GP_SIGMA   # noqa: E402

MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]
AUDIT_MEAN_SY = 1.457782
H_FRAC_BY_MODEL = {"tabicl_v2": 1e-3 / AUDIT_MEAN_SY,
                   "tabpfn_v2": 1e-1 / AUDIT_MEAN_SY,
                   "tabswift": 1e-1 / AUDIT_MEAN_SY}
FRACS = [0.003, 0.01, 0.03, 0.1, 0.3]      # of ||y - mean(y)||
Q_SEED = 0
AUDIT_SEEDS = [42, 100, 200, 300, 400]


def load_model_outputs(mid):
    jp = RESULTS / f"chunk2_estimator_{mid}.json"
    npz = ARRAYS / f"chunk2_arrays_{mid}.npz"
    if not jp.exists() or not npz.exists():
        return None, None
    return json.load(open(jp)), np.load(npz)


# ---------------------------------------------------------------------- 4.1
def item_4_1():
    print("=" * 78)
    print("4.1  SURE vs held-out squared error")
    print("=" * 78)
    out = {}
    for mid in MODELS:
        meta, z = load_model_outputs(mid)
        if meta is None:
            print(f"  {mid}: chunk 2 output absent -- NOT MEASURED")
            out[mid] = None
            continue
        rows = []
        for r in meta["rows"]:
            tag = f"{mid}__{r['did']}__{r['seed']}"
            y_ctx, m_ctx = z[f"{tag}__y_ctx"], z[f"{tag}__m_ctx"]
            y_test, m_test = z[f"{tag}__y_test"], z[f"{tag}__m_test"]
            J = z[f"{tag}__J_ctx"].astype(np.float64)
            r_cal = z[f"{tag}__r_cal"]
            n = len(y_ctx)
            trJ = float(np.trace(J))
            s2_cal = float(np.mean(r_cal ** 2))
            rss = float(np.sum((m_ctx - y_ctx) ** 2))
            sure = rss + 2.0 * s2_cal * trJ - n * s2_cal
            heldout_mse = float(np.mean((y_test - m_test) ** 2))
            rows.append({
                "did": r["did"], "name": r["name"], "seed": r["seed"],
                "n_ctx": n, "trJ": trJ, "n_neg_Jii": int(np.sum(np.diag(J) < 0)),
                "sigma2_cal": s2_cal, "sigma2_hat": r["sigma2_hat"],
                "rss_ctx": rss,
                "R_SURE": sure, "R_SURE_per_point": sure / n,
                "heldout_mse": heldout_mse,
                "heldout_risk_per_point": heldout_mse - s2_cal,
                "sure_gap": sure / n - (heldout_mse - s2_cal),
            })
        gaps = np.array([x["sure_gap"] for x in rows])
        negs = np.array([x["n_neg_Jii"] for x in rows], float)
        trs = np.array([x["trJ"] for x in rows])
        out[mid] = {
            "rows": rows,
            "n_splits": len(rows),
            "sure_gap_mean": float(np.mean(gaps)), "sure_gap_sd": float(np.std(gaps, ddof=1)),
            "trJ_mean": float(np.mean(trs)), "trJ_min": float(np.min(trs)),
            "trJ_max": float(np.max(trs)),
            "n_neg_Jii_total": int(negs.sum()),
            "n_splits_with_neg": int(np.sum(negs > 0)),
            "corr_gap_vs_negcount": {
                "spearman": spearman(negs, gaps),
                "pearson": (float(np.corrcoef(negs, gaps)[0, 1])
                            if np.ptp(negs) > 0 and np.ptp(gaps) > 0 else None),
                "n_splits": len(rows),
                "note": ("correlation over the splits available; no causal reading. "
                         "Undefined when the negative-diagonal count is constant.")},
        }
        o = out[mid]
        print(f"  {mid:<11} splits {o['n_splits']:3d}  trJ in [{o['trJ_min']:.2f}, "
              f"{o['trJ_max']:.2f}]  neg Jii total {o['n_neg_Jii_total']:4d} "
              f"({o['n_splits_with_neg']} splits)")
        print(f"              SURE gap mean {o['sure_gap_mean']:+.5g}  sd {o['sure_gap_sd']:.5g}"
              f"   spearman(gap, neg count) "
              f"{o['corr_gap_vs_negcount']['spearman'] if o['corr_gap_vs_negcount']['spearman'] is not None else float('nan'):+.4f}")
    return out


# ---------------------------------------------------------------------- 4.2
def leading_negative_direction(J, y):
    Q = get_Q(y, seed=Q_SEED)
    Jr = Q.T @ J @ Q
    S = (Jr + Jr.T) / 2.0
    w, V = np.linalg.eigh(S)
    v = V[:, 0]
    u = Q @ v
    u = u / np.linalg.norm(u)
    return u, float(w[0]), float(negeig(Jr)), float(asym(Jr))


def probe_direction(predict, X, y, u, fracs):
    """D(t) = u.(m(y+tu) - m(y-tu)) / 2t, plus one-sided moves for 4.2d."""
    scale = float(np.linalg.norm(y - np.mean(y)))
    m0 = np.asarray(predict(X, y, X)).ravel()
    rows = []
    for fr in fracs:
        t = fr * scale
        mp = np.asarray(predict(X, y + t * u, X)).ravel()
        mm = np.asarray(predict(X, y - t * u, X)).ravel()
        rows.append({
            "frac_of_centred_norm": fr, "t_abs": t,
            "D_central": float(u @ (mp - mm) / (2 * t)),
            "label_raise": t,                                   # = ||t u||, u unit
            "pred_move_plus": float(u @ (mp - m0)),
            "pred_move_minus": float(u @ (mm - m0)),
        })
    return rows, scale


def item_4_2_controls():
    print("\n" + "=" * 78)
    print("4.2c  controls: exact GP (must be >= 0) and targeted imitator (must go < 0)")
    print("=" * 78)
    out = {}
    for name in ["exactgp_wrapped", "targeted_imitator"]:
        rows = []
        for s in AUDIT_SEEDS:
            X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
            if name == "exactgp_wrapped":
                egp = ExactGP(X, sigma=GP_SIGMA)
                inner = wrap(egp.predict)
                J = egp.jacobian(y)
            else:
                ti = TargetedImitator(X, y, seed=s)
                inner = ti.predict
                u_std = (y - np.mean(y)) / (np.std(y) + 1e-8)
                # CORRECTED analytic Jacobian. TargetedImitator.inner applies
                #   -(c/2) * (v.u) * v          -> Jacobian -(c/2) v v^T
                # while TargetedImitator.inner_jacobian returns -c * outer(v,v),
                # twice the rank-one term the map applies. This is the same
                # doubling FINAL_NUMBERS.md Section 2.3 records for
                # audit_phase2_imitator.py's get_analytic_jacobian, and it is
                # corrected here the same way rather than by editing the tier-0
                # control, which other results depend on.
                #
                # With the as-coded Jacobian, lambda_min is -0.335129 on seed 42
                # while the measured directional derivative of the map is
                # -0.029472. With the correction both read -0.088180, and the
                # full vector agrees to 7.1e-12. -0.088180 also reproduces
                # FINAL_NUMBERS 2.3's seed-42 negeig of 0.088194.
                J = (ti.egp.jacobian(u_std) + ti.M_anti
                     - (ti.c / 2.0) * np.outer(ti.v, ti.v))
            u, lam, ne, az = leading_negative_direction(J, y)
            predict = lambda Xt, yt, Xq: inner(yt)               # noqa: E731
            pr, scale = probe_direction(predict, X, y, u, FRACS)
            rows.append({"seed": s, "lambda_min": lam, "negeig": ne, "asym": az,
                         "centred_norm": scale, "probes": pr})
            print(f"  {name:<18} seed {s:<5} lam_min {lam:+.5f}  negeig {ne:.5f}  "
                  + "  ".join(f"D({p['frac_of_centred_norm']})={p['D_central']:+.4f}"
                              for p in pr[:3]))
        out[name] = rows
    return out


def item_4_2_models(which):
    print("\n" + "=" * 78)
    print("4.2a/b/d  directional monotonicity on the frozen models")
    print("=" * 78)
    import torch
    from models import load

    data = np.load(ARRAYS / "chunk2_data.npz")
    out = {}
    for mid in which:
        meta, z = load_model_outputs(mid)
        if meta is None:
            print(f"  {mid}: chunk 2 output absent -- NOT MEASURED")
            out[mid] = None
            continue
        model = load(mid, task="regression", device="cuda")

        def predict(X_train, y_train, X_test, _m=model, _id=mid):
            _m.estimator.fit(np.asarray(X_train, float), np.asarray(y_train, float))
            if _id == "tabpfn_v2":
                return np.asarray(_m.estimator.predict(np.asarray(X_test, float),
                                                       output_type="mean")).ravel()
            return np.asarray(_m.estimator.predict(np.asarray(X_test, float))).ravel()

        rows = []
        for r in meta["rows"]:
            t0 = time.time()
            tag = f"{mid}__{r['did']}__{r['seed']}"
            y = z[f"{tag}__y_ctx"].astype(np.float64)
            J = z[f"{tag}__J_ctx"].astype(np.float64)
            X = data[f"{r['did']}__{r['seed']}__X_ctx"]
            u, lam, ne, az = leading_negative_direction(J, y)
            pr, scale = probe_direction(predict, X, y, u, FRACS)
            rows.append({"did": r["did"], "name": r["name"], "seed": r["seed"],
                         "lambda_min": lam, "negeig": ne, "asym_reduced": az,
                         "centred_norm": scale, "probes": pr,
                         "seconds": round(time.time() - t0, 1)})
            print(f"  {mid:<11} did={r['did']:<6} seed {r['seed']:<5} "
                  f"lam_min {lam:+.5f}  negeig {ne:.5f}  "
                  + "  ".join(f"D({p['frac_of_centred_norm']})={p['D_central']:+.4f}"
                              for p in pr), flush=True)
            with open(RESULTS / f"chunk4_direction_{mid}.json", "w") as f:
                json.dump(jsonable(rows), f, indent=2)
        out[mid] = rows
        del model
        torch.cuda.empty_cache()
    return out


def summarise_4_2(models_out, controls):
    """Sign summary at each amplitude."""
    summ = {}
    for name, rows in list(controls.items()) + list(models_out.items()):
        if not rows:
            summ[name] = None
            continue
        per_frac = {}
        for i, fr in enumerate(FRACS):
            D = np.array([r["probes"][i]["D_central"] for r in rows])
            per_frac[str(fr)] = {"n": len(D), "n_negative": int(np.sum(D < 0)),
                                 "mean": float(np.mean(D)), "min": float(np.min(D)),
                                 "max": float(np.max(D))}
        lam = np.array([r["lambda_min"] for r in rows])
        summ[name] = {"per_frac": per_frac, "n_rows": len(rows),
                      "lambda_min_mean": float(np.mean(lam)),
                      "lambda_min_min": float(np.min(lam)),
                      "n_lambda_min_negative": int(np.sum(lam < 0))}
    return summ


def main():
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or MODELS
    res = {"_config": {"fracs_of_centred_norm": FRACS, "Q_seed": Q_SEED,
                       "h_frac_by_model": H_FRAC_BY_MODEL, "audit_seeds": AUDIT_SEEDS,
                       "sure_sigma2": "conformal-calibration residual variance mean(r_cal^2)"}}
    res["4.1_sure"] = item_4_1()
    res["4.2c_controls"] = item_4_2_controls()
    res["4.2_models"] = item_4_2_models(which)
    res["4.2_summary"] = summarise_4_2(res["4.2_models"], res["4.2c_controls"])

    print("\n" + "=" * 78)
    print("4.2 summary -- sign of D(t) = u^T J u by amplitude")
    print("=" * 78)
    for name, s in res["4.2_summary"].items():
        if s is None:
            print(f"  {name:<18} NOT MEASURED")
            continue
        line = "  ".join(f"{fr}:{s['per_frac'][str(fr)]['n_negative']}/{s['per_frac'][str(fr)]['n']}"
                         for fr in FRACS)
        print(f"  {name:<18} lam_min mean {s['lambda_min_mean']:+.5f}  "
              f"negative in {s['n_lambda_min_negative']}/{s['n_rows']} rows   "
              f"D(t)<0 counts  {line}")

    p = RESULTS / "chunk4_results.json"
    with open(p, "w") as f:
        json.dump(jsonable(res), f, indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
