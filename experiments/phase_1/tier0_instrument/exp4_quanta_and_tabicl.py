"""Closing measurements: TabICL reduced basis, and artifact floors for TabICL and TabSwift.

Part A  measure each model's output quantum by a fine scan of m(y + t q_1)
Part B  quantise the wrapped exact GP at that quantum, push it through that
        model's exact probe configuration, read the manufactured asym/negeig
        off a map whose true values are exactly zero
Part C  TabICL reduced-basis A1/A2, amplitude sweep, non-degeneracy

Method for Part A follows diagnostic_a.py: jumps = |diff(preds)| along the scan,
median taken over nonzero jumps. A quantised output concentrates those jumps at
multiples of the step; a continuous output spreads them smoothly around the local
slope times the scan increment. Both statistics are reported so the regime is
distinguishable rather than assumed.
"""
import sys
import json
import time
import numpy as np
from pathlib import Path


class _MockAnalytics:
    def __getattr__(self, name):
        return lambda *a, **k: None


sys.modules['analytics'] = _MockAnalytics()
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from models import load
from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.surrogates import ExactGP
from experiments.phase_1.core.metrics import asym, negeig, negfrac, get_Q, reduced_jacobian

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk1_control_validation import wrap, GP_SIGMA

SEEDS = [42, 100, 200, 300, 400]
Q_SEED = 0
HERE = Path(__file__).resolve().parent
NPZ = HERE / "exp4_tabicl_reduced.npz"
TABPFN_DELTA = 6.87e-4          # diagnostic_a_results.json, for scale


def quantize(f, d):
    return lambda y: np.round(np.asarray(f(y)).ravel() / d) * d


# ------------------------------------------------------------------ Part A
def scan_quantum(model_id, t_range, n_points=501, seed=42):
    """Fine scan of m(y + t q_1) across the probe range; jump statistics."""
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
    Q = get_Q(y, seed=Q_SEED)
    q1 = Q[:, 0]
    model = load(model_id, task="regression", device="cuda")
    ts = np.linspace(-t_range, t_range, n_points)
    preds = []
    for t in ts:
        model.estimator.fit(X, y + t * q1)
        p = model.estimator.predict(X, output_type="mean") if model_id == "tabpfn_v2" \
            else model.estimator.predict(X)
        preds.append(np.asarray(p).ravel())
    preds = np.array(preds)
    del model
    torch.cuda.empty_cache()

    jumps = np.abs(np.diff(preds, axis=0)).ravel()
    nz = jumps[jumps > 0]
    dt = ts[1] - ts[0]
    return {
        "model": model_id, "t_range": t_range, "n_points": n_points,
        "dt": float(dt),
        "frac_exactly_zero": float(np.mean(jumps == 0)),
        "median_nonzero_jump": float(np.median(nz)) if nz.size else 0.0,
        "min_nonzero_jump": float(nz.min()) if nz.size else 0.0,
        "max_jump": float(jumps.max()),
        "p10_nonzero": float(np.percentile(nz, 10)) if nz.size else 0.0,
        "p90_nonzero": float(np.percentile(nz, 90)) if nz.size else 0.0,
        "pred_dtype": str(preds.dtype),
        "output_scale": float(np.abs(preds).mean()),
    }


def show_quantum(q):
    print(f"  {q['model']:<12} scan t in [{-q['t_range']:.0e}, {q['t_range']:.0e}], "
          f"{q['n_points']} points, dt={q['dt']:.3e}")
    print(f"    fraction of exactly-zero jumps  {q['frac_exactly_zero']:.4f}")
    print(f"    median nonzero jump             {q['median_nonzero_jump']:.4e}")
    print(f"    10th / 90th pct nonzero jump    {q['p10_nonzero']:.4e} / {q['p90_nonzero']:.4e}")
    print(f"    min nonzero / max jump          {q['min_nonzero_jump']:.4e} / {q['max_jump']:.4e}")
    print(f"    output dtype {q['pred_dtype']}, mean |m| {q['output_scale']:.4f}")


# ------------------------------------------------------------------ Part B
def artifact_floor(delta, t, dither=False, N=10):
    """Quantised wrapped exact GP: true asym, negeig, negfrac all exactly zero."""
    rows = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        Q = get_Q(y, seed=Q_SEED)
        egp = ExactGP(X, sigma=GP_SIGMA)
        A = Q.T @ egp.jacobian(y) @ Q
        J, _, _ = reduced_jacobian(quantize(wrap(egp.predict), delta), y, Q, t,
                                   dither=dither, N=N,
                                   dither_halfwidth=delta if dither else None, seed=s)
        rows.append({"seed": s, "asym": float(asym(J)), "negeig": float(negeig(J)),
                     "negfrac": float(negfrac(J)),
                     "normF_err": float(np.linalg.norm(J - A, 'fro') / np.linalg.norm(A, 'fro')),
                     "norm_F": float(np.linalg.norm(J, 'fro'))})
    return rows


# ------------------------------------------------------------------ Part C
def tabicl_reduced(ts=(1e-3, 1e-2, 1e-1)):
    model = load("tabicl_v2", task="regression", device="cuda")
    out, store = {}, {}
    for t in ts:
        rows = []
        print(f"\n  t = {t:.0e}", flush=True)
        for s in SEEDS:
            t0 = time.time()
            X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
            Q = get_Q(y, seed=Q_SEED)

            def predict(y_eval):
                model.estimator.fit(X, y_eval)
                return model.estimator.predict(X)

            J, _, _ = reduced_jacobian(predict, y, Q, t)
            m = J.shape[0]
            nd = float(np.linalg.norm(J - np.eye(m), 'fro') / np.linalg.norm(J, 'fro'))
            r = {"seed": s, "asym": float(asym(J)), "negeig": float(negeig(J)),
                 "negfrac": float(negfrac(J)), "norm_F": float(np.linalg.norm(J, 'fro')),
                 "nondeg_ratio": nd}
            rows.append(r)
            store[f"tabicl_t{t:.0e}__J__{s}"] = J
            np.savez_compressed(NPZ, **store)
            print(f"    seed {s:<5} {time.time()-t0:5.1f}s  asym {r['asym']:.6f}  "
                  f"negeig {r['negeig']:.6f}  ||J||_F {r['norm_F']:.4f}  "
                  f"||J-I||/||J|| {nd:.4f}", flush=True)
        out[f"t{t:.0e}"] = rows
    del model
    torch.cuda.empty_cache()
    return out


def main():
    res = {}

    print("=" * 78)
    print("PART A  output quantum, direct scan")
    print("=" * 78)
    res["quantum_tabicl"] = scan_quantum("tabicl_v2", 1e-3)
    show_quantum(res["quantum_tabicl"])
    json.dump(res, open(HERE / "exp4_results.json", "w"), indent=2)
    res["quantum_tabswift"] = scan_quantum("tabswift", 1e-1)
    show_quantum(res["quantum_tabswift"])
    print(f"\n  TabPFN reference (diagnostic_a_results.json): median jump {TABPFN_DELTA:.4e}")
    json.dump(res, open(HERE / "exp4_results.json", "w"), indent=2)

    print("\n" + "=" * 78)
    print("PART B  matched quantised controls (true asym/negeig exactly 0)")
    print("=" * 78)
    d_icl = res["quantum_tabicl"]["median_nonzero_jump"]
    d_swf = res["quantum_tabswift"]["median_nonzero_jump"]
    for tag, delta, t in [("tabicl_floor", d_icl, 1e-3), ("tabswift_floor", d_swf, 1e-1)]:
        rows = artifact_floor(delta, t)
        res[tag] = {"delta": delta, "t": t, "rows": rows}
        a = np.array([r["asym"] for r in rows]); n = np.array([r["negeig"] for r in rows])
        print(f"  {tag:<16} delta={delta:.4e}, t={t:.0e}, no dither")
        print(f"    asym   [" + "  ".join(f"{x:.6e}" for x in a) + f"]  mean {a.mean():.6e}")
        print(f"    negeig [" + "  ".join(f"{x:.6e}" for x in n) + f"]  mean {n.mean():.6e}")
        print(f"    ||J||_F rel err mean {np.mean([r['normF_err'] for r in rows]):.6e}")
    json.dump(res, open(HERE / "exp4_results.json", "w"), indent=2)

    print("\n" + "=" * 78)
    print("PART C  TabICL v2 reduced basis, amplitude sweep")
    print("=" * 78)
    res["tabicl_reduced"] = tabicl_reduced()
    json.dump(res, open(HERE / "exp4_results.json", "w"), indent=2)
    print(f"\nSaved: {HERE/'exp4_results.json'}  and  {NPZ}")


if __name__ == "__main__":
    main()
