"""Experiment 3 — re-probe TabPFN v2 (and TabSwift) at t = 1e-1, reduced basis.

Chunk 3 measured the quantisation artifact on a control whose true asymmetry is
exactly zero, as a function of probe amplitude:

    t=1e-02  artifact asym 0.096875     t=1e-01  artifact asym 0.009744
    t=3e-02  artifact asym 0.032778     t=3e-01  artifact asym 0.003267

asym proportional to 1/t, exactly as a fixed quantisation error propagated
through (m+ - m-)/2t predicts. Chunk 3 also showed N is the wrong knob: asym
decays as N^-0.30, so removing the artifact by dither count would need N ~ 2e4.

All asym values here are canonical ||J - J^T||_F / ||J||_F. Every number in
tier2_audit/ is half this (RESULTS_REPORT.md 5.3).

Reduced Jacobians are saved to exp3_reduced_jacobians.npz.
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
from experiments.phase_1.core.metrics import (asym, negeig, negfrac, get_Q, reduced_jacobian)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk1_control_validation import wrap, GP_SIGMA

SEEDS = [42, 100, 200, 300, 400]
DELTA = 6.87e-4
Q_SEED = 0
HERE = Path(__file__).resolve().parent
NPZ = HERE / "exp3_reduced_jacobians.npz"


def quantize(f, d):
    return lambda y: np.round(np.asarray(f(y)).ravel() / d) * d


def metrics(J):
    return {"asym": float(asym(J)), "negeig": float(negeig(J)),
            "negfrac": float(negfrac(J)), "norm_F": float(np.linalg.norm(J, 'fro'))}


def run_model(model_id, t, dither, N, store, tag):
    model = load(model_id, task="regression", device="cuda")
    rows = []
    for s in SEEDS:
        t0 = time.time()
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        Q = get_Q(y, seed=Q_SEED)

        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            if model_id == "tabpfn_v2":
                return model.estimator.predict(X, output_type="mean")
            return model.estimator.predict(X)

        J, _, _ = reduced_jacobian(predict, y, Q, t, dither=dither, N=N,
                                   dither_halfwidth=DELTA if dither else None, seed=s)
        m = metrics(J); m["seed"] = s
        rows.append(m)
        store[f"{tag}__J__{s}"] = J
        np.savez_compressed(NPZ, **store)
        print(f"    seed {s:<5} {time.time()-t0:6.1f}s  asym {m['asym']:.6f}  "
              f"negeig {m['negeig']:.6f}  negfrac {m['negfrac']:.4f}  "
              f"||J||_F {m['norm_F']:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def show(tag, rows):
    print(f"  {tag}")
    for k in ["asym", "negeig", "norm_F"]:
        v = np.array([r[k] for r in rows])
        print(f"    {k:<8} [" + "  ".join(f"{x:.6f}" for x in v) +
              f"]  mean {v.mean():.6f}  std {v.std():.6f}")


def control_artifact(t):
    """Quantized wrapped ExactGP: true asym, negeig, negfrac all exactly zero."""
    rows = []
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        Q = get_Q(y, seed=Q_SEED)
        p = quantize(wrap(ExactGP(X, sigma=GP_SIGMA).predict), DELTA)
        J, _, _ = reduced_jacobian(p, y, Q, t, dither=True, N=10,
                                   dither_halfwidth=DELTA, seed=s)
        rows.append(metrics(J))
    return rows


def main():
    out, store = {}, {}

    print("=" * 78)
    print("3.4  Quantised-ExactGP artifact floor (true values all exactly 0)")
    print("=" * 78)
    for t in [1e-2, 1e-1]:
        rows = control_artifact(t)
        out[f"artifact_t{t:.0e}"] = rows
        show(f"quantised wrapped ExactGP, t={t:.0e}, dither halfwidth=delta N=10", rows)

    print("\n" + "=" * 78)
    print("3.3  TabPFN v2 amplitude sweep, no dither (plateau shape, 5 seeds each)")
    print("=" * 78)
    for t in [1e-2, 3e-2, 1e-1, 3e-1]:
        print(f"  t = {t:.0e}", flush=True)
        rows = run_model("tabpfn_v2", t, False, 1, store, f"tabpfn_nodither_t{t:.0e}")
        out[f"tabpfn_nodither_t{t:.0e}"] = rows
        json.dump(out, open(HERE / "exp3_reprobe.json", "w"), indent=2)

    print("\n" + "=" * 78)
    print("3.1  TabPFN v2 at t=1e-1, dither halfwidth=delta, N=10  -- HEADLINE")
    print("=" * 78)
    out["tabpfn_dither_t1e-01"] = run_model("tabpfn_v2", 1e-1, True, 10, store,
                                            "tabpfn_dither_t1e-01")
    json.dump(out, open(HERE / "exp3_reprobe.json", "w"), indent=2)

    print("\n" + "=" * 78)
    print("3.5  TabSwift amplitude sweep, no dither")
    print("=" * 78)
    for t in [1e-2, 1e-1, 1.0]:
        print(f"  t = {t:.0e}", flush=True)
        rows = run_model("tabswift", t, False, 1, store, f"tabswift_t{t:.0e}")
        out[f"tabswift_t{t:.0e}"] = rows
        json.dump(out, open(HERE / "exp3_reprobe.json", "w"), indent=2)

    json.dump(out, open(HERE / "exp3_reprobe.json", "w"), indent=2)
    print(f"\nSaved: {HERE / 'exp3_reprobe.json'}  and  {NPZ}")


if __name__ == "__main__":
    main()
