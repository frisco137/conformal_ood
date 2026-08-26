"""Chunk 3 — dithered-path controls for TabPFN.

TabPFN's headline numbers all come through the dithered path. That path has
never been validated against a map with a known answer.

The control: an exact GP posterior mean, wrapped in the affine normaliser, with
its output rounded to a grid of step delta matched to TabPFN's measured output
quantisation (median jump 6.87e-4, diagnostic_a_results.json).

    true asym      = 0     (GP posterior mean is symmetric PSD)
    true negeig    = 0
    true curvature = 0     (the inner map is linear)
    true J         = Q^T W Q exactly, by N1(ii)

Everything the pipeline reports above those is manufactured by quantisation.

DITHER WIDTH. Chunk 0 found two conventions in the tree. TabPFN's headline run,
batch2_tabpfn_5seed.py:67, draws U[-delta, +delta]. The control built to
validate it, batch2_c4.py:62, draws U[-delta/2, +delta/2]. Both are run here;
the FAITHFUL one is halfwidth = delta.

Controls only. No model calls, no GPU.
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.surrogates import ExactGP
from experiments.phase_1.core.metrics import (
    asym, negeig, negfrac, get_Q, reduced_jacobian,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk1_control_validation import wrap, TargetedImitator, GP_SIGMA

SEEDS = [42, 100, 200, 300, 400]
DELTA = 6.87e-4          # TabPFN's measured median output jump
T_PFN = 1e-2             # TabPFN's probe amplitude
N_PFN = 10               # TabPFN's dither count
Q_SEED = 0

# TabPFN's measured values, canonical definitions, from results_tabpfn_5seed.json
# (asym there is the halved convention; doubled here)
TABPFN = {"asym": 1.3309, "negeig": 0.4003, "negfrac": 0.3469, "curvature": 66.4157,
          "norm_F": 6.999}


def quantize(f, delta):
    return lambda y_eval: np.round(np.asarray(f(y_eval)).flatten() / delta) * delta


def build(seed, kind, quantized):
    X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=seed)
    Q = get_Q(y, seed=Q_SEED)
    if kind == "gp":
        egp = ExactGP(X, sigma=GP_SIGMA)
        base, G = wrap(egp.predict), egp.jacobian(y)
    else:
        m = TargetedImitator(X, y, seed=seed)
        u = (y - np.mean(y)) / (np.std(y) + 1e-8)
        base = m.predict
        # Was re-derived inline here because TargetedImitator.inner_jacobian
        # shipped -c*outer(v,v) for a map applying -(c/2)*(v.u)*v. Fixed at
        # source in experiments/core/controls.py; the method is bit-identical
        # to the inline form, so this script's recorded output is unchanged.
        G = m.inner_jacobian(u)
    predict = quantize(base, DELTA) if quantized else base
    return y, Q, predict, Q.T @ G @ Q


def measure(kind, quantized, t, dither, halfwidth, N, seeds=SEEDS):
    rows = []
    for s in seeds:
        y, Q, predict, A = build(s, kind, quantized)
        J, curvs, _ = reduced_jacobian(predict, y, Q, t, dither=dither, N=N,
                                       dither_halfwidth=halfwidth, seed=s)
        rows.append({
            "asym": float(asym(J)), "negeig": float(negeig(J)),
            "negfrac": float(negfrac(J)), "curvature": float(np.mean(curvs)),
            "norm_F": float(np.linalg.norm(J, 'fro')),
            "J_err": float(np.linalg.norm(J - A, 'fro') / np.linalg.norm(A, 'fro')),
            "asym_true": float(asym(A)), "negeig_true": float(negeig(A)),
            "norm_F_true": float(np.linalg.norm(A, 'fro')),
        })
    return rows


def line(tag, rows, keys=("asym", "negeig", "negfrac", "curvature", "norm_F", "J_err")):
    print(f"\n  {tag}")
    for k in keys:
        v = np.array([r[k] for r in rows])
        print(f"    {k:<10} [" + "  ".join(f"{x:.6e}" for x in v) +
              f"]  mean {v.mean():.6e}  std {v.std():.3e}")


def main():
    out = {}

    print("=" * 78)
    print("3.1  The control and its analytic truth")
    print("=" * 78)
    y, Q, _, A = build(42, "gp", True)
    print(f"  quantized wrapped ExactGP, delta = {DELTA:.3e}, seed 42")
    print(f"  analytic reduced Jacobian Q^T W Q:")
    print(f"    asym    {asym(A):.6e}   (true value 0)")
    print(f"    negeig  {negeig(A):.6e}   (true value 0)")
    print(f"    negfrac {negfrac(A):.6e}   (true value 0)")
    print(f"    ||J||_F {np.linalg.norm(A,'fro'):.6f}")
    print(f"  true curvature is 0: the inner map is linear, quantisation is the")
    print(f"  only source of a nonzero second difference.")

    print("\n" + "=" * 78)
    print("3.2 / 3.3  TabPFN's exact configuration, t=1e-2, reduced basis")
    print("=" * 78)
    cfgs = [
        ("3.3  dither OFF",                      False, None,      1),
        ("3.2  dither ON,  halfwidth=delta/2  (batch2_c4.py control)", True, DELTA / 2, N_PFN),
        ("3.2  dither ON,  halfwidth=delta    (TabPFN's ACTUAL run)",  True, DELTA,     N_PFN),
    ]
    for tag, dith, hw, N in cfgs:
        rows = measure("gp", True, T_PFN, dith, hw, N)
        line(tag, rows)
        out[tag] = rows

    print("\n  For reference, the same map with NO quantisation (Chunk 1 floor):")
    line("     unquantized wrapped ExactGP, t=1e-2, no dither",
         measure("gp", False, T_PFN, False, None, 1))

    print("\n" + "=" * 78)
    print("3.4  Targeted imitator through the same dithered configuration")
    print("=" * 78)
    print("  Analytic asym is 0.6008 canonical (= 0.3004 in the halved convention")
    print("  the brief quotes). Does a REAL violation survive the dithered path?")
    for lbl, qz in [("unquantized", False), ("quantized at delta", True)]:
        rows = measure("imitator", qz, T_PFN, True, DELTA, N_PFN)
        line(f"3.4  targeted imitator, {lbl}, dither ON halfwidth=delta N=10", rows)
        v = np.array([r["asym"] for r in rows]); vt = np.array([r["asym_true"] for r in rows])
        ne = np.array([r["negeig"] for r in rows]); net = np.array([r["negeig_true"] for r in rows])
        print(f"    asym   measured/analytic ratio  {v.mean()/vt.mean():.6f}")
        print(f"    negeig measured/analytic ratio  {ne.mean()/net.mean():.6f}")
        out[f"3.4_{lbl}"] = rows

    print("\n" + "=" * 78)
    print("3.5  Dither-count sweep on the quantized ExactGP, t=1e-2, halfwidth=delta")
    print("=" * 78)
    print("  Pure quantisation noise should fall as 1/sqrt(N); a real property is flat.")
    Ns = [10, 25, 50, 100, 200]
    sweep = {}
    for N in Ns:
        rows = measure("gp", True, T_PFN, True, DELTA, N)
        sweep[N] = rows
        a = np.array([r["asym"] for r in rows]); c = np.array([r["curvature"] for r in rows])
        ne = np.array([r["negeig"] for r in rows]); je = np.array([r["J_err"] for r in rows])
        print(f"    N={N:<5} asym {a.mean():.6f} +/- {a.std():.6f}   "
              f"negeig {ne.mean():.6f}   curv {c.mean():.4f}   J_err {je.mean():.6f}")
    out["3.5_sweep"] = {str(k): v for k, v in sweep.items()}

    ln = np.log(np.array(Ns, dtype=float))
    for k in ["asym", "negeig", "J_err"]:
        lv = np.log([np.mean([r[k] for r in sweep[N]]) for N in Ns])
        slope = np.polyfit(ln, lv, 1)[0]
        print(f"    log-log slope of {k:<7} vs N : {slope:+.4f}   (pure noise predicts -0.5)")
        out[f"3.5_slope_{k}"] = float(slope)

    print("\n" + "=" * 78)
    print("EXTRA  Amplitude sweep -- would a larger probe have avoided this?")
    print("=" * 78)
    amp = {}
    for t in [1e-2, 3e-2, 1e-1, 3e-1]:
        rows = measure("gp", True, t, True, DELTA, N_PFN)
        amp[f"{t:.0e}"] = rows
        a = np.array([r["asym"] for r in rows]); c = np.array([r["curvature"] for r in rows])
        print(f"    t={t:<6.0e} asym {a.mean():.6f} +/- {a.std():.6f}   curv {c.mean():.6e}   "
              f"J_err {np.mean([r['J_err'] for r in rows]):.6f}")
    out["amplitude_sweep"] = amp

    print("\n" + "=" * 78)
    print("GATE 3  control (zero true asym, zero true curvature) vs TabPFN v2")
    print("=" * 78)
    ctrl = out["3.2  dither ON,  halfwidth=delta    (TabPFN's ACTUAL run)"]
    print(f"    {'quantity':<12} {'control (truth=0)':>20} {'TabPFN v2':>14} {'artifact share':>16}")
    for k in ["asym", "negeig", "negfrac", "curvature"]:
        c = np.mean([r[k] for r in ctrl])
        print(f"    {k:<12} {c:>20.6f} {TABPFN[k]:>14.4f} {100*c/TABPFN[k]:>15.2f}%")

    p = Path(__file__).resolve().parent / "chunk3_results.json"
    json.dump(out, open(p, "w"), indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
