"""Experiment 2 — mechanism profiling in AMBIENT coordinates.

Reduced-basis profiling was abandoned because it has no power: Nadaraya-Watson,
a map that is exactly row-scaled symmetric by construction, profiles to 3.5e-16
in ambient coordinates and to 0.245 after projection -- row profiling raises its
asymmetry from 0.249 to 0.260 (CONTROL_VALIDATION.md, Chunk 1). A test that
cannot recognise a known kernel smoother cannot exclude one.

The two systems:
    column   J_ij * sigma_j^2 = J_ji * sigma_i^2     heteroscedastic Bayes, J = Cov(f|y) Sigma^-1
    row      d_i * J_ij       = d_j * J_ji           kernel smoother / vote,  J = D^-1 K

Both are solved in log space by weighted least squares. The residual is computed
explicitly as ||A log s - b||_2 / ||b||_2; numpy's lstsq returns an EMPTY
residual array on this design (columns sum to zero, rank n-1), which is why
item3.py:95 printed 0.00e+00 for every fit it ever reported.

Row and column residuals are identical by construction: b_row = -b_col, so the
fit is the negation and the norms coincide. Only the post-profile asym can
distinguish the two mechanisms. The residual must not be reported as
discriminating between them.

Reads the ambient Jacobians measured by exp12_ambient_jacobians.py.
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.phase_1.core.context import generate_audit_context
from experiments.phase_1.core.surrogates import ExactGP, NadarayaWatson
from experiments.phase_1.core.metrics import asym, central_jacobian, profile_jacobian

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunk1_control_validation import wrap, TargetedImitator, GP_SIGMA

SEEDS = [42, 100, 200, 300, 400]
HERE = Path(__file__).resolve().parent


def prof(J, mode):
    p = profile_jacobian(J, mode, weighted=True)
    return p["asym"], p["residual_rel"], p["profile"]


def summarise(tag, rows):
    print(f"\n  {tag}")
    for k in ["raw_asym", "col_asym", "col_resid", "row_asym", "row_resid"]:
        v = np.array([r[k] for r in rows], dtype=float)
        print(f"    {k:<10} [" + "  ".join(f"{x:.6e}" for x in v) +
              f"]  mean {v.mean():.6e}  std {v.std():.3e}")
    pr = np.concatenate([r["row_profile"] for r in rows])
    pc = np.concatenate([r["col_profile"] for r in rows])
    print(f"    recovered row profile   range [{pr.min():.4e}, {pr.max():.4e}]  "
          f"all positive: {bool((pr > 0).all())}")
    print(f"    recovered col profile   range [{pc.min():.4e}, {pc.max():.4e}]  "
          f"all positive: {bool((pc > 0).all())}")


def analyse(J_by_seed):
    rows = []
    for s, J in J_by_seed.items():
        ca, cr, cp = prof(J, "column")
        ra, rr, rp = prof(J, "row")
        rows.append({"seed": s, "raw_asym": float(asym(J)),
                     "col_asym": float(ca), "col_resid": float(cr),
                     "row_asym": float(ra), "row_resid": float(rr),
                     "col_profile": cp, "row_profile": rp})
    return rows


def jsonable(rows):
    return [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
            for r in rows]


def main():
    out = {}

    # ---------------------------------------------------------------- 2.4 / 2.5
    print("=" * 78)
    print("2.4  POSITIVE CONTROL  Nadaraya-Watson ambient, J = D^-1 K")
    print("=" * 78)
    print("  Row-scaled symmetric by construction. Gate: row-profiled asym < 1e-12.")
    nw = {}
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        nw[s] = NadarayaWatson(X, lengthscale=1.0).jacobian(y)
    rows_nw = analyse(nw)
    summarise("Nadaraya-Watson (ambient, analytic J)", rows_nw)
    out["nw_ambient"] = jsonable(rows_nw)
    gate_nw = max(r["row_asym"] for r in rows_nw)

    print("\n" + "=" * 78)
    print("2.5  NEGATIVE CONTROLS")
    print("=" * 78)
    egp_J, imi_J = {}, {}
    for s in SEEDS:
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        egp_J[s] = central_jacobian(wrap(ExactGP(X, sigma=GP_SIGMA).predict), y, h=1e-3)
        imi_J[s] = central_jacobian(TargetedImitator(X, y, seed=s).predict, y, h=1e-3)
    rows_egp = analyse(egp_J); rows_imi = analyse(imi_J)
    summarise("wrapped ExactGP  (symmetric: profiling must not inflate it)", rows_egp)
    summarise("targeted imitator (not of diagonal-scaling form: must survive)", rows_imi)
    out["exactgp_wrapped_ambient"] = jsonable(rows_egp)
    out["targeted_imitator_ambient"] = jsonable(rows_imi)

    infl = max(r["col_asym"] / r["raw_asym"] for r in rows_egp)
    print(f"\n  wrapped ExactGP worst inflation factor (profiled / raw): {infl:.4f}")
    print("  Chunk 1 reference: UNWEIGHTED profiling inflated the same control 17x")
    print("  (3.1e-12 -> 5.3e-11); the weighted form is used throughout here.")

    print(f"\n  GATE 2: NW ambient row-profiled asym  {gate_nw:.3e}  "
          f"(required < 1e-12)  -> {'PASS' if gate_nw < 1e-12 else 'FAIL'}")
    print(f"          wrapped ExactGP inflation      {infl:.4f}  "
          f"(required < 2)      -> {'PASS' if infl < 2 else 'FAIL'}")
    out["gate2"] = {"nw_row_asym_worst": float(gate_nw), "egp_inflation_worst": float(infl)}

    if not (gate_nw < 1e-12 and infl < 2):
        print("\n  Gate 2 not met. Model results are not reported.")
        json.dump(out, open(HERE / "exp2_ambient_profiling.json", "w"), indent=2)
        return

    # -------------------------------------------------------------------- 2.6
    print("\n" + "=" * 78)
    print("2.6  MODELS, ambient Jacobians from exp12_ambient_jacobians.npz")
    print("=" * 78)
    npz_path = HERE / "exp12_ambient_jacobians.npz"
    if not npz_path.exists():
        print("  exp12_ambient_jacobians.npz not present -- NOT MEASURED")
        json.dump(out, open(HERE / "exp2_ambient_profiling.json", "w"), indent=2)
        return
    z = np.load(npz_path)
    meta = json.load(open(HERE / "exp12_ambient_meta.json"))
    tags = sorted({k.split("__")[0] for k in z.files})
    for tag in tags:
        J_by_seed = {s: z[f"{tag}__J__{s}"] for s in SEEDS if f"{tag}__J__{s}" in z.files}
        if not J_by_seed:
            continue
        m = meta.get(tag, {})
        cfg = f"h={m.get('h')}, dither={m.get('dither')}" + \
              (f", N={m.get('N')}" if m.get("dither") else "")
        rows = analyse(J_by_seed)
        summarise(f"{tag}   [{cfg}]   seeds {sorted(J_by_seed)}", rows)
        out[tag] = {"config": cfg, "rows": jsonable(rows)}

    json.dump(out, open(HERE / "exp2_ambient_profiling.json", "w"), indent=2)
    print(f"\nSaved: {HERE / 'exp2_ambient_profiling.json'}")


if __name__ == "__main__":
    main()
