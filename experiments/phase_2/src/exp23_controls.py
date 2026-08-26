"""EXPERIMENT 2.3 controls -- 2.3.9 (analytic maps through the identical pipeline).

A CORRECTION TO THE STATED CONTROL, made before running it.

The brief asks the exact GP to "show chance-level AUC on [the asymmetry] score
and above-chance on the column norm". The second half is not attainable. For an
exact GP the Jacobian is

    J = K(K + sigma^2 I)^-1

which contains no `y` at all. Corrupting labels leaves J bit-identical, so
EVERY Jacobian-derived score -- column norm included -- is unchanged by
corruption and its AUC is exactly chance by construction, not merely near it.
Measured directly: ||J(y_clean) - J(y_corrupt)||_F = 0.

That makes the exact GP a control for one thing only: that no Jacobian score
carries spurious corruption signal. It cannot show that the harness can detect
corruption at all, because for a linear smoother the Jacobian is label-blind.
Only a residual-based score can do that on this map, so the exact GP's residual
is carried as the harness check instead.

The control the brief wants -- Bayes-consistent, symmetric, but with a
label-DEPENDENT Jacobian -- is the HIERARCHICAL GP. Its J is nonlinear in y
(measured: ||J1-J2||_F/||J1||_F = 0.0674 under a 20% flip) and symmetric to
7.5e-16, so its asymmetry score sits at the numerical floor while its column
norm is free to carry signal. Both maps are run.

Writes results/exp23_controls.json and arrays/exp23_controls.npz.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC.parent.parent.parent))
warnings.filterwarnings("ignore")

from experiments.phase_1.core.surrogates import ExactGP, HierarchicalGP        # noqa: E402
from experiments.phase_2.src.paths import RESULTS, ARRAYS              # noqa: E402
from experiments.phase_2.src.estimators import jsonable                # noqa: E402
from experiments.phase_2.src.exp23_corrupt import (                    # noqa: E402
    SEEDS, SCHEMES, RATES, DIDS, corrupt_indices, apply_corruption, scores_from_J,
)

GP_SIGMA = 0.5


def run_map(name, X, y):
    if name == "exactgp":
        g = ExactGP(X, sigma=GP_SIGMA)
        return g.jacobian(y), g.predict(y)
    g = HierarchicalGP(X, sigma=GP_SIGMA)
    return g.jacobian(y), g.predict(y)


def main():
    data = np.load(ARRAYS / "chunk2_data.npz")
    manifest = {r["did"]: r for r in json.load(open(RESULTS / "chunk2_datasets.json"))["accepted"]}
    rows, arrays = [], {}

    for name in ["exactgp", "hiergp"]:
        print("\n" + "=" * 78); print(f"CONTROL {name}"); print("=" * 78)
        for did in DIDS:
            for s in SEEDS:
                X = data[f"{did}__{s}__X_ctx"]
                y0 = data[f"{did}__{s}__y_ctx"].astype(float)
                n = len(y0)
                J_clean, _ = run_map(name, X, y0)
                for rate in RATES:
                    idx, k = corrupt_indices(did, s, rate, n)
                    for scheme in SCHEMES:
                        yc, changed = apply_corruption(y0, idx, scheme, did, s, rate)
                        J, m_ctx = run_map(name, X, yc)
                        sc, lam = scores_from_J(J, yc)
                        tag = f"{name}__{did}__{s}__{scheme}__{int(rate*100)}"
                        arrays[f"{tag}__J"] = J.astype(np.float32)
                        arrays[f"{tag}__y_corr"] = yc
                        arrays[f"{tag}__m_ctx"] = m_ctx
                        arrays[f"{tag}__idx"] = idx
                        rows.append({
                            "model": name, "did": did, "name": manifest[did]["name"],
                            "seed": s, "scheme": scheme, "rate": rate,
                            "n_ctx": n, "n_corrupt": int(k), "n_changed": int(changed),
                            "corrupt_idx": idx.tolist(), "lam_min": lam,
                            "trJ": float(np.trace(J)),
                            "asym_rc_absmax": float(np.max(np.abs(sc["asym_rc"]))),
                            "J_change_vs_clean_rel": float(
                                np.linalg.norm(J - J_clean) / np.linalg.norm(J_clean)),
                            "asym_frob": float(np.linalg.norm(J - J.T)
                                               / np.linalg.norm(J)),
                        })
            print(f"  did={did} done", flush=True)
        d = [r for r in rows if r["model"] == name]
        print(f"  {name}: J change vs clean, relative Frobenius -- "
              f"mean {np.mean([r['J_change_vs_clean_rel'] for r in d]):.6e}  "
              f"max {np.max([r['J_change_vs_clean_rel'] for r in d]):.6e}")
        print(f"  {name}: asym(J) -- mean {np.mean([r['asym_frob'] for r in d]):.6e}")
        print(f"  {name}: max |rownorm - colnorm| -- mean "
              f"{np.mean([r['asym_rc_absmax'] for r in d]):.6e}")

    np.savez_compressed(ARRAYS / "exp23_controls.npz", **arrays)
    with open(RESULTS / "exp23_controls.json", "w") as f:
        json.dump(jsonable({"_config": {"dids": DIDS, "seeds": SEEDS,
                                        "schemes": SCHEMES, "rates": RATES,
                                        "gp_sigma": GP_SIGMA}, "rows": rows}), f, indent=2)
    print(f"\nSaved: {RESULTS/'exp23_controls.json'}")


if __name__ == "__main__":
    main()
