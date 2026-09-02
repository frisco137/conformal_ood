"""T0.2 -- determinism of the audit path on TabICL. Closes Phase 2's E0.1 gap.

Phase 2 listed E0.1 as NOT MEASURED: "requires a GPU refit-twice pass on the audit
path". This is that pass. Everything downstream in Phase 3 assumes the instrument
returns the same number twice; if it does not, every violation is partly jitter
and halt condition H1 fires.

WHAT IS RUN
-----------
The audit path exactly as Phase 1 runs it -- TabICL v2, the five audit contexts,
t = 1e-3, no dither, reduced basis Q = get_Q(y, seed=0) -- twice, in two
independent model loads, and the two results are differenced.

Two runs are compared at three levels, because they fail differently:
  m(y)      the forward pass alone. Isolates decoder / kernel nondeterminism.
  Q^T J Q   the full reduced Jacobian, 98 columns x 2 evaluations each.
  asym / negeig   the reported scalars. These are what the acceptance is on.

ACCEPTANCE (plan section 2, T0.2)
  relative difference in asym and negeig <= 1% of the reported values.
FAIL -> halt condition H1.

Writes results/t0_2_determinism.json.
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

import torch                                                        # noqa: E402
from models import load                                             # noqa: E402
from experiments.phase_1.core.context import generate_audit_context  # noqa: E402
from experiments.phase_1.core.metrics import (                       # noqa: E402
    asym, get_Q, negeig, reduced_jacobian,
)

SEEDS = [42, 100, 200, 300, 400]
T = 1e-3
Q_SEED = 0
MODEL = "tabicl_v2"


def one_pass(tag):
    """A complete independent pass: fresh model load, all five contexts."""
    model = load(MODEL, task="regression", device="cuda")
    out = {}
    for s in SEEDS:
        t0 = time.time()
        X, y, _ = generate_audit_context(n=100, d=5, sigma=1.0, seed=s)
        Q = get_Q(y, seed=Q_SEED)

        def predict(y_eval):
            model.estimator.fit(X, y_eval)
            return model.estimator.predict(X)

        m = np.asarray(predict(y)).ravel()
        J, _, _ = reduced_jacobian(predict, y, Q, T)
        out[s] = {"m": m, "J": J,
                  "asym": float(asym(J)), "negeig": float(negeig(J))}
        print(f"  [{tag}] seed {s:<5} {time.time()-t0:6.1f}s  "
              f"asym {out[s]['asym']:.9f}  negeig {out[s]['negeig']:.9f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out


def relerr(a, b):
    d = abs(a - b)
    return d / abs(b) if abs(b) > 0 else (0.0 if d == 0 else float("inf"))


def main():
    print("=" * 78)
    print("T0.2  determinism of the audit path -- TabICL v2, 2 independent passes")
    print(f"      t={T:.0e}, no dither, Q seed {Q_SEED}, seeds {SEEDS}")
    print("=" * 78)

    print("\nPASS 1")
    A = one_pass("1")
    print("\nPASS 2")
    B = one_pass("2")

    rows, store = [], {}
    for s in SEEDS:
        a, b = A[s], B[s]
        dm = float(np.max(np.abs(a["m"] - b["m"])))
        dm_rel = dm / float(np.max(np.abs(b["m"])))
        dJ = float(np.max(np.abs(a["J"] - b["J"])))
        dJ_rel = float(np.linalg.norm(a["J"] - b["J"], "fro")
                       / np.linalg.norm(b["J"], "fro"))
        r = {"seed": s,
             "m_max_abs_diff": dm, "m_max_rel_diff": dm_rel,
             "J_max_abs_diff": dJ, "J_rel_fro_diff": dJ_rel,
             "asym_1": a["asym"], "asym_2": b["asym"],
             "asym_rel_diff": relerr(a["asym"], b["asym"]),
             "negeig_1": a["negeig"], "negeig_2": b["negeig"],
             "negeig_rel_diff": relerr(a["negeig"], b["negeig"]),
             "bit_identical_m": bool(np.array_equal(a["m"], b["m"])),
             "bit_identical_J": bool(np.array_equal(a["J"], b["J"]))}
        rows.append(r)
        store[f"m1__{s}"] = a["m"]; store[f"m2__{s}"] = b["m"]

    worst_asym = max(r["asym_rel_diff"] for r in rows)
    worst_negeig = max(r["negeig_rel_diff"] for r in rows)
    passed = worst_asym <= 0.01 and worst_negeig <= 0.01

    print("\n" + "=" * 78)
    print("PER SEED")
    print("=" * 78)
    print(f"  {'seed':<6}{'max|dm|':>12}{'rel':>10}{'||dJ||_F/||J||':>16}"
          f"{'asym rel':>12}{'negeig rel':>13}{'bit-identical':>15}")
    for r in rows:
        bi = "m and J" if r["bit_identical_J"] else ("m only" if r["bit_identical_m"] else "no")
        print(f"  {r['seed']:<6}{r['m_max_abs_diff']:>12.3e}{r['m_max_rel_diff']:>10.2e}"
              f"{r['J_rel_fro_diff']:>16.3e}{r['asym_rel_diff']:>12.3e}"
              f"{r['negeig_rel_diff']:>13.3e}{bi:>15}")

    print("\n" + "=" * 78)
    print(f"  worst asym   relative difference : {worst_asym:.3e}   (gate 1e-2)")
    print(f"  worst negeig relative difference : {worst_negeig:.3e}   (gate 1e-2)")
    print(f"  T0.2 VERDICT: {'PASS' if passed else '*** FAIL -- HALT CONDITION H1 ***'}")
    print("=" * 78)

    out = {"_config": {"model": MODEL, "t": T, "Q_seed": Q_SEED, "seeds": SEEDS,
                       "n": 100, "d": 5, "sigma": 1.0, "dither": False,
                       "gate": "rel diff in asym and negeig <= 1%"},
           "rows": rows,
           "worst_asym_rel_diff": worst_asym,
           "worst_negeig_rel_diff": worst_negeig,
           "T0_2_PASS": bool(passed),
           "halt_H1": bool(not passed)}
    p = HERE.parent / "results" / "t0_2_determinism.json"
    json.dump(out, open(p, "w"), indent=2)
    np.savez_compressed(HERE.parent / "results" / "t0_2_predictions.npz", **store)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
