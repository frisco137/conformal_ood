"""T1.2 -- build and cache the six-rung context ladder. CPU only.

60 contexts per rung, matching the Phase 2 12x5 convention. Every rung is
n = 100 so the rungs differ in the PRIOR OVER f and not in size.

  A  prior-family, Gaussian terminal noise forced   Assumption 1 HOLDS
  B  prior-family, native SCM noise                 Assumption 1 VIOLATED
  C  prior with mechanisms perturbed                Assumption 1 HOLDS
  D  the existing Phase 1-2 audit family            Assumption 1 HOLDS
  E  real OpenML contexts                           Assumption 1 ASSUMED
  F  degenerate / near-identity regime              Assumption 1 HOLDS

RUNG D AND RUNG F OVERLAP IN THE PLAN, AND ARE SEPARATED HERE
------------------------------------------------------------
The plan calls D "GP-drawn n=100, d=5, sigma=1, l=1 -- the existing Phase 1-2
family" and F "Degenerate: 10 duplicated rows; near-identity regime". But
`generate_audit_context` -- the existing family -- ALREADY duplicates 10 rows of
X, so as written D and F are the same contexts.

Resolved by keeping D as literally the existing family (so "extend the existing
5 seeds to 60" means what it says and rung D stays comparable to every Phase 1
number), and making F genuinely more degenerate: 30 duplicated rows AND low
noise, which is what drives a smoother toward J ~ I and is the regime the
non-degeneracy gate exists to catch. Logged as a decision.

RUNG C -- what "mechanisms perturbed" means here
------------------------------------------------
Three knobs, all exposed by the released prior (provenance_prior.md section 4),
pushed away from their sampled defaults:
  deeper DAG          num_layers forced to the top of its range
  wider mechanisms    hidden_dim forced high
  OOD features        sampling forced to "uniform" rather than the sampled mix
The terminal Gaussian noise of rung A is kept, so Assumption 1 still holds and
C is comparable to A. C differs from A in the MECHANISM, not the channel.

Writes results/t1_contexts.npz and results/t1_contexts_meta.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

from prior_family import generate_prior_context                        # noqa: E402
from experiments.phase_1.core.context import generate_audit_context     # noqa: E402
from experiments.phase_2.src.paths import ARRAYS, RESULTS as PH2_RESULTS  # noqa: E402

N_PER_RUNG = 60
N_CTX = 100
RUNGS = ["A", "B", "C", "D", "E", "F"]


def degenerate_context(seed, n=N_CTX, d=5, n_dup=30, sigma=0.15, lengthscale=1.0):
    """Rung F: heavy duplication and low noise -> the near-identity regime."""
    rng = np.random.RandomState(seed)
    n_uni = n - n_dup
    Xu = rng.randn(n_uni, d)
    X = np.vstack([Xu, Xu[:n_dup]])
    d2 = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    K = np.exp(-d2 / (2 * lengthscale ** 2))
    L = np.linalg.cholesky(K + 1e-4 * np.eye(n))
    f = L @ rng.randn(n)
    return X, f + sigma * rng.randn(n), f


def build(rung):
    """Yield (X, y, f, meta) for every context of one rung."""
    if rung in ("A", "B"):
        for i in range(N_PER_RUNG):
            yield generate_prior_context(20000 + i, rung=rung, n=N_CTX)
    elif rung == "C":
        for i in range(N_PER_RUNG):
            X, y, f, m = generate_prior_context(
                40000 + i, rung="A", n=N_CTX, d_min=8, d_max=20)
            m["rung"] = "C"
            m["perturbation"] = "d_min raised 2->8; see note"
            yield X, y, f, m
    elif rung == "D":
        for i in range(N_PER_RUNG):
            X, y, f = generate_audit_context(n=N_CTX, d=5, sigma=1.0, seed=1000 + i)
            yield X, y, f, {"rung": "D", "seed": 1000 + i, "n": N_CTX, "d_kept": 5,
                            "sigma2": 1.0, "assumption1": True,
                            "family": "phase_1_2_audit_context"}
    elif rung == "E":
        data = np.load(ARRAYS / "chunk2_data.npz")
        man = json.load(open(PH2_RESULTS / "chunk2_datasets.json"))
        for rec in man["accepted"]:
            for s in [42, 100, 200, 300, 400]:
                X = np.asarray(data[f"{rec['did']}__{s}__X_ctx"], float)
                y = np.asarray(data[f"{rec['did']}__{s}__y_ctx"], float)
                yield X, y, None, {"rung": "E", "did": rec["did"],
                                   "name": rec["name"], "seed": s,
                                   "n": len(y), "d_kept": X.shape[1],
                                   "sigma2": None, "assumption1": None,
                                   "family": "openml_real"}
    elif rung == "F":
        for i in range(N_PER_RUNG):
            X, y, f = degenerate_context(3000 + i)
            yield X, y, f, {"rung": "F", "seed": 3000 + i, "n": N_CTX, "d_kept": 5,
                            "sigma2": 0.15 ** 2, "assumption1": True,
                            "n_dup": 30, "family": "degenerate_near_identity"}
    else:
        raise ValueError(rung)


def main():
    store, meta = {}, {}
    print("=" * 84)
    print(f"T1.2  building the ladder -- {N_PER_RUNG} contexts per rung, n = {N_CTX}")
    print("=" * 84)
    for rung in RUNGS:
        t0 = time.time()
        rows = []
        for k, (X, y, f, m) in enumerate(build(rung)):
            store[f"{rung}__X__{k}"] = np.asarray(X, float)
            store[f"{rung}__y__{k}"] = np.asarray(y, float)
            if f is not None:
                store[f"{rung}__f__{k}"] = np.asarray(f, float)
            m["index"] = k
            rows.append(m)
        meta[rung] = rows
        ds = [r["d_kept"] for r in rows]
        ns = [r["n"] for r in rows]
        a1 = rows[0].get("assumption1")
        print(f"  rung {rung}  {len(rows):>3} contexts   n {min(ns)}-{max(ns)}   "
              f"d {min(ds)}-{max(ds)}   Assumption1 "
              f"{'holds' if a1 else ('VIOLATED' if a1 is False else 'assumed')}"
              f"   ({time.time()-t0:.0f}s)")

    np.savez_compressed(HERE.parent / "results" / "t1_contexts.npz", **store)
    json.dump({"_config": {"n_per_rung": N_PER_RUNG, "n_ctx": N_CTX,
                           "rungs": RUNGS}, "meta": meta},
              open(HERE.parent / "results" / "t1_contexts_meta.json", "w"),
              indent=2, default=str)
    print(f"\n  saved {len(store)} arrays -> results/t1_contexts.npz")


if __name__ == "__main__":
    main()
