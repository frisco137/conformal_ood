"""EXPERIMENT 2.3 -- does the Jacobian find corrupted labels?

For each split a known subset of context labels is corrupted, the context
Jacobian is re-measured on the corrupted context, and per-point scores are
ranked against the known corrupted set.

CORRUPTION (2.3.1). Indices are drawn once per (did, seed, rate) and reused for
all three schemes and all three models, so every comparison is on the same
ground truth.
    flip     y_i <- 2*mean(y_clean) - y_i
    shuffle  labels of the selected subset permuted among themselves
    noise    y_i <- y_i + eps,  eps ~ N(0, (3*std(y_clean))^2)
`mean` and `std` are of the CLEAN context. For shuffle the number of indices
whose label actually changed is recorded, since a random permutation can fix a
point.

PROBE STEP. h = h_frac * std(y_corrupted): the models renormalise by the
statistics of the context they are actually given, so the step follows the
corrupted context. h_frac is unchanged from chunk 2.

SCORES (2.3.3), per context point:
    colnorm   ||J[:,i]||      how much label i moves all predictions
    absJii    |J_ii|
    rownorm   ||J[i,:]||      how much all labels move prediction i
    asym_rc   ||J[i,:]|| - ||J[:,i]||   zero for every point of a symmetric J;
                                        the A1-specific score
    w         negative-eigendirection loading from 2.2

Baselines are computed in exp23_analyse.py from the arrays saved here.

Writes results/exp23_corrupt_<model>.json and arrays/exp23_corrupt_<model>.npz.
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
sys.path.insert(0, str(SRC.parent.parent.parent))
warnings.filterwarnings("ignore")

import torch                                                            # noqa: E402
from models import load                                                 # noqa: E402
from experiments.core.metrics import get_Q                              # noqa: E402
from experiments.phase_2.src.paths import RESULTS, ARRAYS               # noqa: E402
from experiments.phase_2.src.estimators import context_jacobian, jsonable  # noqa: E402
from experiments.phase_2.src.chunk2_estimators import (                 # noqa: E402
    H_FRAC_BY_MODEL, make_predict,
)

SEEDS = [42, 100, 200, 300, 400]
MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]
Q_SEED = 0
SCHEMES = ["flip", "shuffle", "noise"]
RATES = [0.05, 0.10, 0.20]
NOISE_SD_MULT = 3.0

#: 4 of the 12 chunk-2 datasets. Each combination costs 201 model calls (one
#: base prediction plus a 2n central-difference context Jacobian), so the full
#: 12 x 5 x 3 x 3 grid is 108,540 calls per model -- about 10.6 h for TabICL
#: alone at its measured 0.35 s/call, and more under GPU contention. The grid
#: run here is 4 datasets x 5 seeds x 3 schemes x 3 rates = 180 combinations,
#: 36,180 calls per model.
#:
#: ALL THREE SCHEMES AND ALL THREE RATES ARE KEPT, because the brief enumerates
#: them; the reduction is taken entirely on the dataset axis, which was already
#: a free choice. The four retained span the full set on both axes that matter
#: here: feature dimension 7 to 25, and target SD 0.886 to 1083. The eight
#: dropped are named so the reduction is not silent, and 20 splits per
#: (scheme, rate) cell remain.
DIDS = [229, 509, 522, 581]
DROPPED = [223, 540, 547, 549, 560, 579, 582, 583]


def corrupt_indices(did, seed, rate, n):
    k = int(round(rate * n))
    rng = np.random.RandomState(1000 * seed + int(rate * 100) + did)
    return np.sort(rng.choice(n, k, replace=False)), k


def apply_corruption(y, idx, scheme, did, seed, rate):
    y = np.asarray(y, float).copy()
    rng = np.random.RandomState(2000 * seed + int(rate * 100) + did)
    mu, sd = float(np.mean(y)), float(np.std(y))
    changed = len(idx)
    if scheme == "flip":
        y[idx] = 2.0 * mu - y[idx]
    elif scheme == "shuffle":
        perm = rng.permutation(len(idx))
        newv = y[idx][perm]
        changed = int(np.sum(newv != y[idx]))
        y[idx] = newv
    elif scheme == "noise":
        y[idx] = y[idx] + rng.normal(0.0, NOISE_SD_MULT * sd, len(idx))
    else:
        raise ValueError(scheme)
    return y, changed


def scores_from_J(J, y):
    Q = get_Q(y, seed=Q_SEED)
    Jr = Q.T @ J @ Q
    S = (Jr + Jr.T) / 2.0
    ev, V = np.linalg.eigh(S)
    u = Q @ V[:, 0]; u = u / np.linalg.norm(u)
    w = u ** 2; w = w / w.sum()
    col = np.linalg.norm(J, axis=0)
    row = np.linalg.norm(J, axis=1)
    return {"colnorm": col, "absJii": np.abs(np.diag(J)), "rownorm": row,
            "asym_rc": row - col, "w": w}, float(ev[0])


def main():
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or MODELS
    data = np.load(ARRAYS / "chunk2_data.npz")
    manifest = {r["did"]: r for r in json.load(open(RESULTS / "chunk2_datasets.json"))["accepted"]}

    for mid in which:
        print("\n" + "=" * 78)
        print(f"MODEL {mid}   {len(DIDS)} datasets x {len(SEEDS)} seeds x "
              f"{len(SCHEMES)} schemes x {len(RATES)} rates")
        print("=" * 78, flush=True)
        model = load(mid, task="regression", device="cuda")
        predict = make_predict(model, mid)
        h_frac = H_FRAC_BY_MODEL[mid]
        rows, arrays = [], {}

        for did in DIDS:
            for s in SEEDS:
                X = data[f"{did}__{s}__X_ctx"]
                y0 = data[f"{did}__{s}__y_ctx"].astype(float)
                n = len(y0)
                for rate in RATES:
                    idx, k = corrupt_indices(did, s, rate, n)
                    for scheme in SCHEMES:
                        t0 = time.time()
                        yc, changed = apply_corruption(y0, idx, scheme, did, s, rate)
                        h = h_frac * float(np.std(yc))
                        m_ctx = np.asarray(predict(X, yc, X)).ravel()
                        J = context_jacobian(predict, X, yc, h)
                        sc, lam = scores_from_J(J, yc)

                        tag = f"{mid}__{did}__{s}__{scheme}__{int(rate*100)}"
                        arrays[f"{tag}__J"] = J.astype(np.float32)
                        arrays[f"{tag}__y_corr"] = yc
                        arrays[f"{tag}__m_ctx"] = m_ctx
                        arrays[f"{tag}__idx"] = idx
                        rows.append({
                            "model": mid, "did": did, "name": manifest[did]["name"],
                            "seed": s, "scheme": scheme, "rate": rate,
                            "n_ctx": n, "n_corrupt": int(k), "n_changed": int(changed),
                            "corrupt_idx": idx.tolist(),
                            "h": h, "h_frac": h_frac, "std_y_corr": float(np.std(yc)),
                            "std_y_clean": float(np.std(y0)),
                            "lam_min": lam, "trJ": float(np.trace(J)),
                            "n_neg_Jii": int(np.sum(np.diag(J) < 0)),
                            "asym_rc_absmax": float(np.max(np.abs(sc["asym_rc"]))),
                            "seconds": round(time.time() - t0, 1),
                        })
                        r = rows[-1]
                        print(f"  did={did:<5} s{s:<5} {scheme:<8} r={rate:.2f} "
                              f"{r['seconds']:6.1f}s  k={k:<3} changed={changed:<3} "
                              f"lam_min {lam:+.4f}  trJ {r['trJ']:7.2f}", flush=True)
                        np.savez_compressed(ARRAYS / f"exp23_corrupt_{mid}.npz", **arrays)
                        with open(RESULTS / f"exp23_corrupt_{mid}.json", "w") as f:
                            json.dump(jsonable({"_config": {
                                "dids": DIDS, "dids_dropped": DROPPED, "seeds": SEEDS,
                                "schemes": SCHEMES, "rates": RATES,
                                "noise_sd_mult": NOISE_SD_MULT, "Q_seed": Q_SEED,
                                "h_rule": "h = h_frac * std(y_corrupted)",
                                "h_frac": h_frac}, "rows": rows}), f, indent=2)
        del model
        torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
