"""EXPERIMENT 2.3, item 2.3.7 -- actual leave-one-out on the CORRUPTED context.

For each context point i the model is refitted without row i and the change in
fit is measured two ways:

    loo_test_shift_i = || m_test^(-i) - m_test ||_2      change in held-out fit
    loo_self_i       = | y_i - m_i^(-i) |                the point's own
                                                         out-of-fold residual

`loo_self` is the classical mislabel detector; `loo_test_shift` is the influence
form used in 2.2.5. Both are ranked against the known corrupted set.

REDUCED GRID, stated. Each combination costs 100 refits plus one base
prediction. The full 2.3 grid (180 combinations) would be 18,100 calls per
model on top of the 36,180 the Jacobians already cost. This runs the 10% rate
only, keeping all 3 schemes, all 4 datasets and all 5 seeds: 60 combinations,
6,060 calls per model. The 5% and 20% rates are NOT MEASURED for this item.

Writes results/exp23_loo_<model>.json and arrays/exp23_loo_<model>.npz.
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
from experiments.phase_2.src.paths import RESULTS, ARRAYS               # noqa: E402
from experiments.phase_2.src.estimators import jsonable                 # noqa: E402
from experiments.phase_2.src.chunk2_estimators import make_predict      # noqa: E402
from experiments.phase_2.src.exp23_corrupt import (                     # noqa: E402
    SEEDS, SCHEMES, DIDS, corrupt_indices, apply_corruption,
)

MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]
RATE = 0.10


def main():
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or MODELS
    data = np.load(ARRAYS / "chunk2_data.npz")
    manifest = {r["did"]: r for r in
                json.load(open(RESULTS / "chunk2_datasets.json"))["accepted"]}

    for mid in which:
        print("\n" + "=" * 78)
        print(f"MODEL {mid}   2.3.7 actual LOO, rate {RATE:.2f} only")
        print("=" * 78, flush=True)
        model = load(mid, task="regression", device="cuda")
        predict = make_predict(model, mid)
        rows, arrays = [], {}
        for did in DIDS:
            for s in SEEDS:
                X = data[f"{did}__{s}__X_ctx"]
                y0 = data[f"{did}__{s}__y_ctx"].astype(float)
                Xt = data[f"{did}__{s}__X_test"]
                n = len(y0)
                idx, k = corrupt_indices(did, s, RATE, n)
                for scheme in SCHEMES:
                    t0 = time.time()
                    yc, changed = apply_corruption(y0, idx, scheme, did, s, RATE)
                    m_test = np.asarray(predict(X, yc, Xt)).ravel()
                    shift = np.empty(n); self_r = np.empty(n)
                    for i in range(n):
                        keep = np.ones(n, bool); keep[i] = False
                        mt = np.asarray(predict(X[keep], yc[keep], Xt)).ravel()
                        mi = np.asarray(predict(X[keep], yc[keep], X[i:i + 1])).ravel()[0]
                        shift[i] = float(np.linalg.norm(mt - m_test))
                        self_r[i] = float(abs(yc[i] - mi))
                    tag = f"{mid}__{did}__{s}__{scheme}__{int(RATE*100)}"
                    arrays[f"{tag}__loo_test_shift"] = shift
                    arrays[f"{tag}__loo_self"] = self_r
                    arrays[f"{tag}__idx"] = idx
                    rows.append({"model": mid, "did": did, "name": manifest[did]["name"],
                                 "seed": s, "scheme": scheme, "rate": RATE,
                                 "n_ctx": n, "n_corrupt": int(k), "n_changed": int(changed),
                                 "corrupt_idx": idx.tolist(),
                                 "seconds": round(time.time() - t0, 1)})
                    print(f"  did={did:<5} s{s:<5} {scheme:<8} {rows[-1]['seconds']:6.1f}s",
                          flush=True)
                    np.savez_compressed(ARRAYS / f"exp23_loo_{mid}.npz", **arrays)
                    with open(RESULTS / f"exp23_loo_{mid}.json", "w") as f:
                        json.dump(jsonable({"_config": {
                            "rate": RATE, "rates_not_measured": [0.05, 0.20],
                            "dids": DIDS, "seeds": SEEDS, "schemes": SCHEMES,
                            "note": "2 predictions per refit: test block and the held-out row"},
                            "rows": rows}), f, indent=2)
        del model
        torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
