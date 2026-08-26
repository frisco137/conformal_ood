"""CHUNK 2, part 0 -- step-size plateau on the FROZEN models and on real data.

Chunk 1.6 predicted the resolvable step per model by quantising a control at
that model's measured output quantum. This checks the prediction on the real
models, and fixes the probe step used by chunk 2.

The step is swept as a FRACTION of std(y_ctx). It has to be: every model in the
roster renormalises the target internally and un-normalises its output, so the
output quantum expressed in the original target units is proportional to
std(y). The accepted datasets span target SDs from 0.89 (`no2`) to 1083
(`places`), three orders of magnitude, so a single absolute step cannot sit in
the plateau for all of them.

Reported per model per dataset: tr J, n - tr J, the count of negative diagonal
entries, and mean J_** over 10 queries. A step is in the plateau where tr J is
flat in h.

Three datasets spanning the target-scale range are used:
  did=583  fri_c1_1000_50   target sd 0.9995
  did=223  stock            target sd 6.536
  did=509  places           target sd 1083

Writes chunk2_stepsize_<models>.json.
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

import torch                                                     # noqa: E402
from models import load                                          # noqa: E402
from experiments.phase_2.src.paths import RESULTS, ARRAYS        # noqa: E402
from experiments.phase_2.src.estimators import (                 # noqa: E402
    context_jacobian, jacobian_star_batch, jsonable,
)
from experiments.phase_2.src.chunk2_estimators import (          # noqa: E402
    H_FRAC_BY_MODEL, AUDIT_MEAN_SY, make_predict,
)

DIDS = [583, 223, 509]
SEED = 200
H_FRACS = [1e-4, 1e-3, 1e-2, 3e-2, 6.8598e-2, 2e-1, 5e-1]
N_QUERY = 10
MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]


def main():
    which = sys.argv[1:] or MODELS
    data = np.load(ARRAYS / "chunk2_data.npz")
    manifest = json.load(open(RESULTS / "chunk2_datasets.json"))
    names = {r["did"]: r["name"] for r in manifest["accepted"]}

    # one file per invocation: this script is run concurrently on two GPUs, and
    # a shared output file would have the second writer clobber the first.
    out_path = RESULTS / f"chunk2_stepsize_{'_'.join(which)}.json"
    res = {}
    res["_config"] = {"dids": DIDS, "seed": SEED, "h_fracs": H_FRACS,
                      "n_query": N_QUERY, "audit_mean_sy": AUDIT_MEAN_SY,
                      "h_frac_chosen": H_FRAC_BY_MODEL,
                      "rule": "h = h_frac * std(y_ctx)"}

    for mid in which:
        print("\n" + "=" * 78)
        print(f"{mid}   chosen h_frac = {H_FRAC_BY_MODEL[mid]:.4e}")
        print("=" * 78)
        model = load(mid, task="regression", device="cuda")
        predict = make_predict(model, mid)
        res.setdefault(mid, {})
        for did in DIDS:
            X = data[f"{did}__{SEED}__X_ctx"]
            y = data[f"{did}__{SEED}__y_ctx"]
            Xq = data[f"{did}__{SEED}__X_test"][:N_QUERY]
            sd = float(np.std(y))
            m_star = np.asarray(predict(X, y, Xq)).ravel()
            rows = []
            print(f"  did={did} {names[did]:<18} std(y)={sd:.5g}")
            for hf in H_FRACS:
                t0 = time.time()
                h = hf * sd
                J = context_jacobian(predict, X, y, h)
                Js, _ = jacobian_star_batch(predict, X, y, Xq, m_star, h)
                trJ = float(np.trace(J))
                rows.append({"h_frac": hf, "h_abs": h, "trJ": trJ,
                             "n_minus_trJ": len(y) - trJ,
                             "n_neg_Jii": int(np.sum(np.diag(J) < 0)),
                             "Jstar_mean": float(np.mean(Js)),
                             "Jstar_min": float(np.min(Js)),
                             "Jstar_max": float(np.max(Js)),
                             "seconds": round(time.time() - t0, 1)})
                r = rows[-1]
                print(f"    h_frac {hf:9.3e}  h {h:11.5g}  trJ {trJ:9.3f}  "
                      f"n-trJ {r['n_minus_trJ']:9.3f}  neg Jii {r['n_neg_Jii']:3d}  "
                      f"mean J_** {r['Jstar_mean']:+.5f}", flush=True)
            res[mid][str(did)] = {"name": names[did], "std_y": sd, "rows": rows}
            with open(out_path, "w") as f:
                json.dump(jsonable(res), f, indent=2)
        del model
        torch.cuda.empty_cache()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
