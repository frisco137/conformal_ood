"""EXPERIMENT 2.2, model-call items -- leave-one-out influence (2.2.5) and
label-perturbation sensitivity (2.2.7).

Everything else in 2.2 is a function of Jacobians already on disk and runs in
exp22_analyse.py with no GPU.

2.2.5  LEAVE-ONE-OUT INFLUENCE.  For every context point i, refit the model on
the context with row i removed and predict the same held-out rows:

    influence_i = || m_test^(-i) - m_test ||_2

No subsample is taken. The brief allows a subsample and asks for its size; the
full 100 points per split were affordable (100 refits x 60 splits x 3 models =
18000 calls, plus the base prediction already on disk), so the subsampling
caveat does not apply and the rank correlation in 2.2.5 is over all 100 context
points of every split.

2.2.7  LABEL-PERTURBATION SENSITIVITY.  i_high = argmax w. i_low is drawn from
the points with w below its median and chosen to MATCH i_high on |J_ii|, so the
comparison isolates the eigendirection rather than the diagonal sensitivity. The
achieved match |,|J_ii| - |J_i'i'|,| is recorded per split and reported; a split
whose best match is poor is visible rather than averaged away.

Perturbation is delta = 1.0 * std(y_ctx), applied in both signs, and the
reported sensitivity is the mean of the two magnitudes:

    sens_i = ( ||m_test(y + delta e_i) - m_test|| + ||m_test(y - delta e_i) - m_test|| ) / 2

Writes results/exp22_loo_<model>.json and arrays/exp22_loo_<model>.npz.
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

import torch                                                        # noqa: E402
from models import load                                             # noqa: E402
from experiments.phase_1.core.metrics import get_Q                          # noqa: E402
from experiments.phase_2.src.paths import RESULTS, ARRAYS           # noqa: E402
from experiments.phase_2.src.estimators import jsonable             # noqa: E402
from experiments.phase_2.src.chunk2_estimators import make_predict  # noqa: E402

SEEDS = [42, 100, 200, 300, 400]
MODELS = ["tabicl_v2", "tabpfn_v2", "tabswift"]
Q_SEED = 0
PERTURB_SD_MULT = 1.0


def loading(J, y):
    """w_i = u_i^2 normalised, u = Q v, v the unit eigenvector of lam_min(sym Q^T J Q)."""
    Q = get_Q(y, seed=Q_SEED)
    Jr = Q.T @ J @ Q
    S = (Jr + Jr.T) / 2.0
    ev, V = np.linalg.eigh(S)
    u = Q @ V[:, 0]
    u = u / np.linalg.norm(u)
    w = u ** 2
    return w / w.sum(), float(ev[0]), u


def main():
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or MODELS
    data = np.load(ARRAYS / "chunk2_data.npz")
    manifest = json.load(open(RESULTS / "chunk2_datasets.json"))

    for mid in which:
        Z = np.load(ARRAYS / f"chunk2_arrays_{mid}.npz")
        print("\n" + "=" * 78)
        print(f"MODEL {mid}   full LOO over all 100 context points per split")
        print("=" * 78, flush=True)
        model = load(mid, task="regression", device="cuda")
        predict = make_predict(model, mid)

        rows, arrays = [], {}
        for rec in manifest["accepted"]:
            did = rec["did"]
            for s in SEEDS:
                t0 = time.time()
                tag = f"{mid}__{did}__{s}"
                X = data[f"{did}__{s}__X_ctx"]
                y = data[f"{did}__{s}__y_ctx"].astype(float)
                Xt = data[f"{did}__{s}__X_test"]
                m_test = Z[f"{tag}__m_test"]
                J = Z[f"{tag}__J_ctx"].astype(np.float64)
                n = len(y)

                # ---- 2.2.5 full leave-one-out
                infl = np.empty(n)
                for i in range(n):
                    keep = np.ones(n, bool); keep[i] = False
                    m_loo = np.asarray(predict(X[keep], y[keep], Xt)).ravel()
                    infl[i] = float(np.linalg.norm(m_loo - m_test))

                # ---- 2.2.7 matched-pair label perturbation
                w, lam_min, _ = loading(J, y)
                d = np.abs(np.diag(J))
                i_hi = int(np.argmax(w))
                cand = np.where(w <= np.median(w))[0]
                i_lo = int(cand[np.argmin(np.abs(d[cand] - d[i_hi]))])
                delta = PERTURB_SD_MULT * float(np.std(y))

                def sens(idx):
                    out = []
                    for sgn in (+1.0, -1.0):
                        yp = y.copy(); yp[idx] += sgn * delta
                        mp = np.asarray(predict(X, yp, Xt)).ravel()
                        out.append(float(np.linalg.norm(mp - m_test)))
                    return float(np.mean(out)), out

                s_hi, s_hi_both = sens(i_hi)
                s_lo, s_lo_both = sens(i_lo)

                arrays[f"{tag}__loo_influence"] = infl
                rows.append({
                    "model": mid, "did": did, "name": rec["name"], "seed": s,
                    "n_ctx": n, "n_test": int(len(m_test)),
                    "lam_min": lam_min,
                    "loo_min": float(infl.min()), "loo_max": float(infl.max()),
                    "loo_mean": float(infl.mean()),
                    "perturb_delta": delta,
                    "i_high": i_hi, "i_low_matched": i_lo,
                    "w_high": float(w[i_hi]), "w_low": float(w[i_lo]),
                    "Jii_high": float(d[i_hi]), "Jii_low": float(d[i_lo]),
                    "Jii_match_abs_gap": float(abs(d[i_hi] - d[i_lo])),
                    "Jii_match_rel_gap": float(abs(d[i_hi] - d[i_lo]) / max(d[i_hi], 1e-12)),
                    "sens_high": s_hi, "sens_low": s_lo,
                    "sens_high_both": s_hi_both, "sens_low_both": s_lo_both,
                    "sens_ratio_high_over_low": float(s_hi / s_lo) if s_lo > 0 else None,
                    "seconds": round(time.time() - t0, 1),
                })
                r = rows[-1]
                print(f"  did={did:<6} seed {s:<5} {r['seconds']:6.1f}s  "
                      f"lam_min {lam_min:+.4f}  LOO [{infl.min():.4g}, {infl.max():.4g}]  "
                      f"i_hi {i_hi:3d} i_lo {i_lo:3d}  Jii gap {r['Jii_match_rel_gap']:.3f}  "
                      f"sens hi/lo {r['sens_ratio_high_over_low'] if r['sens_ratio_high_over_low'] else float('nan'):.3f}",
                      flush=True)

                np.savez_compressed(ARRAYS / f"exp22_loo_{mid}.npz", **arrays)
                with open(RESULTS / f"exp22_loo_{mid}.json", "w") as f:
                    json.dump(jsonable({"_config": {
                        "seeds": SEEDS, "Q_seed": Q_SEED,
                        "loo": "full, all 100 context points, no subsample",
                        "perturb_sd_mult": PERTURB_SD_MULT,
                        "i_low_rule": "w <= median(w), argmin |Jii - Jii(i_high)|"},
                        "rows": rows}), f, indent=2)
        del model
        torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
