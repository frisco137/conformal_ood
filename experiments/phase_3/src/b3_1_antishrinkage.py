"""Block 3.1 -- anti-shrinkage, de-confounded. Registered as P14/P15/P16.

THE CONFOUND THIS REMOVES
-------------------------
Round 1 observed that TabICL interpolates on noisy prior-family contexts
(trJ/n 0.84-0.94) and does not on clean low-noise ones (rung F, 0.674). But those
rungs also differ in feature dimension and DAG depth, so the noise reading was
confounded.

Here ONE context is fixed and ONLY sigma moves:

    X, f  fixed          eps ~ N(0, I) drawn once and FIXED across the sweep
    y(s) = f + s * eps   for six s log-spaced over two decades

so every difference across the sweep is attributable to the noise scale alone.
Holding eps fixed rather than redrawing it also removes sampling noise from the
comparison: the six y vectors lie on a ray, not in a cloud.

WHY THE CONTROLS ARE NOT OPTIONAL HERE
--------------------------------------
For an exact GP, J = K(K + s^2 I)^-1 has eigenvalues lam_i / (lam_i + s^2), which
are strictly decreasing in s^2. So trJ/n MUST fall as noise rises -- P14 is a
theorem, and if the harness does not reproduce it the harness is wrong and
nothing else here is interpretable. The controls are the harness check.

sigma is applied in the ORIGINAL target units, and the models renormalise
internally, so what actually varies is the signal-to-noise ratio of the context.
Var(f) is recorded per base context so SNR = Var(f)/s^2 can be quoted.

Usage:  python b3_1_antishrinkage.py [n_base]
Writes results/b3_1_antishrinkage.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats


class _MockAnalytics:
    def __getattr__(self, name):
        return lambda *a, **k: None


sys.modules["analytics"] = _MockAnalytics()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

import torch                                                            # noqa: E402
from models import load                                                 # noqa: E402
from controls import NoiseHyperpriorGP                                  # noqa: E402
from experiments.phase_1.core.context import generate_audit_context      # noqa: E402
from experiments.phase_1.core.controls import GP_SIGMA, wrap             # noqa: E402
from experiments.phase_1.core.metrics import (                          # noqa: E402
    asym, central_jacobian, get_Q, negeig, reduced_jacobian,
)
from experiments.phase_1.core.surrogates import ExactGP, HierarchicalGP  # noqa: E402

SIGMAS = np.geomspace(0.03, 3.0, 6)        # two decades
N_BASE = 4                                  # base contexts (X, f, eps) held fixed
T = 1e-3
Q_SEED = 0


def block(J, y, n):
    Q = get_Q(y, seed=Q_SEED)
    Jr = Q.T @ J @ Q
    return {"asym": float(asym(Jr)), "negeig": float(negeig(Jr)),
            "trJ_over_n": float(np.trace(J) / n),
            "J_minus_I": float(np.linalg.norm(J - np.eye(n), "fro")
                               / np.linalg.norm(J, "fro")),
            "normF": float(np.linalg.norm(J, "fro"))}


def main():
    n_base = int(sys.argv[1]) if len(sys.argv) > 1 else N_BASE
    model = load("tabicl_v2", task="regression", device="cuda")
    out = {"_config": {"sigmas": SIGMAS.tolist(), "n_base": n_base, "t": T,
                       "eps_fixed_across_sweep": True, "n": 100, "d": 5,
                       "registered": ["P14", "P15", "P16"]},
           "rows": []}

    print("=" * 100)
    print("B3.1  anti-shrinkage, de-confounded -- ONE context, only sigma moves")
    print(f"      sigma in {[f'{s:.3f}' for s in SIGMAS]}   (two decades), eps FIXED")
    print("=" * 100)

    for b in range(n_base):
        X, _, f = generate_audit_context(n=100, d=5, sigma=1.0, seed=500 + b)
        n = len(f)
        eps = np.random.RandomState(9000 + b).randn(n)      # FIXED across the sweep
        varf = float(np.var(f))
        print(f"\n  base {b}  Var(f) = {varf:.4f}")
        print(f"    {'sigma':>8}{'SNR':>8}   "
              f"{'TabICL trJ/n':>13}{'asym':>9}{'negeig':>9}   "
              f"{'GP trJ/n':>10}{'hierGP':>9}{'nhGP':>9}")
        for s in SIGMAS:
            y = f + s * eps
            t0 = time.time()

            def predict(y_eval):
                model.estimator.fit(X, y_eval)
                return model.estimator.predict(X)

            Jm = np.empty((n, n))
            for i in range(n):
                e = np.zeros(n); e[i] = T
                Jm[:, i] = (np.asarray(predict(y + e)).ravel()
                            - np.asarray(predict(y - e)).ravel()) / (2 * T)
            m = block(Jm, y, n)

            # controls, closed form, through the same wrapper the model wears
            ctl = {}
            for nm, obj in [("exactgp", ExactGP(X, sigma=float(s))),
                            ("hiergp", HierarchicalGP(X, sigma=float(s))),
                            ("nhgp", NoiseHyperpriorGP(X))]:
                Jc = central_jacobian(wrap(obj.predict), y, h=T)
                ctl[nm] = block(Jc, y, n)

            row = {"base": b, "sigma": float(s), "snr": varf / float(s) ** 2,
                   "var_f": varf, "model": m, "controls": ctl,
                   "seconds": time.time() - t0}
            out["rows"].append(row)
            print(f"    {s:>8.3f}{row['snr']:>8.2f}   "
                  f"{m['trJ_over_n']:>13.4f}{m['asym']:>9.4f}{m['negeig']:>9.4f}   "
                  f"{ctl['exactgp']['trJ_over_n']:>10.4f}"
                  f"{ctl['hiergp']['trJ_over_n']:>9.4f}"
                  f"{ctl['nhgp']['trJ_over_n']:>9.4f}", flush=True)

    # ------------------------------------------------------------ P14, P15, P16
    print("\n" + "=" * 100)
    print("REGISTERED PREDICTIONS")
    print("=" * 100)
    ss = [r["sigma"] for r in out["rows"]]
    res = {}

    def sp(vals):
        return float(stats.spearmanr(ss, vals).statistic)

    for nm in ("exactgp", "hiergp", "nhgp"):
        res[nm] = {"trJ_over_n": sp([r["controls"][nm]["trJ_over_n"] for r in out["rows"]]),
                   "asym": sp([r["controls"][nm]["asym"] for r in out["rows"]])}
    res["tabicl"] = {k: sp([r["model"][k] for r in out["rows"]])
                     for k in ("trJ_over_n", "asym", "negeig", "J_minus_I")}

    p14 = res["exactgp"]["trJ_over_n"] <= -0.9 and res["hiergp"]["trJ_over_n"] <= -0.9
    tj = res["tabicl"]["trJ_over_n"]
    p15 = tj >= 0.5
    p15_flat = abs(tj) < 0.3
    p16 = res["tabicl"]["asym"] > 0 and res["tabicl"]["negeig"] > 0

    print(f"\n  Spearman(sigma, trJ/n)   exact GP {res['exactgp']['trJ_over_n']:+.4f}"
          f"   hier GP {res['hiergp']['trJ_over_n']:+.4f}"
          f"   noise-hyper GP {res['nhgp']['trJ_over_n']:+.4f}")
    print(f"                            TabICL   {tj:+.4f}")
    print(f"\n  P14 controls trJ/n decreasing (<= -0.9) : "
          f"{'PASS' if p14 else '*** FAIL -- HARNESS SUSPECT ***'}")
    if p15:
        v = "PASS -- TabICL ANTI-SHRINKS"
    elif p15_flat:
        v = "FLAT -- neither shrinks nor anti-shrinks; its own result"
    else:
        v = "*** FAILS -- TabICL shrinks like the controls; the round-1 finding is WITHDRAWN ***"
    print(f"  P15 TabICL trJ/n increasing (>= +0.5): {tj:+.4f}   {v}")
    print(f"  P16 TabICL asym {res['tabicl']['asym']:+.4f}  "
          f"negeig {res['tabicl']['negeig']:+.4f}  "
          f"-> {'PASS' if p16 else 'MISS (violation flat or falling in sigma)'}")

    out["spearman"] = res
    out["P14"] = {"PASS": bool(p14)}
    out["P15"] = {"spearman": tj, "PASS": bool(p15), "FLAT": bool(p15_flat),
                  "withdrawn": bool(not p15 and not p15_flat)}
    out["P16"] = {"PASS": bool(p16)}

    del model
    torch.cuda.empty_cache()
    p = HERE.parent / "results" / "b3_1_antishrinkage.json"
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
