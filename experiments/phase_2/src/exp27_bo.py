"""EXPERIMENT 2.7 (C) -- the BO loop, with 2.6 riding on it at no extra model calls.

RUN BECAUSE 2.4 SHOWED THE DEFECT REACHES THE DECISION: native and Jacobian
agree on 0.160 of TabICL decisions and 0.550 / 0.517 of TabPFN's, and the
Jacobian channel's regret is worse than greedy on every model and both
acquisitions (p between 7.2e-07 and 4.1e-03).

SURROGATES (2.7.2), all sharing the initial design within a seed so trajectories
are paired:
    tabpfn_native      TabPFN v2 mean + its own bar-distribution variance
    tabpfn_conformal   TabPFN v2 mean + split-conformal width from a held-out
                       calibration slice of the observed set
    tabpfn_jacobian    TabPFN v2 mean + sigma_hat^2/(1 - J_**)
    gp                 marginal-likelihood GP, ConstantKernel*RBF + WhiteKernel,
                       2 restarts -- THE STRONG INCUMBENT, not crippled
    random             the floor

WHY ONLY TabPFN v2 CARRIES THE THREE VARIANCE ARMS. The Jacobian arm needs
J_** at every candidate, which is two forward passes per candidate per iteration:
2 x 128 x 20 x 5 x 5 = 128,000 calls. At TabPFN's measured 0.07 s that is about
2.5 h; at TabICL's 0.35 s it would be 12.4 h and at TabSwift's 0.11 s about 3.9 h,
on top of everything else. TabPFN v2 is also the checkpoint actually deployed as a
BO surrogate in the literature the brief cites. TabICL v2 and TabSwift are
NOT RUN in the loop and that is stated rather than approximated.

COST. Per iteration the native/conformal/gp/random arms cost one batched
prediction; the Jacobian arm costs 2 x n_pool. The audit quantities of 2.7.7 cost
a further 2n per iteration on the Jacobian arm only, where n is the current
context size.

Writes results/exp27_bo_<objectives>.json and arrays/exp27_bo_<objectives>.npz.
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
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC.parent.parent.parent))
warnings.filterwarnings("ignore")

import torch                                                        # noqa: E402
from models import load                                             # noqa: E402
from experiments.core.metrics import get_Q, asym, negeig            # noqa: E402
from experiments.phase_2.src.paths import RESULTS, ARRAYS           # noqa: E402
from experiments.phase_2.src.estimators import (                    # noqa: E402
    jsonable, conformal_halfwidth, context_jacobian, jacobian_star_batch, sigma2_hat,
)
from experiments.phase_2.src.chunk2_estimators import (             # noqa: E402
    H_FRAC_BY_MODEL, make_predict,
)
from experiments.phase_2.src.exp24_acquisition import ei, ucb, z_for  # noqa: E402

N_INIT = 10
T_ITER = 20
N_POOL = 128
SEEDS = [0, 1, 2, 3, 4]
DELTA = 0.1
CONF_LEVEL = 0.90
ARMS = ["tabpfn_native", "tabpfn_conformal", "tabpfn_jacobian", "gp", "random"]
MODEL_ID = "tabpfn_v2"


# ------------------------------------------------------------------ objectives
def branin(x):
    x1 = -5 + 15 * x[:, 0]; x2 = 15 * x[:, 1]
    a, b, c = 1.0, 5.1 / (4 * np.pi ** 2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return -(a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s)


_H3A = np.array([[3., 10., 30.], [.1, 10., 35.], [3., 10., 30.], [.1, 10., 35.]])
_H3P = np.array([[.3689, .1170, .2673], [.4699, .4387, .7470],
                 [.1091, .8732, .5547], [.03815, .5743, .8828]])
_H3C = np.array([1., 1.2, 3., 3.2])


def hartmann3(x):
    d = ((x[:, None, :] - _H3P[None]) ** 2 * _H3A[None]).sum(-1)
    return (np.exp(-d) * _H3C).sum(-1)


_H6A = np.array([[10., 3., 17., 3.5, 1.7, 8.], [.05, 10., 17., .1, 8., 14.],
                 [3., 3.5, 1.7, 10., 17., 8.], [17., 8., .05, 10., .1, 14.]])
_H6P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                        [2329, 4135, 8307, 3736, 1004, 9991],
                        [2348, 1451, 3522, 2883, 3047, 6650],
                        [4047, 8828, 8732, 5743, 1091, 381]])


def hartmann6(x):
    d = ((x[:, None, :] - _H6P[None]) ** 2 * _H6A[None]).sum(-1)
    return (np.exp(-d) * _H3C).sum(-1)


def ackley5(x):
    z = -32.768 + 65.536 * x
    n = z.shape[1]
    return -(-20 * np.exp(-0.2 * np.sqrt((z ** 2).sum(1) / n))
             - np.exp(np.cos(2 * np.pi * z).sum(1) / n) + 20 + np.e)


def rosenbrock5(x):
    z = -5 + 15 * x
    return -(100 * (z[:, 1:] - z[:, :-1] ** 2) ** 2 + (1 - z[:, :-1]) ** 2).sum(1)


#: name -> (fn on the unit cube, dimension, known maximum of the maximised form)
OBJECTIVES = {
    "branin2":     (branin, 2, -0.397887),
    "hartmann3":   (hartmann3, 3, 3.86278),
    "hartmann6":   (hartmann6, 6, 3.32237),
    "ackley5":     (ackley5, 5, 0.0),
    "rosenbrock5": (rosenbrock5, 5, 0.0),
}


def sobol(n, d, seed):
    return stats.qmc.Sobol(d=d, scramble=True, seed=seed).random(n)


def gp_fit_predict(Xo, yo, Xp):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    k = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e3))
         + WhiteKernel(1e-2, (1e-8, 1e2)))
    g = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2,
                                 random_state=0).fit(Xo, yo)
    out = g.predict(Xp, return_std=True)
    return np.asarray(out[0]).ravel(), np.asarray(out[1]).ravel() ** 2


def run_trajectory(arm, fn, dim, seed, predict, est, log):
    rng = np.random.default_rng(10_000 + seed)
    Xo = sobol(N_INIT, dim, seed)
    yo = fn(Xo)
    h_frac = H_FRAC_BY_MODEL[MODEL_ID]
    best = [float(yo.max())]
    audit, sixes = [], []
    for t in range(T_ITER):
        Xp = sobol(N_POOL, dim, seed * 1000 + t + 1)
        ybest = float(yo.max())
        if arm == "random":
            pick = int(rng.integers(N_POOL))
            mu_pick = float("nan")
        elif arm == "gp":
            mu, v = gp_fit_predict(Xo, yo, Xp)
            a, _ = ei(mu, np.sqrt(np.maximum(v, 0)), ybest)
            pick = int(np.argmax(a)); mu_pick = float(mu[pick])
        else:
            est.fit(Xo, yo)
            mu = np.asarray(est.predict(Xp, output_type="mean")).ravel()
            if arm == "tabpfn_native":
                full = est.predict(Xp, output_type="full")
                crit, lg = full["criterion"], full["logits"]
                lg = lg if torch.is_tensor(lg) else torch.as_tensor(lg)
                lg = lg.to(next(crit.buffers()).device).float()
                s2 = crit.variance(lg).detach().cpu().numpy().ravel()
            elif arm == "tabpfn_conformal":
                ncal = max(3, len(yo) // 3)
                idx = rng.permutation(len(yo))
                tr, ca = idx[ncal:], idx[:ncal]
                est.fit(Xo[tr], yo[tr])
                mcal = np.asarray(est.predict(Xo[ca], output_type="mean")).ravel()
                q = conformal_halfwidth(np.abs(yo[ca] - mcal), 1 - CONF_LEVEL)
                est.fit(Xo, yo)
                q = float(q) if np.isfinite(q) else float(np.std(yo) + 1e-9)
                s2 = np.full(N_POOL, (q / z_for(CONF_LEVEL)) ** 2)
            else:                                   # tabpfn_jacobian
                h = h_frac * float(np.std(yo) + 1e-12)
                J = context_jacobian(predict, Xo, yo, h)
                trJ = float(np.trace(J))
                mctx = np.asarray(predict(Xo, yo, Xo)).ravel()
                s2h, dof, defined = sigma2_hat(mctx, yo, trJ)
                Js, _ = jacobian_star_batch(predict, Xo, yo, Xp, mu, h)
                if defined:
                    den = 1.0 - Js
                    s2 = np.where(den > 0, s2h / np.where(den > 0, den, 1.0),
                                  1e-3 * abs(s2h))
                    s2 = np.maximum(s2, 1e-3 * abs(s2h))
                else:
                    s2 = np.full(N_POOL, float(np.var(yo) + 1e-12))
                n = len(yo)
                Qm = get_Q(yo, seed=0) if n > 3 else None
                Jr = Qm.T @ J @ Qm if Qm is not None else J
                audit.append({"t": t, "n": n, "trJ": trJ, "n_minus_trJ": dof,
                              "sigma2_hat_defined": bool(defined),
                              "sigma2_hat": s2h if defined else None,
                              "n_neg_Jii": int(np.sum(np.diag(J) < 0)),
                              "asym": float(asym(Jr)), "negeig": float(negeig(Jr)),
                              "Jstar_undefined_frac": float(np.mean(~np.isfinite(s2)))})
            a, _ = ei(mu, np.sqrt(np.maximum(s2, 0)), ybest)
            pick = int(np.argmax(a)); mu_pick = float(mu[pick])
        xn = Xp[pick:pick + 1]; yn = float(fn(xn)[0])

        # ---- 2.6: the sign of the update at the acquired point
        if arm.startswith("tabpfn"):
            est.fit(Xo, yo)
            mu_before = float(np.asarray(est.predict(xn, output_type="mean")).ravel()[0])
            Xo2 = np.vstack([Xo, xn]); yo2 = np.append(yo, yn)
            est.fit(Xo2, yo2)
            mu_after = float(np.asarray(est.predict(xn, output_type="mean")).ravel()[0])
            jstar = None
            if arm == "tabpfn_jacobian":
                jstar = float(Js[pick])
            surprise = yn - mu_before
            move = mu_after - mu_before
            sixes.append({"t": t, "J_star": jstar, "surprise": surprise, "move": move,
                          "sign_violation": bool(np.sign(move) != np.sign(surprise)
                                                 and surprise != 0),
                          "shrinkage": (move / surprise) if surprise != 0 else None})
            Xo, yo = Xo2, yo2
        elif arm == "gp":
            mu_before = float(gp_fit_predict(Xo, yo, xn)[0][0])
            Xo = np.vstack([Xo, xn]); yo = np.append(yo, yn)
            mu_after = float(gp_fit_predict(Xo, yo, xn)[0][0])
            surprise = yn - mu_before; move = mu_after - mu_before
            sixes.append({"t": t, "J_star": None, "surprise": surprise, "move": move,
                          "sign_violation": bool(np.sign(move) != np.sign(surprise)
                                                 and surprise != 0),
                          "shrinkage": (move / surprise) if surprise != 0 else None})
        else:
            Xo = np.vstack([Xo, xn]); yo = np.append(yo, yn)
        best.append(float(yo.max()))
    return best, audit, sixes


def main():
    objs = [a for a in sys.argv[1:] if not a.startswith("--")] or list(OBJECTIVES)
    # one file per invocation: this is run concurrently on two GPUs over disjoint
    # objective sets, and a shared output file would have one writer clobber the other
    stem = "exp27_bo_" + "_".join(objs)
    model = load(MODEL_ID, task="regression", device="cuda")
    est = model.estimator
    predict = make_predict(model, MODEL_ID)
    rows, arrays = [], {}
    for oname in objs:
        fn, dim, fmax = OBJECTIVES[oname]
        for seed in SEEDS:
            for arm in ARMS:
                t0 = time.time()
                best, audit, sixes = run_trajectory(arm, fn, dim, seed, predict, est, None)
                reg = [fmax - b for b in best]
                rows.append({"objective": oname, "dim": dim, "f_max": fmax,
                             "seed": seed, "arm": arm,
                             "best": best, "simple_regret": reg,
                             "final_regret": reg[-1], "audit": audit, "six": sixes,
                             "seconds": round(time.time() - t0, 1)})
                arrays[f"{oname}__{seed}__{arm}__regret"] = np.array(reg)
                print(f"  {oname:<12} seed {seed}  {arm:<18} final regret "
                      f"{reg[-1]:12.5g}  {rows[-1]['seconds']:6.1f}s", flush=True)
                np.savez_compressed(ARRAYS / f"{stem}.npz", **arrays)
                with open(RESULTS / f"{stem}.json", "w") as f:
                    json.dump(jsonable({"_config": {
                        "n_init": N_INIT, "T": T_ITER, "n_pool": N_POOL, "seeds": SEEDS,
                        "arms": ARMS, "model": MODEL_ID, "acquisition": "EI (log space)",
                        "conf_level": CONF_LEVEL,
                        "objectives": {k: {"dim": v[1], "f_max": v[2]}
                                       for k, v in OBJECTIVES.items()},
                        "not_run": ["tabicl_v2", "tabswift"],
                        "not_run_reason": ("the Jacobian arm costs 2 x n_pool calls per "
                                           "iteration; 12.4 h for TabICL and 3.9 h for "
                                           "TabSwift on top of the rest")},
                        "rows": rows}), f, indent=2)
    del model
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
