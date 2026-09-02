"""T1.7 -- audit the nano-PFN across training compute. The progress measure.

This is the measurement T1.7 exists for. The nano-PFN's prior is known exactly
because we wrote it, so "in-family" holds by construction with none of the
provenance caveats T0.1 forced on rung A.

REGISTERED PREDICTIONS (plan T1.7)
----------------------------------
  fixed arm   sigma^2 = 1.0 always. Brown applies EXACTLY to this prior, so a
              model converging on its own PPD must converge on a symmetric PSD
              Jacobian. Projected asym should DECAY toward the instrument floor
              as training compute grows.
  mixed arm   sigma^2 ~ log-uniform [0.25, 4]. By T3.1 the exact posterior mean
              for a sigma^2-mixing prior is ITSELF asymmetric, so no amount of
              training drives the violation to zero: projected asym should
              PLATEAU near the P6 class floor.

If both decay, T3.1's class-floor argument is wrong. If neither decays, the model
is not learning the PPD and the compute sweep says so. Either is reportable.

THE INSTRUMENT FLOOR FOR THIS MODEL
-----------------------------------
The nano-PFN runs in float32, so its own quantisation floor is not the float64
3.1e-12 of Phase 1's analytic controls. It is measured here directly: a quantised
wrapped exact GP at the nano-PFN's own measured output quantum, through the same
probe. Without that, "decays toward the instrument floor" has no referent.

Contexts are rung A's, reused from the ladder cache, so the nano-PFN and TabICL
are audited on the SAME contexts and their numbers are directly comparable.

Usage:  python t1_7_audit_nanopfn.py [arm ...]
Writes results/t1_7_nanopfn_audit.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
warnings.filterwarnings("ignore")

from nanopfn import NanoPFN, PFNConfig                                  # noqa: E402
from experiments.phase_1.core.controls import GP_SIGMA, quantize, wrap   # noqa: E402
from experiments.phase_1.core.metrics import (                          # noqa: E402
    asym, get_Q, negeig, reduced_jacobian,
)
from experiments.phase_1.core.surrogates import ExactGP                  # noqa: E402

RES = HERE.parent / "results"
N_CONTEXTS = 20
T = 1e-3
DEVICE = "cuda"


def load_ckpt(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = NanoPFN(PFNConfig(**ck["cfg"])).to(DEVICE)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def audit_one(model, X, y):
    Q = get_Q(y, seed=0)
    n = len(y)

    def predict(y_eval):
        return model.predict_np(X, y_eval, X, device=DEVICE)

    Jr, _, _ = reduced_jacobian(predict, y, Q, T)
    Ja = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = T
        Ja[:, i] = (predict(y + e) - predict(y - e)) / (2 * T)
    return {"asym": float(asym(Jr)), "negeig": float(negeig(Jr)),
            "ambient_asym": float(asym(Ja)),
            "J_minus_I": float(np.linalg.norm(Ja - np.eye(n), "fro")
                               / np.linalg.norm(Ja, "fro")),
            "trJ_over_n": float(np.trace(Ja) / n)}


def measure_quantum(model, X, y, n_points=301):
    """The nano-PFN's own output quantum: median nonzero jump on a fine scan."""
    Q = get_Q(y, seed=0)
    q1 = Q[:, 0]
    ts = np.linspace(-1e-2, 1e-2, n_points)
    P = np.array([model.predict_np(X, y + t * q1, X, device=DEVICE) for t in ts])
    j = np.abs(np.diff(P, axis=0)).ravel()
    nz = j[j > 0]
    return (float(np.median(nz)) if nz.size else 0.0,
            float(np.mean(j == 0)), str(P.dtype))


def main():
    arms = sys.argv[1:] or ["fixed", "mixed"]
    z = np.load(RES / "t1_contexts.npz")
    ctxs = [(z[f"A__X__{k}"], z[f"A__y__{k}"]) for k in range(N_CONTEXTS)]
    out = {"_config": {"n_contexts": N_CONTEXTS, "t": T, "rung": "A",
                       "contexts": "shared with the TabICL ladder"}, "arms": {}}

    print("=" * 96)
    print("T1.7  nano-PFN audit across training compute -- rung A contexts")
    print("=" * 96)

    # ---- the instrument floor for THIS model, measured not assumed
    m0, _ = load_ckpt(sorted(RES.glob("nanopfn_fixed_step*.pt"))[0])
    delta, frac0, dt = measure_quantum(m0, *ctxs[0])
    floors = []
    for X, y in ctxs[:5]:
        Qq = get_Q(y, seed=0)
        p = quantize(wrap(ExactGP(X, sigma=GP_SIGMA).predict), max(delta, 1e-12))
        Jf, _, _ = reduced_jacobian(p, y, Qq, T)
        floors.append((float(asym(Jf)), float(negeig(Jf))))
    fa = float(np.mean([f[0] for f in floors]))
    fn = float(np.mean([f[1] for f in floors]))
    out["instrument_floor"] = {"output_quantum": delta, "frac_zero_jumps": frac0,
                               "dtype": dt, "asym": fa, "negeig": fn}
    print(f"\n  instrument floor for the nano-PFN ({dt}, quantum {delta:.3e}):"
          f"  asym {fa:.3e}   negeig {fn:.3e}")

    for arm in arms:
        ckpts = sorted(RES.glob(f"nanopfn_{arm}_step*.pt"),
                       key=lambda p: int(p.stem.split("step")[1]))
        if not ckpts:
            print(f"\n  arm '{arm}': no checkpoints yet, skipping")
            continue
        print(f"\n  ARM '{arm}'   {len(ckpts)} checkpoints")
        print(f"    {'step':>7}{'NLL':>10}{'asym':>19}{'negeig':>19}"
              f"{'||J-I||':>10}{'trJ/n':>9}{'asym/floor':>12}")
        rows = []
        for cp in ckpts:
            model, ck = load_ckpt(cp)
            t0 = time.time()
            rs = [audit_one(model, X, y) for X, y in ctxs]
            agg = {k: (float(np.mean([r[k] for r in rs])),
                       float(np.std([r[k] for r in rs], ddof=1)))
                   for k in rs[0]}
            rows.append({"step": ck["step"], "nll": ck.get("nll"),
                         **{k: v[0] for k, v in agg.items()},
                         **{f"{k}_sd": v[1] for k, v in agg.items()},
                         "asym_over_floor": agg["asym"][0] / fa if fa > 0 else None,
                         "seconds": time.time() - t0})
            r = rows[-1]
            print(f"    {r['step']:>7}{r['nll']:>10.4f}"
                  f"{r['asym']:>12.4f} +-{r['asym_sd']:<5.3f}"
                  f"{r['negeig']:>12.4f} +-{r['negeig_sd']:<5.3f}"
                  f"{r['J_minus_I']:>10.3f}{r['trJ_over_n']:>9.3f}"
                  f"{r['asym_over_floor']:>12.1f}x", flush=True)
            del model
            torch.cuda.empty_cache()

        first, last = rows[0], rows[-1]
        trend = last["asym"] / first["asym"] if first["asym"] > 0 else None
        out["arms"][arm] = {"rows": rows, "asym_first": first["asym"],
                            "asym_last": last["asym"], "asym_ratio_last_first": trend,
                            "prior_config": ck["prior_config"]}
        print(f"    -> asym {first['asym']:.4f} (step {first['step']}) "
              f"-> {last['asym']:.4f} (step {last['step']})   ratio {trend:.3f}")

    if "fixed" in out["arms"] and "mixed" in out["arms"]:
        f_, m_ = out["arms"]["fixed"], out["arms"]["mixed"]
        print("\n" + "=" * 96)
        print("REGISTERED COMPARISON")
        print("=" * 96)
        print(f"  fixed  sigma^2: asym {f_['asym_first']:.4f} -> {f_['asym_last']:.4f}"
              f"   ratio {f_['asym_ratio_last_first']:.3f}   (predicted: DECAY to the floor)")
        print(f"  mixed  sigma^2: asym {m_['asym_first']:.4f} -> {m_['asym_last']:.4f}"
              f"   ratio {m_['asym_ratio_last_first']:.3f}   (predicted: PLATEAU at the P6 floor)")
        out["comparison"] = {
            "fixed_decays_more_than_mixed":
                bool(f_["asym_ratio_last_first"] < m_["asym_ratio_last_first"]),
            "fixed_final_over_floor": f_["asym_last"] / fa if fa > 0 else None,
            "mixed_final_over_floor": m_["asym_last"] / fa if fa > 0 else None}
        print(f"  fixed decays more than mixed: "
              f"{out['comparison']['fixed_decays_more_than_mixed']}")

    json.dump(out, open(RES / "t1_7_nanopfn_audit.json", "w"), indent=2, default=float)
    print(f"\nSaved: {RES / 't1_7_nanopfn_audit.json'}")


if __name__ == "__main__":
    main()
